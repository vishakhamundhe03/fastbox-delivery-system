"""
FastBox Logistics Simulator
============================
Assignment: Mystery Delivery System
Company  : Nexgensis Technologies Pvt. Ltd.

Author assumptions (documented as required):
--------------------------------------------
1. ASSIGNMENT RULE: Each package is independently assigned to the nearest agent
   based on Euclidean distance from the agent's *starting position* to the
   package's warehouse.  Ties are broken alphabetically by agent ID (A1 < A2).

2. DELIVERY ORDER: Within each agent, packages are delivered in ascending package-ID
   order (P1 before P2, etc.) for deterministic, reproducible results.

3. TRAVEL PATH PER PACKAGE:
       current_position → warehouse → destination
   The agent's current position updates to the destination after each delivery.

4. EFFICIENCY METRIC: total_distance / packages_delivered  (lower = better).

5. BEST AGENT: The agent with the *lowest* efficiency score — i.e., the one who
   covers the least distance per package delivered.

6. RANDOM DELAYS (Bonus): Each delivery incurs a random delay of 0–30 minutes,
   simulating real-world uncertainty (traffic, loading time, etc.).

7. MID-DAY AGENT (Bonus): A new agent can join mid-day and absorb undelivered
   packages.  Re-assignment runs only over the remaining (undelivered) package set.

8. COORDINATE SYSTEM: Flat 2-D Cartesian grid; Euclidean distance throughout.
"""

import json
import math
import random
import csv
import os
import sys
from copy import deepcopy
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_distance(point_a: list[float], point_b: list[float]) -> float:
    """
    Compute the straight-line (Euclidean) distance between two 2-D points.

    Args:
        point_a: [x1, y1]
        point_b: [x2, y2]

    Returns:
        Non-negative float distance.
    """
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def round2(value: float) -> float:
    """Round a float to 2 decimal places (used in all distance outputs)."""
    return round(value, 2)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: JSON Parsing
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> dict:
    """
    Read and parse the input JSON file manually using Python's built-in json module.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed dict with keys: 'warehouses', 'agents', 'packages'.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError       : If required keys are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)  # Manual parsing via stdlib — no third-party dependency

    # Validate required top-level keys
    for key in ("warehouses", "agents", "packages"):
        if key not in data:
            raise ValueError(f"Missing required key in data file: '{key}'")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Agent-Package Assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_packages(warehouses: dict, agents: dict, packages: list) -> dict:
    """
    Assign each package to the nearest agent.

    For every package, iterate over all agents and calculate the Euclidean
    distance from each agent's starting position to the package's warehouse.
    The agent with the minimum distance wins the assignment.

    Tie-breaking: alphabetically by agent ID (assumption #1).

    Args:
        warehouses : dict mapping warehouse ID → [x, y]
        agents     : dict mapping agent ID → [x, y]
        packages   : list of package dicts (keys: id, warehouse, destination)

    Returns:
        dict mapping agent_id → list of package dicts assigned to that agent.
    """
    # Initialise empty delivery queue for every agent
    assignment: dict = {agent_id: [] for agent_id in sorted(agents.keys())}

    for package in packages:
        warehouse_pos = warehouses[package["warehouse"]]

        best_agent = None
        best_dist  = float("inf")

        # Evaluate every agent; sorted() enforces alphabetical tie-breaking
        for agent_id in sorted(agents.keys()):
            agent_pos = agents[agent_id]
            dist      = euclidean_distance(agent_pos, warehouse_pos)

            if dist < best_dist:
                best_dist  = dist
                best_agent = agent_id

        assignment[best_agent].append(package)
        print(f"  [ASSIGN] Package {package['id']} → Agent {best_agent} "
              f"(dist to {package['warehouse']}: {round2(best_dist)} units)")

    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Delivery Simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_deliveries(
    agents_start: dict,
    warehouses: dict,
    assignment: dict,
    enable_delays: bool = False,
    delay_seed: Optional[int] = None
) -> tuple[dict, list]:
    """
    Simulate one full day of deliveries for all agents.

    Each agent:
      1. Starts at their registered position.
      2. For each assigned package (sorted by package ID):
         a. Travels current_pos → warehouse.
         b. Picks up the package (optional random delay applied here).
         c. Travels warehouse → destination.
         d. Updates current_pos to destination.

    Args:
        agents_start  : dict of agent_id → [x, y] starting positions.
        warehouses    : dict of warehouse_id → [x, y].
        assignment    : output of assign_packages().
        enable_delays : If True, simulate random pickup delays (Bonus feature).
        delay_seed    : Optional RNG seed for reproducibility of delays.

    Returns:
        Tuple of:
          - results  dict: agent_id → {packages_delivered, total_distance,
                                        efficiency, delivery_log, total_delay_min}
          - delivery_log list: chronological log of all delivery events.
    """
    if delay_seed is not None:
        random.seed(delay_seed)

    results     = {}
    global_log  = []          # master event log across all agents

    for agent_id in sorted(agents_start.keys()):
        current_pos   = list(agents_start[agent_id])   # mutable copy
        total_distance = 0.0
        total_delay    = 0                              # minutes (bonus)
        agent_log      = []                            # per-agent event list

        # Sort packages by ID for deterministic delivery order (assumption #2)
        packages = sorted(assignment.get(agent_id, []), key=lambda p: p["id"])

        print(f"\n  [SIM] Agent {agent_id} starts at {current_pos}, "
              f"assigned {len(packages)} package(s): {[p['id'] for p in packages]}")

        for pkg in packages:
            warehouse_pos = warehouses[pkg["warehouse"]]
            destination   = pkg["destination"]

            # Leg 1: agent's current position → warehouse
            leg1 = euclidean_distance(current_pos, warehouse_pos)

            # Optional: random delay at pickup (0–30 minutes)
            delay_min = 0
            if enable_delays:
                delay_min    = random.randint(0, 30)
                total_delay += delay_min

            # Leg 2: warehouse → destination
            leg2 = euclidean_distance(warehouse_pos, destination)

            trip_distance  = leg1 + leg2
            total_distance += trip_distance

            event = {
                "package"         : pkg["id"],
                "from"            : current_pos[:],
                "warehouse"       : pkg["warehouse"],
                "warehouse_pos"   : warehouse_pos,
                "destination"     : destination,
                "leg1_dist"       : round2(leg1),
                "leg2_dist"       : round2(leg2),
                "trip_distance"   : round2(trip_distance),
                "delay_minutes"   : delay_min,
                "status"          : "DELIVERED",
            }
            agent_log.append(event)
            global_log.append({**event, "agent": agent_id})

            # Update agent's position to the delivery destination
            current_pos = list(destination)

            print(f"    [PKG {pkg['id']}] {pkg['warehouse']}{warehouse_pos} → "
                  f"dest{destination}  |  leg1={round2(leg1):.2f}  leg2={round2(leg2):.2f}  "
                  f"trip={round2(trip_distance):.2f}"
                  + (f"  delay={delay_min}min" if enable_delays else ""))

        # Guard: handle agents with no packages
        packages_delivered = len(packages)
        efficiency = (
            round2(total_distance / packages_delivered)
            if packages_delivered > 0
            else 0.0
        )

        results[agent_id] = {
            "packages_delivered" : packages_delivered,
            "total_distance"     : round2(total_distance),
            "efficiency"         : efficiency,
            "delivery_log"       : agent_log,
            "total_delay_min"    : total_delay,
        }

        print(f"  [RESULT] Agent {agent_id}: delivered={packages_delivered}  "
              f"distance={round2(total_distance):.2f}  efficiency={efficiency:.2f}"
              + (f"  total_delay={total_delay}min" if enable_delays else ""))

    return results, global_log


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def build_report(results: dict) -> dict:
    """
    Build the final summary report and identify the best (most efficient) agent.

    Best agent = lowest efficiency score (least distance travelled per package).
    Agents with 0 packages are excluded from best-agent consideration.

    Args:
        results: Output of simulate_deliveries().

    Returns:
        dict matching the required report schema, plus 'best_agent' key.
    """
    report = {}

    for agent_id, data in results.items():
        report[agent_id] = {
            "packages_delivered" : data["packages_delivered"],
            "total_distance"     : data["total_distance"],
            "efficiency"         : data["efficiency"],
        }

    # Determine best agent (lowest efficiency, must have delivered ≥1 package)
    eligible = {
        aid: info
        for aid, info in report.items()
        if info["packages_delivered"] > 0
    }

    best_agent = min(eligible, key=lambda aid: eligible[aid]["efficiency"])
    report["best_agent"] = best_agent

    return report


def save_report(report: dict, filepath: str) -> None:
    """
    Persist the report dict to a JSON file with pretty-printing.

    Args:
        report   : Output of build_report().
        filepath : Destination path for report.json.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  [SAVED] Report written to: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Bonus — ASCII Route Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_routes(
    warehouses: dict,
    agents_start: dict,
    assignment: dict,
    grid_width: int = 60,
    grid_height: int = 30,
    x_max: int = 120,
    y_max: int = 100,
) -> None:
    """
    Render a simple ASCII map showing warehouses, agent starting positions,
    and package destinations.

    Coordinate scaling:
        screen_x = int(x / x_max * grid_width)
        screen_y = int((y_max - y) / y_max * grid_height)   ← y-axis flipped

    Legend:
        W1/W2/W3 = warehouses
        A1/A2/A3 = agent starting positions
        *        = package destination
        .        = empty cell
    """
    # Create blank grid filled with '.'
    grid = [["." for _ in range(grid_width + 1)] for _ in range(grid_height + 1)]

    def to_screen(x: float, y: float) -> tuple[int, int]:
        """Convert world coords → screen (col, row)."""
        col = int(x / x_max * grid_width)
        row = int((y_max - y) / y_max * grid_height)
        col = max(0, min(col, grid_width))
        row = max(0, min(row, grid_height))
        return col, row

    def place(label: str, x: float, y: float) -> None:
        """Place a label on the grid (last character if label > 1 char)."""
        col, row = to_screen(x, y)
        # Use the label's first 2 chars so "W1", "A1" fit
        for i, ch in enumerate(label[:2]):
            if col + i <= grid_width:
                grid[row][col + i] = ch

    # Plot warehouses
    for wid, pos in warehouses.items():
        place(wid, pos[0], pos[1])

    # Plot agent starting positions
    for aid, pos in agents_start.items():
        place(aid, pos[0], pos[1])

    # Plot all package destinations with '*'
    for pkg_list in assignment.values():
        for pkg in pkg_list:
            col, row = to_screen(pkg["destination"][0], pkg["destination"][1])
            if grid[row][col] == ".":
                grid[row][col] = "*"

    # Render
    border = "+" + "-" * (grid_width + 1) + "+"
    print("\n  ASCII Route Map  (W=warehouse, A=agent, *=destination)\n")
    print("  " + border)
    for row in grid:
        print("  |" + "".join(row) + "|")
    print("  " + border)
    print("  X-axis: 0 →", x_max, "   Y-axis: 0 ↑", y_max, "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Bonus — New Agent Joining Mid-Day
# ─────────────────────────────────────────────────────────────────────────────

def add_mid_day_agent(
    new_agent_id: str,
    new_agent_pos: list[float],
    agents: dict,
    warehouses: dict,
    current_assignment: dict,
    delivered_ids: set[str],
) -> dict:
    """
    Simulate a new agent joining mid-day and taking over undelivered packages.

    Strategy:
      - Collect all packages not yet delivered.
      - Re-run assignment over the extended agent pool (original + new agent).
      - Return the updated assignment.

    Assumption: Already-delivered packages stay delivered; only remaining ones
    are redistributed. Existing agents' partial progress is not penalised.

    Args:
        new_agent_id    : e.g. "A4"
        new_agent_pos   : [x, y] starting position of the new agent.
        agents          : original agents dict.
        warehouses      : warehouses dict.
        current_assignment : existing package assignment (before mid-day join).
        delivered_ids   : set of package IDs already delivered at join time.

    Returns:
        Updated assignment dict including the new agent.
    """
    # Combine original + new agent
    extended_agents = {**agents, new_agent_id: new_agent_pos}

    # Collect only the undelivered packages
    undelivered = [
        pkg
        for pkg_list in current_assignment.values()
        for pkg in pkg_list
        if pkg["id"] not in delivered_ids
    ]

    if not undelivered:
        print(f"  [MID-DAY] No undelivered packages for {new_agent_id} to take.")
        new_assignment = {**current_assignment, new_agent_id: []}
        return new_assignment

    print(f"\n  [MID-DAY] Agent {new_agent_id} joins at {new_agent_pos}. "
          f"Re-assigning {len(undelivered)} undelivered package(s).")

    # Clear existing assignments for undelivered packages
    base_assignment: dict = {aid: [] for aid in sorted(extended_agents.keys())}

    # Keep already-delivered packages in their agent buckets (for record)
    for aid, pkg_list in current_assignment.items():
        for pkg in pkg_list:
            if pkg["id"] in delivered_ids:
                base_assignment[aid].append(pkg)

    # Re-assign undelivered packages with expanded agent pool
    new_assignment = assign_packages(warehouses, extended_agents, undelivered)
    # Merge with delivered records
    for aid in base_assignment:
        if aid in new_assignment:
            new_assignment[aid] = base_assignment[aid] + new_assignment[aid]
        else:
            new_assignment[aid] = base_assignment[aid]

    return new_assignment


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Bonus — Export Top Performer to CSV
# ─────────────────────────────────────────────────────────────────────────────

def export_top_performer_csv(report: dict, results: dict, filepath: str) -> None:
    """
    Write the best agent's detailed delivery log to a CSV file.

    Columns: package, warehouse, destination_x, destination_y,
             leg1_dist, leg2_dist, trip_distance, delay_minutes, status

    Args:
        report   : Output of build_report() (contains 'best_agent').
        results  : Output of simulate_deliveries() (contains 'delivery_log').
        filepath : Path for the output CSV.
    """
    best_agent = report["best_agent"]
    log        = results[best_agent]["delivery_log"]

    fieldnames = [
        "agent", "package", "warehouse", "warehouse_x", "warehouse_y",
        "destination_x", "destination_y",
        "leg1_dist", "leg2_dist", "trip_distance",
        "delay_minutes", "status",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for event in log:
            writer.writerow({
                "agent"         : best_agent,
                "package"       : event["package"],
                "warehouse"     : event["warehouse"],
                "warehouse_x"   : event["warehouse_pos"][0],
                "warehouse_y"   : event["warehouse_pos"][1],
                "destination_x" : event["destination"][0],
                "destination_y" : event["destination"][1],
                "leg1_dist"     : event["leg1_dist"],
                "leg2_dist"     : event["leg2_dist"],
                "trip_distance" : event["trip_distance"],
                "delay_minutes" : event["delay_minutes"],
                "status"        : event["status"],
            })

    print(f"  [CSV] Top performer ({best_agent}) exported to: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  FastBox Logistics Simulator — One-Day Operations Report")
    print("=" * 65)

    # ── 1. Load and parse input data ─────────────────────────────────────────
    data_file = "data.json"
    print(f"\n[STEP 1] Loading data from '{data_file}' ...")
    data       = load_data(data_file)
    warehouses = data["warehouses"]
    agents     = data["agents"]
    packages   = data["packages"]

    print(f"  Warehouses : {list(warehouses.keys())}")
    print(f"  Agents     : {list(agents.keys())}")
    print(f"  Packages   : {[p['id'] for p in packages]}")

    # Sanity check: total packages must be deliverable
    assert len(packages) > 0, "No packages to deliver!"
    assert len(agents)   > 0, "No agents available!"

    # ── 2. Assign packages to nearest agents ─────────────────────────────────
    print("\n[STEP 2] Assigning packages to nearest agents ...")
    assignment = assign_packages(warehouses, agents, packages)

    for agent_id, pkgs in assignment.items():
        print(f"  → {agent_id} will deliver: {[p['id'] for p in pkgs]}")

    # ── 3. Simulate deliveries (standard run, no delays) ─────────────────────
    print("\n[STEP 3] Simulating deliveries ...")
    results, global_log = simulate_deliveries(
        agents_start  = agents,
        warehouses    = warehouses,
        assignment    = assignment,
        enable_delays = False,
    )

    # Verify all packages are accounted for
    total_delivered = sum(r["packages_delivered"] for r in results.values())
    assert total_delivered == len(packages), (
        f"Mismatch! Delivered {total_delivered} but expected {len(packages)}."
    )
    print(f"\n  ✓ All {total_delivered}/{len(packages)} packages delivered.")

    # ── 4. Generate and save report ───────────────────────────────────────────
    print("\n[STEP 4] Generating report ...")
    report = build_report(results)
    print(f"\n  Final Report:")
    for agent_id, info in report.items():
        if agent_id == "best_agent":
            continue
        print(f"    {agent_id}: delivered={info['packages_delivered']}  "
              f"distance={info['total_distance']:.2f}  "
              f"efficiency={info['efficiency']:.2f} units/pkg")
    print(f"  ★ Best Agent: {report['best_agent']}")

    save_report(report, "report.json")

    # ── BONUS A: ASCII Route Visualisation ───────────────────────────────────
    print("\n[BONUS A] ASCII Route Visualisation:")
    visualise_routes(warehouses, agents, assignment)

    # ── BONUS B: Simulation with Random Delays ───────────────────────────────
    print("[BONUS B] Re-running simulation WITH random delivery delays ...")
    results_delays, _ = simulate_deliveries(
        agents_start  = agents,
        warehouses    = warehouses,
        assignment    = assignment,
        enable_delays = True,
        delay_seed    = 42,   # fixed seed → reproducible
    )
    report_delays = build_report(results_delays)
    save_report(report_delays, "report_with_delays.json")
    print("  [SAVED] Delay report written to: report_with_delays.json")

    for agent_id, data in results_delays.items():
        if isinstance(data, dict) and "total_delay_min" in data:
            print(f"    {agent_id}: distance={data['total_distance']:.2f}  "
                  f"total_delay={data['total_delay_min']} min")

    # ── BONUS C: New Agent Joining Mid-Day ───────────────────────────────────
    print("\n[BONUS C] New agent A4 joins mid-day ...")

    # Simulate: A4 arrives after A3 has already delivered P3
    already_delivered = {"P3"}
    extended_assignment = add_mid_day_agent(
        new_agent_id    = "A4",
        new_agent_pos   = [25, 50],
        agents          = agents,
        warehouses      = warehouses,
        current_assignment = assignment,
        delivered_ids      = already_delivered,
    )

    extended_agents = {**agents, "A4": [25, 50]}
    results_ext, _ = simulate_deliveries(
        agents_start  = extended_agents,
        warehouses    = warehouses,
        assignment    = extended_assignment,
        enable_delays = False,
    )
    report_ext = build_report(results_ext)
    save_report(report_ext, "report_with_new_agent.json")
    print("  [SAVED] New-agent report written to: report_with_new_agent.json")

    # ── BONUS D: Export Top Performer to CSV ─────────────────────────────────
    print("\n[BONUS D] Exporting top performer to CSV ...")
    export_top_performer_csv(report, results, "top_performer.csv")

    print("\n" + "=" * 65)
    print("  Simulation complete. Output files:")
    print("    report.json                — primary delivery report")
    print("    report_with_delays.json    — report with random delays")
    print("    report_with_new_agent.json — report with mid-day agent A4")
    print("    top_performer.csv          — best agent's delivery log")
    print("=" * 65)


if __name__ == "__main__":
    main()
