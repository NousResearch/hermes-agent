#!/usr/bin/env python3
"""Check declared capacity-planning constraints using only the Python stdlib."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _number(value: Any, label: str, errors: list[str], *, positive: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label} must be a number")
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        qualifier = "positive and finite" if positive else "non-negative and finite"
        errors.append(f"{label} must be {qualifier}")
        return None
    return result


def evaluate(problem: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    capacities = problem.get("zone_capacity_kw")
    workloads = problem.get("workloads")
    placements = plan.get("placements")
    budget = _number(problem.get("budget_limit"), "budget_limit", errors)

    if not isinstance(capacities, dict) or not capacities:
        errors.append("zone_capacity_kw must be a non-empty object")
        capacities = {}
    normalized_capacities: dict[str, float] = {}
    for zone, value in capacities.items():
        if not isinstance(zone, str) or not zone:
            errors.append("zone names must be non-empty strings")
            continue
        number = _number(value, f"zone_capacity_kw[{zone}]", errors)
        if number is not None:
            normalized_capacities[zone] = number

    if not isinstance(workloads, list) or not workloads:
        errors.append("workloads must be a non-empty array")
        workloads = []
    workload_specs: dict[str, tuple[float, int]] = {}
    for index, workload in enumerate(workloads):
        label = f"workloads[{index}]"
        if not isinstance(workload, dict):
            errors.append(f"{label} must be an object")
            continue
        workload_id = workload.get("id")
        if not isinstance(workload_id, str) or not workload_id:
            errors.append(f"{label}.id must be a non-empty string")
            continue
        if workload_id in workload_specs:
            errors.append(f"duplicate workload id: {workload_id}")
            continue
        required_kw = _number(workload.get("required_kw"), f"{label}.required_kw", errors, positive=True)
        min_zones = workload.get("min_zones")
        if isinstance(min_zones, bool) or not isinstance(min_zones, int) or min_zones < 1:
            errors.append(f"{label}.min_zones must be a positive integer")
            continue
        if required_kw is not None:
            workload_specs[workload_id] = (required_kw, min_zones)

    if not isinstance(placements, list):
        errors.append("placements must be an array")
        placements = []

    workload_kw = {workload_id: 0.0 for workload_id in workload_specs}
    workload_zones = {workload_id: set() for workload_id in workload_specs}
    zone_kw = {zone: 0.0 for zone in normalized_capacities}
    total_cost = 0.0
    seen_pairs: set[tuple[str, str]] = set()
    for index, placement in enumerate(placements):
        label = f"placements[{index}]"
        if not isinstance(placement, dict):
            errors.append(f"{label} must be an object")
            continue
        workload_id, zone = placement.get("workload_id"), placement.get("zone")
        if not isinstance(workload_id, str) or not workload_id:
            errors.append(f"{label}.workload_id must be a non-empty string")
            continue
        if workload_id not in workload_specs:
            errors.append(f"{label} references unknown workload: {workload_id}")
            continue
        if not isinstance(zone, str) or not zone:
            errors.append(f"{label}.zone must be a non-empty string")
            continue
        if zone not in normalized_capacities:
            errors.append(f"{label} references unknown zone: {zone}")
            continue
        pair = (workload_id, zone)
        if pair in seen_pairs:
            errors.append(f"duplicate placement for workload {workload_id} in {zone}")
            continue
        seen_pairs.add(pair)
        kw = _number(placement.get("kw"), f"{label}.kw", errors, positive=True)
        cost = _number(placement.get("monthly_cost"), f"{label}.monthly_cost", errors)
        if kw is not None:
            workload_kw[workload_id] += kw
            workload_zones[workload_id].add(zone)
            zone_kw[zone] += kw
        if cost is not None:
            total_cost += cost

    for workload_id, (required_kw, min_zones) in workload_specs.items():
        if workload_kw[workload_id] < required_kw:
            errors.append(f"workload {workload_id} requires {required_kw:g} kW; assigned {workload_kw[workload_id]:g} kW")
        if len(workload_zones[workload_id]) < min_zones:
            errors.append(f"workload {workload_id} requires {min_zones} distinct zones; assigned {len(workload_zones[workload_id])}")
    for zone, assigned in zone_kw.items():
        if assigned > normalized_capacities[zone]:
            errors.append(f"zone {zone} capacity exceeded: {assigned:g} kW > {normalized_capacities[zone]:g} kW")
    if budget is not None and total_cost > budget:
        errors.append(f"budget exceeded: {total_cost:g} > {budget:g}")

    return {
        "feasible": not errors,
        "errors": errors,
        "summary": {
            "total_monthly_cost": round(total_cost, 6),
            "zone_kw": {zone: round(value, 6) for zone, value in zone_kw.items()},
            "workload_kw": {workload_id: round(value, 6) for workload_id, value in workload_kw.items()},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    try:
        problem = json.loads(args.problem.read_text(encoding="utf-8"))
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        if not isinstance(problem, dict) or not isinstance(plan, dict):
            raise ValueError("problem and plan files must each contain a JSON object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"feasible": False, "errors": [str(exc)]}, indent=2))
        return 2
    result = evaluate(problem, plan)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["feasible"] else 1


if __name__ == "__main__":
    sys.exit(main())
