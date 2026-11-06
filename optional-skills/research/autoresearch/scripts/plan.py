#!/usr/bin/env python3
"""Create and update a bounded autoresearch experiment plan."""

from __future__ import annotations

import argparse
import json
from functools import wraps
from pathlib import Path
from typing import Any

from _util import atomic_write, exclusive_lock, now_iso, read_json

VALID_TYPES = ("investigate", "deepen", "verify", "synthesize")
VALID_STATUSES = ("pending", "in_progress", "merged", "reverted", "failed")
ALLOWED_TRANSITIONS = {
    "pending": {"in_progress", "failed"},
    "in_progress": {"merged", "reverted", "failed"},
    "merged": set(),
    "reverted": set(),
    "failed": set(),
}


class PlanError(ValueError):
    """Raised when plan data violates its contract."""


def locked_run(function):
    @wraps(function)
    def wrapper(run_dir, *args, **kwargs):
        with exclusive_lock(Path(run_dir) / ".autoresearch.lock"):
            return function(run_dir, *args, **kwargs)

    return wrapper


def plan_path(run_dir: str) -> Path:
    return Path(run_dir) / "plan.json"


def read_required(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if not data:
        raise PlanError(f"missing or invalid JSON: {path}")
    return data


def hard_cap(run_dir: str) -> int:
    config = read_required(Path(run_dir) / "config.json")
    return int(config["max_experiments_hard_cap"])


def normalize_experiments(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise PlanError("experiments must be a JSON array")
    experiments: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise PlanError("every experiment must be an object")
        experiment = dict(item)
        experiment_id = experiment.get("id")
        if not isinstance(experiment_id, int) or isinstance(experiment_id, bool) or experiment_id <= 0:
            raise PlanError("experiment id must be a positive integer")
        if experiment_id in seen_ids:
            raise PlanError(f"duplicate experiment id: {experiment_id}")
        seen_ids.add(experiment_id)
        experiment_type = experiment.get("type", "investigate")
        if experiment_type not in VALID_TYPES:
            raise PlanError(f"invalid experiment type: {experiment_type}")
        hypothesis = experiment.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise PlanError(f"experiment {experiment_id} needs a hypothesis")
        experiment["type"] = experiment_type
        experiment["hypothesis"] = hypothesis.strip()
        experiment["target_section"] = str(experiment.get("target_section", ""))
        experiment["status"] = experiment.get("status", "pending")
        if experiment["status"] not in VALID_STATUSES:
            raise PlanError(f"invalid experiment status: {experiment['status']}")
        experiments.append(experiment)
    return experiments


@locked_run
def write_plan(run_dir: str, experiments_json: str) -> dict[str, Any]:
    try:
        raw = json.loads(experiments_json)
    except json.JSONDecodeError as exc:
        raise PlanError(f"invalid experiments JSON: {exc.msg}") from exc
    experiments = normalize_experiments(raw)
    existing_plan = read_plan(run_dir)
    existing_by_id = {
        int(experiment["id"]): experiment
        for experiment in existing_plan.get("experiments", [])
    }
    incoming_by_id = {int(experiment["id"]): experiment for experiment in experiments}
    protected_statuses = {"in_progress", "merged", "reverted", "failed"}
    for experiment_id, existing in existing_by_id.items():
        if existing.get("status") not in protected_statuses:
            continue
        incoming = incoming_by_id.get(experiment_id)
        if incoming is None:
            raise PlanError(f"cannot remove experiment history: {experiment_id}")
        stable_fields = ("id", "type", "hypothesis", "target_section", "status")
        if any(incoming.get(field) != existing.get(field) for field in stable_fields):
            raise PlanError(f"cannot rewrite experiment history: {experiment_id}")
        incoming.clear()
        incoming.update(existing)
    for experiment in experiments:
        existing = existing_by_id.get(int(experiment["id"]))
        if existing is None and experiment["status"] != "pending":
            raise PlanError("new experiments must start in pending status")
        if existing is not None and existing.get("status") == "pending" and experiment["status"] != "pending":
            raise PlanError("pending experiments must change status through update-experiment")
    cap = hard_cap(run_dir)
    if len(experiments) > cap:
        raise PlanError(f"plan has {len(experiments)} experiments; hard cap is {cap}")
    if any(int(experiment["id"]) > cap for experiment in experiments):
        raise PlanError(f"experiment ids must be between 1 and the hard cap ({cap})")
    timestamp = now_iso()
    atomic_write(
        plan_path(run_dir),
        {
            "experiments": experiments,
            "created": existing_plan.get("created") or timestamp,
            "last_updated": timestamp,
        },
    )
    return {"status": "plan_written", "count": len(experiments), "hard_cap": cap}


def read_plan(run_dir: str) -> dict[str, Any]:
    return read_required(plan_path(run_dir))


@locked_run
def update_experiment(
    run_dir: str, experiment_id: int, status: str, reason: str | None = None
) -> dict[str, Any]:
    data = read_plan(run_dir)
    for experiment in data.get("experiments", []):
        if experiment["id"] == experiment_id:
            current_status = str(experiment.get("status", "pending"))
            if status == current_status:
                return {
                    "status": "unchanged",
                    "experiment_id": experiment_id,
                    "new_status": status,
                }
            if status not in ALLOWED_TRANSITIONS[current_status]:
                raise PlanError(f"invalid status transition: {current_status} -> {status}")
            experiment["status"] = status
            experiment["updated"] = now_iso()
            if reason:
                experiment["reason"] = reason
            data["last_updated"] = now_iso()
            atomic_write(plan_path(run_dir), data)
            return {
                "status": "updated",
                "experiment_id": experiment_id,
                "new_status": status,
            }
    raise PlanError(f"experiment not found: {experiment_id}")


@locked_run
def add_experiment(
    run_dir: str, experiment_type: str, hypothesis: str, target_section: str
) -> dict[str, Any]:
    data = read_plan(run_dir)
    experiments = data.get("experiments", [])
    cap = hard_cap(run_dir)
    if len(experiments) >= cap:
        raise PlanError(f"cannot add experiment; hard cap is {cap}")
    max_id = max((int(experiment["id"]) for experiment in experiments), default=0)
    new_experiment = {
        "id": max_id + 1,
        "type": experiment_type,
        "hypothesis": hypothesis.strip(),
        "target_section": target_section,
        "status": "pending",
        "added_during_run": True,
    }
    normalized = normalize_experiments([new_experiment])[0]
    experiments.append(normalized)
    data["experiments"] = experiments
    data["last_updated"] = now_iso()
    atomic_write(plan_path(run_dir), data)
    return {"status": "added", "experiment": normalized, "hard_cap": cap}


def next_pending(run_dir: str) -> dict[str, Any]:
    for experiment in read_plan(run_dir).get("experiments", []):
        if experiment["status"] == "pending":
            return experiment
    return {"status": "all_done"}


def summarize(run_dir: str) -> dict[str, Any]:
    experiments = read_plan(run_dir).get("experiments", [])
    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for experiment in experiments:
        status = str(experiment.get("status", "unknown"))
        experiment_type = str(experiment.get("type", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
        by_type[experiment_type] = by_type.get(experiment_type, 0) + 1
    return {"total": len(experiments), "by_status": by_status, "by_type": by_type}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    write_parser = commands.add_parser("write")
    write_parser.add_argument("run_dir")
    write_parser.add_argument("experiments_json")

    read_parser = commands.add_parser("read")
    read_parser.add_argument("run_dir")

    update_parser = commands.add_parser("update-experiment")
    update_parser.add_argument("run_dir")
    update_parser.add_argument("experiment_id", type=int)
    update_parser.add_argument("status", choices=VALID_STATUSES)
    update_parser.add_argument("--reason")

    add_parser = commands.add_parser("add-experiment")
    add_parser.add_argument("run_dir")
    add_parser.add_argument("type", choices=VALID_TYPES)
    add_parser.add_argument("hypothesis")
    add_parser.add_argument("target_section", nargs="?", default="")

    next_parser = commands.add_parser("next-pending")
    next_parser.add_argument("run_dir")

    summary_parser = commands.add_parser("summary")
    summary_parser.add_argument("run_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "write":
            result = write_plan(args.run_dir, args.experiments_json)
        elif args.command == "read":
            result = read_plan(args.run_dir)
        elif args.command == "update-experiment":
            result = update_experiment(
                args.run_dir, args.experiment_id, args.status, args.reason
            )
        elif args.command == "add-experiment":
            result = add_experiment(
                args.run_dir, args.type, args.hypothesis, args.target_section
            )
        elif args.command == "next-pending":
            result = next_pending(args.run_dir)
        else:
            result = summarize(args.run_dir)
        print(json.dumps(result, indent=2))
    except PlanError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
