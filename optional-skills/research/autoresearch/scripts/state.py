#!/usr/bin/env python3
"""Manage persistent state for bounded autoresearch runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

from _util import atomic_write, exclusive_lock, now_iso, read_json

DEFAULT_MAX_DURATION_MINUTES = 180
DEFAULT_MAX_EXPERIMENTS = 10
VALID_CONTROL_ACTIONS = ("none", "pause", "resume", "stop", "adjust")
VALID_PHASES = (
    "planning",
    "executing",
    "paused",
    "completed",
    "stopped",
    "failed",
)
PHASE_TRANSITIONS = {
    "planning": {"executing", "paused", "stopped", "failed"},
    "executing": {"paused", "completed", "stopped", "failed"},
    "paused": {"executing", "stopped", "failed"},
    "completed": set(),
    "stopped": set(),
    "failed": set(),
}


def locked_run(function):
    @wraps(function)
    def wrapper(run_dir, *args, **kwargs):
        with exclusive_lock(Path(run_dir) / ".autoresearch.lock"):
            return function(run_dir, *args, **kwargs)

    return wrapper


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def gen_id(domain: str) -> str:
    digest = hashlib.sha256(
        f"{domain}{now_iso()}{os.getpid()}".encode("utf-8")
    ).hexdigest()[:6]
    safe_domain = re.sub(r"[^a-z0-9]+", "-", domain.lower()).strip("-")
    safe_domain = safe_domain[:32] or "general"
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{safe_domain}_{digest}_{date}"


@locked_run
def init_run(
    run_dir: str,
    goal: str,
    domain: str,
    scope: str,
    max_experiments: int = DEFAULT_MAX_EXPERIMENTS,
    max_duration_minutes: int = DEFAULT_MAX_DURATION_MINUTES,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    state_paths = [
        run_path / "config.json",
        run_path / "status.json",
        run_path / "control.json",
        run_path / "plan.json",
        run_path / "results.log",
    ]
    if any(path.exists() for path in state_paths):
        raise ValueError(f"run already initialized: {run_path}")
    workspace = run_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    config = {
        "goal": goal,
        "domain": domain,
        "scope": scope,
        "max_experiments": max_experiments,
        "max_experiments_hard_cap": max_experiments,
        "max_duration_minutes": max_duration_minutes,
        "created": now_iso(),
    }
    atomic_write(run_path / "config.json", config)
    atomic_write(
        run_path / "status.json",
        {
            "phase": "planning",
            "experiments_done": 0,
            "experiments_total": 0,
            "experiments_merged": 0,
            "experiments_reverted": 0,
            "experiments_failed": 0,
            "current_experiment": None,
            "last_updated": now_iso(),
        },
    )
    atomic_write(run_path / "control.json", {"action": "none"})
    atomic_write(run_path / "plan.json", {"experiments": []})
    (run_path / "results.log").touch()
    return {
        "status": "initialized",
        "run_dir": str(run_path),
        "workspace": str(workspace),
    }


def read_required_json(run_dir: str, filename: str) -> dict[str, Any]:
    path = Path(run_dir) / filename
    data = read_json(path)
    if not data:
        raise FileNotFoundError(f"{filename} is missing or invalid in {run_dir}")
    return data


def read_status(run_dir: str) -> dict[str, Any]:
    return read_required_json(run_dir, "status.json")


@locked_run
def update_status(run_dir: str, phase: str, **updates: Any) -> dict[str, Any]:
    path = Path(run_dir) / "status.json"
    status = read_required_json(run_dir, "status.json")
    previous = dict(status)
    current_phase = str(previous.get("phase", "planning"))
    if phase != current_phase and phase not in PHASE_TRANSITIONS.get(current_phase, set()):
        raise ValueError(f"invalid phase transition: {current_phase} -> {phase}")
    status["phase"] = phase
    status["last_updated"] = now_iso()
    aliases = {
        "merged": "experiments_merged",
        "reverted": "experiments_reverted",
    }
    for key, value in updates.items():
        if value is not None:
            status[aliases.get(key, key)] = value
    config = read_required_json(run_dir, "config.json")
    hard_cap = int(config["max_experiments_hard_cap"])
    bounded_fields = (
        "experiments_done",
        "experiments_total",
        "experiments_merged",
        "experiments_reverted",
        "experiments_failed",
    )
    for field in bounded_fields:
        value = int(status.get(field, 0))
        if value < 0 or value > hard_cap:
            raise ValueError(f"{field} must be between 0 and {hard_cap}")
        if value < int(previous.get(field, 0)):
            raise ValueError(f"{field} cannot move backwards")
    outcomes = sum(
        int(status.get(field, 0))
        for field in ("experiments_merged", "experiments_reverted", "experiments_failed")
    )
    if outcomes > int(status.get("experiments_done", 0)):
        raise ValueError("merged + reverted + failed cannot exceed experiments_done")
    plan = read_required_json(run_dir, "plan.json")
    experiments = plan.get("experiments", [])
    expected = {
        "experiments_merged": sum(
            experiment.get("status") == "merged" for experiment in experiments
        ),
        "experiments_reverted": sum(
            experiment.get("status") == "reverted" for experiment in experiments
        ),
        "experiments_failed": sum(
            experiment.get("status") == "failed" for experiment in experiments
        ),
    }
    if updates or phase in {"completed", "stopped", "failed"}:
        for field, value in expected.items():
            if int(status.get(field, 0)) != value:
                raise ValueError(f"{field} must match persisted plan state ({value})")
        expected_done = sum(expected.values())
        if int(status.get("experiments_done", 0)) != expected_done:
            raise ValueError(
                f"experiments_done must match terminal plan entries ({expected_done})"
            )
        if phase in {"executing", "paused", "completed", "stopped", "failed"}:
            if int(status.get("experiments_total", 0)) != len(experiments):
                raise ValueError(
                    f"experiments_total must match persisted plan size ({len(experiments)})"
                )
    atomic_write(path, status)
    return status


@locked_run
def write_control(
    run_dir: str, action: str = "none", addendum: str | None = None
) -> dict[str, Any]:
    read_required_json(run_dir, "config.json")
    read_required_json(run_dir, "status.json")
    control = {"action": action, "timestamp": now_iso()}
    if addendum:
        control["addendum"] = addendum
    atomic_write(Path(run_dir) / "control.json", control)
    return control


def read_control(run_dir: str) -> dict[str, Any]:
    return read_required_json(run_dir, "control.json")


@locked_run
def write_checkpoint(run_dir: str, last_completed: int, next_experiment: int) -> dict[str, Any]:
    status = read_required_json(run_dir, "status.json")
    config = read_required_json(run_dir, "config.json")
    hard_cap = int(config["max_experiments_hard_cap"])
    if last_completed < 0 or last_completed > int(status.get("experiments_done", 0)):
        raise ValueError("last_completed must be between 0 and experiments_done")
    if next_experiment < 1 or next_experiment > hard_cap + 1:
        raise ValueError(f"next_experiment must be between 1 and {hard_cap + 1}")
    plan = read_required_json(run_dir, "plan.json")
    by_id = {
        int(experiment["id"]): experiment
        for experiment in plan.get("experiments", [])
    }
    terminal = {"merged", "reverted", "failed"}
    if last_completed and (
        last_completed not in by_id or by_id[last_completed].get("status") not in terminal
    ):
        raise ValueError("last_completed must reference a terminal experiment")
    if next_experiment <= hard_cap:
        if (
            next_experiment not in by_id
            or by_id[next_experiment].get("status") != "pending"
        ):
            raise ValueError("next_experiment must reference a pending experiment")
    elif any(experiment.get("status") == "pending" for experiment in by_id.values()):
        raise ValueError("next_experiment may exceed the cap only when no pending work remains")
    checkpoint = {
        "last_completed": last_completed,
        "next": next_experiment,
        "timestamp": now_iso(),
    }
    atomic_write(Path(run_dir) / "checkpoint.json", checkpoint)
    return checkpoint


def read_checkpoint(run_dir: str) -> dict[str, Any]:
    checkpoint = read_json(Path(run_dir) / "checkpoint.json")
    return checkpoint or {"error": "no checkpoint"}


def check_limits(run_dir: str) -> dict[str, Any]:
    config = read_required_json(run_dir, "config.json")
    status = read_required_json(run_dir, "status.json")
    max_duration = int(config["max_duration_minutes"])
    hard_cap = int(config["max_experiments_hard_cap"])
    experiments_done = int(status.get("experiments_done", 0))
    violations: list[str] = []

    created = datetime.fromisoformat(str(config["created"]))
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    elapsed_minutes = (datetime.now(timezone.utc) - created).total_seconds() / 60
    if elapsed_minutes > max_duration:
        violations.append(
            f"time_exceeded: {elapsed_minutes:.0f}min > {max_duration}min"
        )
    if experiments_done >= hard_cap:
        violations.append(
            f"experiments_exceeded: {experiments_done} >= {hard_cap}"
        )

    return {
        "exceeded": bool(violations),
        "violations": violations,
        "limits": {
            "max_duration_minutes": max_duration,
            "max_experiments_hard_cap": hard_cap,
        },
        "current": {
            "experiments_done": experiments_done,
            "elapsed_minutes": round(elapsed_minutes, 2),
        },
    }


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    gen_id_parser = commands.add_parser("gen-id")
    gen_id_parser.add_argument("domain", nargs="?", default="general")

    init_parser = commands.add_parser("init")
    init_parser.add_argument("run_dir")
    init_parser.add_argument("goal")
    init_parser.add_argument("domain")
    init_parser.add_argument("scope")
    init_parser.add_argument("max_experiments", type=positive_int)
    init_parser.add_argument(
        "--max-duration",
        type=positive_int,
        default=DEFAULT_MAX_DURATION_MINUTES,
        dest="max_duration_minutes",
    )

    status_parser = commands.add_parser("status")
    status_parser.add_argument("run_dir")

    update_parser = commands.add_parser("update-status")
    update_parser.add_argument("run_dir")
    update_parser.add_argument("phase", choices=VALID_PHASES)
    update_parser.add_argument("--experiments-done", type=nonnegative_int)
    update_parser.add_argument("--experiments-total", type=nonnegative_int)
    update_parser.add_argument("--experiments-failed", type=nonnegative_int)
    update_parser.add_argument("--current-experiment", type=positive_int)
    update_parser.add_argument("--merged", type=nonnegative_int)
    update_parser.add_argument("--reverted", type=nonnegative_int)

    control_parser = commands.add_parser("control")
    control_parser.add_argument("run_dir")
    control_parser.add_argument("--action", choices=VALID_CONTROL_ACTIONS, default="none")
    control_parser.add_argument("--addendum")

    read_control_parser = commands.add_parser("read-control")
    read_control_parser.add_argument("run_dir")

    checkpoint_parser = commands.add_parser("checkpoint")
    checkpoint_parser.add_argument("run_dir")
    checkpoint_parser.add_argument("last_completed", type=nonnegative_int)
    checkpoint_parser.add_argument("next_experiment", type=positive_int)

    read_checkpoint_parser = commands.add_parser("read-checkpoint")
    read_checkpoint_parser.add_argument("run_dir")

    limits_parser = commands.add_parser("check-limits")
    limits_parser.add_argument("run_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "gen-id":
            print(gen_id(args.domain))
        elif args.command == "init":
            print_json(
                init_run(
                    args.run_dir,
                    args.goal,
                    args.domain,
                    args.scope,
                    args.max_experiments,
                    args.max_duration_minutes,
                )
            )
        elif args.command == "status":
            print_json(read_status(args.run_dir))
        elif args.command == "update-status":
            updates = {
                key: value
                for key, value in vars(args).items()
                if key
                in {
                    "experiments_done",
                    "experiments_total",
                    "experiments_failed",
                    "current_experiment",
                    "merged",
                    "reverted",
                }
            }
            print_json(update_status(args.run_dir, args.phase, **updates))
        elif args.command == "control":
            print_json(write_control(args.run_dir, args.action, args.addendum))
        elif args.command == "read-control":
            print_json(read_control(args.run_dir))
        elif args.command == "checkpoint":
            print_json(
                write_checkpoint(args.run_dir, args.last_completed, args.next_experiment)
            )
        elif args.command == "read-checkpoint":
            print_json(read_checkpoint(args.run_dir))
        elif args.command == "check-limits":
            print_json(check_limits(args.run_dir))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser = build_parser()
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
