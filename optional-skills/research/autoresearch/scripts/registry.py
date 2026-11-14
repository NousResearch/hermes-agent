#!/usr/bin/env python3
"""Track durable autoresearch runs in the active Hermes profile."""

from __future__ import annotations

import argparse
import json
import re
from functools import wraps
from pathlib import Path
from typing import Any

from _util import atomic_write, exclusive_lock, hermes_home, now_iso

RESEARCH_ID_PATTERN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?\Z"
)
ACTIVE_PHASES = {"starting", "planning", "executing"}
UPDATABLE_FIELDS = {"phase", "cron_job_id", "watchdog_job_id"}
RESERVED_WINDOWS_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
PHASE_TRANSITIONS = {
    "starting": {"planning", "executing", "paused", "stopped", "failed"},
    "planning": {"executing", "paused", "stopped", "failed"},
    "executing": {"paused", "completed", "stopped", "failed"},
    "paused": {"executing", "stopped", "failed"},
    "completed": set(),
    "stopped": set(),
    "failed": set(),
}


class RegistryError(ValueError):
    """Raised when registry data or input is invalid."""


def locked_registry(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        with exclusive_lock(Path(hermes_home()) / "autoresearch" / ".registry.lock"):
            return function(*args, **kwargs)

    return wrapper


def validate_research_id(research_id: str) -> str:
    stem = research_id.split(".", 1)[0].upper()
    if not RESEARCH_ID_PATTERN.fullmatch(research_id) or stem in RESERVED_WINDOWS_NAMES:
        raise RegistryError(
            "research_id must be 1-128 letters, digits, dots, underscores, or hyphens"
        )
    return research_id


def registry_path() -> Path:
    return Path(hermes_home()) / "autoresearch" / "registry.json"


def expected_run_dir(research_id: str) -> Path:
    root = (Path(hermes_home()) / "autoresearch").resolve()
    candidate = root / validate_research_id(research_id)
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise RegistryError(f"run directory escapes autoresearch root: {research_id}")
    return resolved


def validate_stored_run(research_id: str, run: dict[str, Any]) -> Path:
    expected = expected_run_dir(research_id)
    stored = Path(str(run.get("run_dir", ""))).expanduser().resolve()
    if stored != expected or run.get("research_id") != research_id:
        raise RegistryError(f"registry path mismatch for run: {research_id}")
    return expected


def read_registry() -> dict[str, Any]:
    path = registry_path()
    if not path.exists():
        return {"runs": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"invalid registry JSON: {path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("runs"), dict):
        raise RegistryError(f"invalid registry structure: {path}")
    return data


@locked_registry
def register(
    research_id: str,
    user_id: str,
    platform: str,
    chat_id: str,
    goal: str,
    cron_job_id: str,
    watchdog_job_id: str | None = None,
) -> dict[str, Any]:
    research_id = validate_research_id(research_id)
    registry = read_registry()
    if research_id in registry["runs"]:
        raise RegistryError(f"research run already registered: {research_id}")
    run_dir = expected_run_dir(research_id)
    if not (run_dir / "status.json").is_file():
        raise RegistryError(f"research run is not initialized: {run_dir}")
    registry["runs"][research_id] = {
        "research_id": research_id,
        "user_id": user_id,
        "platform": platform,
        "chat_id": chat_id,
        "goal": goal,
        "cron_job_id": cron_job_id,
        "watchdog_job_id": watchdog_job_id,
        "run_dir": str(run_dir),
        "created": now_iso(),
        "phase": "starting",
    }
    atomic_write(registry_path(), registry)
    return {"status": "registered", "research_id": research_id}


def status_for_run(research_id: str, run: dict[str, Any]) -> dict[str, Any] | None:
    run_dir = validate_stored_run(research_id, run)
    path = run_dir / "status.json"
    if path.exists() and path.resolve().parent != run_dir:
        raise RegistryError(f"run status escapes run directory: {path}")
    if not path.exists():
        return None
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"invalid run status JSON: {path}") from exc
    if not isinstance(status, dict):
        raise RegistryError(f"invalid run status structure: {path}")
    return status


def list_runs(
    user_id: str | None = None,
    platform: str | None = None,
    active_only: bool = False,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for research_id, stored in read_registry()["runs"].items():
        validate_research_id(research_id)
        run = dict(stored)
        status = status_for_run(research_id, run)
        if status:
            run["phase"] = status.get("phase", "unknown")
            run["experiments_done"] = status.get("experiments_done", 0)
            run["experiments_total"] = status.get("experiments_total", 0)
        if user_id and run.get("user_id") != user_id:
            continue
        if platform and run.get("platform") != platform:
            continue
        if active_only and run.get("phase") not in ACTIVE_PHASES:
            continue
        runs.append(run)
    return {"count": len(runs), "runs": runs}


def get_run(research_id: str) -> dict[str, Any]:
    research_id = validate_research_id(research_id)
    run = read_registry()["runs"].get(research_id)
    if not run:
        raise RegistryError(f"research run not found: {research_id}")
    result = dict(run)
    result["status"] = status_for_run(research_id, result)
    return result


@locked_registry
def update_run(research_id: str, updates: dict[str, str | None]) -> dict[str, Any]:
    research_id = validate_research_id(research_id)
    unknown = set(updates) - UPDATABLE_FIELDS
    if unknown:
        raise RegistryError(f"unsupported update fields: {', '.join(sorted(unknown))}")
    registry = read_registry()
    if research_id not in registry["runs"]:
        raise RegistryError(f"research run not found: {research_id}")
    run = registry["runs"][research_id]
    validate_stored_run(research_id, run)
    requested_phase = updates.get("phase")
    if requested_phase is not None:
        if requested_phase not in PHASE_TRANSITIONS:
            raise RegistryError(f"invalid phase: {requested_phase}")
        current_phase = str(run.get("phase", "starting"))
        if (
            requested_phase != current_phase
            and requested_phase not in PHASE_TRANSITIONS.get(current_phase, set())
        ):
            raise RegistryError(
                f"invalid phase transition: {current_phase} -> {requested_phase}"
            )
        status = status_for_run(research_id, run)
        if status and status.get("phase") != requested_phase:
            raise RegistryError("registry phase must match persisted run status")
    for key, value in updates.items():
        if value is not None:
            registry["runs"][research_id][key] = value
    atomic_write(registry_path(), registry)
    return {"status": "updated", "research_id": research_id}


@locked_registry
def remove_run(research_id: str) -> dict[str, Any]:
    research_id = validate_research_id(research_id)
    registry = read_registry()
    if research_id not in registry["runs"]:
        raise RegistryError(f"research run not found: {research_id}")
    del registry["runs"][research_id]
    atomic_write(registry_path(), registry)
    return {"status": "removed", "research_id": research_id}


def find_by_job(job_id: str) -> dict[str, Any]:
    for research_id, run in read_registry()["runs"].items():
        validate_stored_run(research_id, run)
        if run.get("cron_job_id") == job_id or run.get("watchdog_job_id") == job_id:
            return dict(run)
    raise RegistryError(f"no research run for job: {job_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    register_parser = commands.add_parser("register")
    register_parser.add_argument("research_id")
    register_parser.add_argument("user_id")
    register_parser.add_argument("platform")
    register_parser.add_argument("chat_id")
    register_parser.add_argument("goal")
    register_parser.add_argument("cron_job_id")
    register_parser.add_argument("--watchdog-job-id")

    list_parser = commands.add_parser("list")
    list_parser.add_argument("--user-id")
    list_parser.add_argument("--platform")
    list_parser.add_argument("--active-only", action="store_true")

    get_parser = commands.add_parser("get")
    get_parser.add_argument("research_id")

    update_parser = commands.add_parser("update")
    update_parser.add_argument("research_id")
    update_parser.add_argument("--phase")
    update_parser.add_argument("--cron-job-id")
    update_parser.add_argument("--watchdog-job-id")

    remove_parser = commands.add_parser("remove")
    remove_parser.add_argument("research_id")

    find_parser = commands.add_parser("find-by-job")
    find_parser.add_argument("job_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            result = register(
                args.research_id,
                args.user_id,
                args.platform,
                args.chat_id,
                args.goal,
                args.cron_job_id,
                args.watchdog_job_id,
            )
        elif args.command == "list":
            result = list_runs(args.user_id, args.platform, args.active_only)
        elif args.command == "get":
            result = get_run(args.research_id)
        elif args.command == "update":
            result = update_run(
                args.research_id,
                {
                    "phase": args.phase,
                    "cron_job_id": args.cron_job_id,
                    "watchdog_job_id": args.watchdog_job_id,
                },
            )
        elif args.command == "remove":
            result = remove_run(args.research_id)
        else:
            result = find_by_job(args.job_id)
        print(json.dumps(result, indent=2))
    except RegistryError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
