#!/usr/bin/env python3
"""Report retrospective autoresearch token and cost telemetry."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from _util import atomic_write, exclusive_lock, hermes_home, now_iso, read_json


def sessions_db() -> Path:
    return Path(hermes_home()) / "state.db"


def registry_path() -> Path:
    return Path(hermes_home()) / "autoresearch" / "registry.json"


def readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def job_totals(job_id: str) -> dict[str, Any] | None:
    database = sessions_db()
    if not database.is_file():
        return None
    pattern = f"cron\\_{escape_like(job_id)}\\_%"
    with readonly_connection(database) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS sessions, "
            "COALESCE(SUM(input_tokens), 0) AS input_tokens, "
            "COALESCE(SUM(output_tokens), 0) AS output_tokens, "
            "COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd "
            "FROM sessions WHERE id LIKE ? ESCAPE '\\'",
            (pattern,),
        ).fetchone()
    if row is None or int(row["sessions"]) == 0:
        return None
    input_tokens = int(row["input_tokens"] or 0)
    output_tokens = int(row["output_tokens"] or 0)
    return {
        "sessions": int(row["sessions"]),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": round(float(row["estimated_cost_usd"] or 0), 4),
    }


def session_cost(session_id: str) -> dict[str, Any]:
    database = sessions_db()
    if not database.is_file():
        return {"error": "No SessionDB"}
    with readonly_connection(database) as connection:
        row = connection.execute(
            "SELECT id, model, input_tokens, output_tokens, estimated_cost_usd "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return {"error": f"Session {session_id} not found"}
    input_tokens = int(row["input_tokens"] or 0)
    output_tokens = int(row["output_tokens"] or 0)
    return {
        "session_id": row["id"],
        "model": row["model"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": row["estimated_cost_usd"],
    }


def job_cost(job_id: str) -> dict[str, Any]:
    totals = job_totals(job_id)
    if totals is None:
        return {"error": f"No sessions for job {job_id}"}
    return {"cron_job_id": job_id, **totals}


def track(
    run_dir: str,
    experiment_id: int,
    input_tokens: int,
    output_tokens: int,
    cost: float | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    if not (run_path / "config.json").is_file() or not (run_path / "status.json").is_file():
        raise ValueError(f"run is not initialized: {run_path}")
    usage_path = run_path / "usage.json"
    with exclusive_lock(run_path / ".autoresearch.lock"):
        usage = read_json(usage_path)
        if "experiments" not in usage:
            usage = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "experiments": {},
            }
        usage["experiments"][str(experiment_id)] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "timestamp": now_iso(),
        }
        entries = usage["experiments"].values()
        usage["total_input_tokens"] = sum(int(entry["input_tokens"]) for entry in entries)
        entries = usage["experiments"].values()
        usage["total_output_tokens"] = sum(int(entry["output_tokens"]) for entry in entries)
        usage["total_tokens"] = usage["total_input_tokens"] + usage["total_output_tokens"]
        costs = [
            float(entry["cost_usd"])
            for entry in usage["experiments"].values()
            if entry["cost_usd"] is not None
        ]
        usage["estimated_cost_usd"] = round(sum(costs), 4) if costs else 0.0
        atomic_write(usage_path, usage)
    return {
        "status": "tracked",
        "cumulative_tokens": usage["total_tokens"],
        "estimated_cost_usd": usage["estimated_cost_usd"],
    }


def registered_run(run_path: Path) -> dict[str, Any]:
    registry = read_json(registry_path())
    run = registry.get("runs", {}).get(run_path.name, {})
    if not isinstance(run, dict):
        return {}
    stored_path = Path(str(run.get("run_dir", ""))).expanduser().resolve()
    if stored_path != run_path:
        return {}
    return run


def usage_summary(run_dir: str) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    run = registered_run(run_path)
    result: dict[str, Any] = {"research_id": run_path.name, "source": "local"}
    usage = read_json(run_path / "usage.json")
    if usage:
        result["local_tracking"] = {
            "total_tokens": usage.get("total_tokens", 0),
            "estimated_cost_usd": usage.get("estimated_cost_usd"),
        }
    cron_job_id = run.get("cron_job_id")
    if cron_job_id:
        totals = job_totals(str(cron_job_id))
        if totals:
            result["source"] = "session_db"
            result["session_db_tracking"] = totals
    watchdog_job_id = run.get("watchdog_job_id")
    if watchdog_job_id:
        totals = job_totals(str(watchdog_job_id))
        if totals:
            result["watchdog_cost"] = totals
    return result


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    session_parser = commands.add_parser("session-cost")
    session_parser.add_argument("session_id")
    job_parser = commands.add_parser("job-cost")
    job_parser.add_argument("cron_job_id")
    for command in ("research-cost", "summary"):
        summary_parser = commands.add_parser(command)
        summary_parser.add_argument("run_dir")
    track_parser = commands.add_parser("track")
    track_parser.add_argument("run_dir")
    track_parser.add_argument("experiment_id", type=nonnegative_int)
    track_parser.add_argument("input_tokens", type=nonnegative_int)
    track_parser.add_argument("output_tokens", type=nonnegative_int)
    track_parser.add_argument("--cost", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "session-cost":
            result = session_cost(args.session_id)
        elif args.command == "job-cost":
            result = job_cost(args.cron_job_id)
        elif args.command in {"research-cost", "summary"}:
            result = usage_summary(args.run_dir)
        else:
            result = track(
                args.run_dir,
                args.experiment_id,
                args.input_tokens,
                args.output_tokens,
                args.cost,
            )
    except (ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
