from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from torben_job_contract import isoformat, run_job, torben_home
except ModuleNotFoundError:  # pytest imports this file as scripts.<module>.
    from scripts.torben_job_contract import isoformat, run_job, torben_home

DEFAULT_AGENT_ROOT = Path("/Users/ericfreeman/.hermes/hermes-agent")
PATCH_MARKERS = {
    "cron/jobs.py": [
        "Skipping cron job during due-check normalization",
        "Skipping cron job '%s' during due-check; malformed schedule or run state",
    ],
    "cron/scheduler.py": [
        "Cron tick could not load due jobs; skipping this tick",
    ],
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone(timezone.utc)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def inspect_patch(agent_root: Path = DEFAULT_AGENT_ROOT) -> dict[str, Any]:
    errors: list[str] = []
    checked: list[str] = []
    for rel_path, markers in PATCH_MARKERS.items():
        path = agent_root / rel_path
        checked.append(str(path))
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"{rel_path}: unreadable ({type(exc).__name__}: {exc})")
            continue
        for marker in markers:
            if marker not in text:
                errors.append(f"{rel_path}: missing patch marker {marker!r}")
    return {"status": "pass" if not errors else "failed", "checked": checked, "errors": errors}


def validate_registry(registry_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    payload = _read_json(registry_path)
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return {"status": "failed", "job_count": 0, "desk_v2_count": 0, "errors": ["jobs must be a list"]}

    desk_v2_count = 0
    shortest_interval_minutes: int | None = None
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            errors.append(f"jobs[{index}] is not an object")
            continue
        name = str(job.get("name") or job.get("id") or f"jobs[{index}]")
        if name.startswith("torben-desk-v2"):
            desk_v2_count += 1
        if not job.get("id"):
            errors.append(f"{name}: missing id")
        schedule = job.get("schedule")
        if not isinstance(schedule, dict):
            errors.append(f"{name}: schedule must be an object")
            continue
        kind = schedule.get("kind")
        if kind == "interval":
            try:
                minutes = int(schedule.get("minutes"))
            except (TypeError, ValueError):
                minutes = 0
            if minutes <= 0:
                errors.append(f"{name}: interval schedule requires positive minutes")
            elif job.get("enabled", True):
                shortest_interval_minutes = (
                    minutes if shortest_interval_minutes is None else min(shortest_interval_minutes, minutes)
                )
        elif kind == "cron":
            expr = str(schedule.get("expr") or schedule.get("cron") or "").strip()
            if not expr:
                errors.append(f"{name}: cron schedule requires expr or cron")
        elif kind == "once":
            if not str(schedule.get("run_at") or "").strip():
                errors.append(f"{name}: once schedule requires run_at")
        else:
            errors.append(f"{name}: unsupported schedule kind {kind!r}")
    if desk_v2_count != 10:
        errors.append(f"expected 10 torben-desk-v2 jobs, found {desk_v2_count}")
    return {
        "status": "pass" if not errors else "failed",
        "job_count": len(jobs),
        "desk_v2_count": desk_v2_count,
        "shortest_interval_minutes": shortest_interval_minutes,
        "errors": errors,
    }


def inspect_liveness(registry_path: Path, *, now: datetime | None = None, drift_factor: float = 1.8) -> dict[str, Any]:
    current = (now or _utc_now()).astimezone(timezone.utc)
    payload = _read_json(registry_path)
    jobs = payload.get("jobs") if isinstance(payload.get("jobs"), list) else []
    interval_drifts: list[dict[str, Any]] = []
    last_dispatches: list[datetime] = []
    shortest_interval_minutes: int | None = None
    for job in jobs:
        if not isinstance(job, dict) or not job.get("enabled", True):
            continue
        name = str(job.get("name") or job.get("id") or "")
        if name.startswith("torben-desk-v2"):
            continue
        schedule = job.get("schedule") if isinstance(job.get("schedule"), dict) else {}
        last_run = _parse_dt(job.get("last_run_at"))
        if last_run:
            last_dispatches.append(last_run)
        if schedule.get("kind") != "interval":
            continue
        try:
            minutes = int(schedule.get("minutes"))
        except (TypeError, ValueError):
            continue
        if minutes <= 0:
            continue
        shortest_interval_minutes = minutes if shortest_interval_minutes is None else min(shortest_interval_minutes, minutes)
        if not last_run:
            continue
        gap_seconds = (current - last_run).total_seconds()
        drift_limit_seconds = minutes * 60 * drift_factor
        if gap_seconds > drift_limit_seconds:
            interval_drifts.append(
                {
                    "name": name,
                    "minutes": minutes,
                    "gap_seconds": round(gap_seconds, 3),
                    "drift_limit_seconds": round(drift_limit_seconds, 3),
                    "last_run_at": isoformat(last_run),
                }
            )

    errors: list[str] = []
    if interval_drifts:
        errors.append("interval drift detected")
    if shortest_interval_minutes and last_dispatches:
        latest = max(last_dispatches)
        stall_limit_seconds = (shortest_interval_minutes + 2) * 60
        latest_gap = (current - latest).total_seconds()
        if latest_gap > stall_limit_seconds:
            errors.append(
                f"no non-desk job dispatched for {round(latest_gap, 3)}s "
                f"(limit {round(stall_limit_seconds, 3)}s)"
            )
    return {
        "status": "pass" if not errors else "failed",
        "checked_at": isoformat(current),
        "interval_drifts": interval_drifts,
        "errors": errors,
    }


def build_report(*, profile_home: Path, agent_root: Path, validate_jobs_only: bool = False) -> dict[str, Any]:
    registry_path = profile_home / "cron" / "jobs.json"
    registry = validate_registry(registry_path)
    patch = {"status": "skipped", "errors": []} if validate_jobs_only else inspect_patch(agent_root)
    liveness = {"status": "skipped", "errors": []} if validate_jobs_only else inspect_liveness(registry_path)
    errors = list(registry.get("errors") or []) + list(patch.get("errors") or []) + list(liveness.get("errors") or [])
    return {
        "task": "torben_scheduler_guard_check",
        "schema": "torben.scheduler-guard.v1",
        "generated_at": isoformat(),
        "wakeAgent": bool(errors),
        "status": "failed" if errors else "pass",
        "registry": registry,
        "patch": patch,
        "liveness": liveness,
        "errors": errors,
    }


def render_failure(report: dict[str, Any]) -> str:
    lines = ["Torben scheduler guard failed.", ""]
    for error in report.get("errors") or []:
        lines.append(f"- {error}")
    lines.append("")
    lines.append("Scheduler patch or registry health needs attention before registry writes continue.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-home", default=str(torben_home()))
    parser.add_argument("--agent-root", default=str(DEFAULT_AGENT_ROOT))
    parser.add_argument("--validate-jobs-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(
        profile_home=Path(args.profile_home),
        agent_root=Path(args.agent_root),
        validate_jobs_only=bool(args.validate_jobs_only),
    )
    if args.json or report["status"] == "pass":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_failure(report), end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    import sys

    if "--json" in sys.argv:
        raise SystemExit(main())
    raise SystemExit(run_job("torben-scheduler-guard-check", lambda: main()))
