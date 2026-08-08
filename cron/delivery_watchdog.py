"""Pure classification rules for the no-agent cron delivery watchdog."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - Windows fallback is exercised by platform integration.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from cron.executions import list_executions


_ALERT_BY_DELIVERY_STATE = {
    "failed": "delivery_failed",
    "uncertain_in_flight": "delivery_uncertain",
}


def _has_unresolved_origin(job: dict[str, Any]) -> bool:
    targets = {target.strip().lower() for target in str(job.get("deliver") or "").split(",")}
    if "origin" not in targets:
        return False
    origin = job.get("origin")
    return not (
        isinstance(origin, dict)
        and str(origin.get("platform") or "").strip()
        and str(origin.get("chat_id") or "").strip()
    )


def _expects_external_delivery(job: dict[str, Any]) -> bool:
    """Return whether a job has a resolvable non-local delivery contract."""
    raw = job.get("deliver", "local")
    if raw is None:
        return False
    for target in str(raw).split(","):
        target = target.strip().lower()
        if not target or target == "local":
            continue
        if target == "origin":
            if job.get("origin"):
                return True
            continue
        return True
    return False


def _latest_by_job(executions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for execution in executions:
        job_id = str(execution.get("job_id") or "")
        if not job_id:
            continue
        current = latest.get(job_id)
        execution_key = (str(execution.get("claimed_at") or ""), str(execution.get("id") or ""))
        current_key = (
            str(current.get("claimed_at") or ""),
            str(current.get("id") or ""),
        ) if current else None
        if current_key is None or execution_key > current_key:
            latest[job_id] = execution
    return latest


def classify_delivery_events(
    jobs: list[dict[str, Any]],
    executions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return actionable delivery anomalies for each job's latest execution.

    This function is deliberately read-only. It does not schedule retries or
    infer a platform-side success from a missing receipt.
    """
    latest = _latest_by_job(executions)
    events: list[dict[str, Any]] = []
    for job in jobs:
        if not job.get("enabled", True):
            continue
        job_id = str(job.get("id") or "")
        execution = latest.get(job_id)
        if job.get("last_status") == "error":
            failure_identity = str(execution.get("id") or "") if execution else ""
            if not failure_identity:
                failure_identity = f"runtime:{job_id}:{job.get('last_run_at') or job.get('last_error') or 'unknown'}"
            events.append(
                {
                    "event": "job_runtime_error",
                    "execution_id": failure_identity,
                    "job_id": job_id,
                    "delivery_state": None,
                    "detail": str(job.get("last_error") or "unknown job error"),
                }
            )
            continue
        if _has_unresolved_origin(job):
            events.append(
                {
                    "event": "unresolved_origin",
                    "execution_id": f"config:{job_id}",
                    "job_id": job_id,
                    "delivery_state": None,
                }
            )
            continue
        execution = latest.get(job_id)
        if not execution:
            continue
        if execution.get("status") == "unknown":
            events.append(
                {
                    "event": "execution_unknown",
                    "execution_id": str(execution.get("id") or ""),
                    "job_id": job_id,
                    "delivery_state": execution.get("delivery_state"),
                }
            )
            continue
        if not _expects_external_delivery(job):
            continue
        if execution.get("status") != "completed":
            continue
        delivery_state = execution.get("delivery_state")
        event = _ALERT_BY_DELIVERY_STATE.get(str(delivery_state))
        if event is None and delivery_state is None:
            event = "missing_delivery_receipt"
        if event is None:
            continue
        events.append(
            {
                "event": event,
                "execution_id": str(execution.get("id") or ""),
                "job_id": job_id,
                "delivery_state": delivery_state,
            }
        )
    return events


def _watchdog_event_key(event: dict[str, Any]) -> tuple[str, str]:
    kind = str(event.get("event") or "").strip()
    execution_id = str(event.get("execution_id") or "").strip()
    if not kind or not execution_id:
        raise ValueError("watchdog event requires event and execution_id")
    return kind, execution_id


def append_new_watchdog_events(
    event_log: Path,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Durably append events not previously reported for the same execution."""
    event_log.parent.mkdir(parents=True, exist_ok=True)
    created = not event_log.exists()
    new_events: list[dict[str, Any]] = []
    with event_log.open("a+", encoding="utf-8") as handle:
        if created:
            os.chmod(event_log, 0o600)
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            existing_keys: set[tuple[str, str]] = set()
            handle.seek(0)
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed watchdog event log at line {line_number}") from exc
                if not isinstance(existing, dict):
                    raise ValueError(f"malformed watchdog event log at line {line_number}")
                existing_keys.add(_watchdog_event_key(existing))

            handle.seek(0, os.SEEK_END)
            for event in events:
                key = _watchdog_event_key(event)
                if key in existing_keys:
                    continue
                record = dict(event)
                record["reported_at"] = datetime.now(timezone.utc).isoformat()
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                existing_keys.add(key)
                new_events.append(event)
            if new_events:
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return new_events


def run_delivery_watchdog(
    jobs_path: Path,
    event_log: Path,
    executions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Classify and durably de-duplicate delivery anomalies from persisted state."""
    raw_jobs = json.loads(jobs_path.read_text(encoding="utf-8-sig"))
    if isinstance(raw_jobs, dict):
        jobs = raw_jobs.get("jobs", [])
    else:
        jobs = raw_jobs
    if not isinstance(jobs, list) or not all(isinstance(job, dict) for job in jobs):
        raise ValueError("jobs registry must be a list or a mapping with a jobs list")
    events = classify_delivery_events(jobs, executions)
    return append_new_watchdog_events(event_log, events)


def _default_paths() -> tuple[Path, Path, Path]:
    from cron.jobs import JOBS_FILE

    cron_dir = Path(JOBS_FILE).parent
    return (
        Path(JOBS_FILE),
        cron_dir / "delivery-watchdog-events.jsonl",
        cron_dir / "delivery-watchdog-baseline.json",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the no-agent watchdog; empty stdout means no new alert."""
    default_jobs, default_event_log, default_baseline = _default_paths()
    parser = argparse.ArgumentParser(description="Audit cron delivery receipts without retries.")
    parser.add_argument("--jobs", type=Path, default=default_jobs)
    parser.add_argument("--event-log", type=Path, default=default_event_log)
    parser.add_argument("--baseline", type=Path, default=default_baseline)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args(argv)

    events = run_delivery_watchdog(
        args.jobs,
        args.event_log,
        list_executions(limit=max(1, min(args.limit, 1000))),
    )
    if not args.baseline.exists():
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            json.dumps({"bootstrapped_at": datetime.now(timezone.utc).isoformat()}) + "\n",
            encoding="utf-8",
        )
        return 0
    if events:
        print(f"⚠️ Cron delivery watchdog: {len(events)} new anomaly/anomalies")
        for event in events:
            print(f"- {event['event']}: job={event['job_id']} execution={event['execution_id']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the script entrypoint.
    raise SystemExit(main(sys.argv[1:]))
