"""Opt-in non-preemptive queue aging and durable dispatch health reports.

Reports contain task identities and timing only, not task/patient copy. Both
standalone and embedded dispatch use this module; no outbound messages are sent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping

from hermes_cli.kanban_provider_lanes import aged_priority


@dataclass(frozen=True)
class SchedulingSettings:
    enabled: bool = False
    aging_seconds: int = 900
    maximum_bonus: int = 20
    stall_seconds: int = 3600
    oldest_limit: int = 10


def resolve_settings(raw: Mapping | None = None) -> SchedulingSettings:
    if raw is None:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("kanban", {}).get("dispatch_scheduling", {})
    if not isinstance(raw, Mapping):
        raise ValueError("kanban.dispatch_scheduling must be a mapping")
    enabled = raw.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("dispatch_scheduling.enabled must be a boolean")
    if not enabled:
        return SchedulingSettings()
    values = {}
    for name, default, minimum, maximum in (
        ("aging_seconds", 900, 1, 86400),
        ("maximum_bonus", 20, 0, 100),
        ("stall_seconds", 3600, 60, 604800),
        ("oldest_limit", 10, 1, 100),
    ):
        value = raw.get(name, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"dispatch_scheduling.{name} outside [{minimum}, {maximum}]")
        values[name] = value
    return SchedulingSettings(enabled=True, **values)


def queue_rows(conn, status: str, settings: SchedulingSettings, *, now: float | None = None):
    """Order waiting rows only; never mutate priority or any running worker.

    A priority difference above maximum_bonus remains an explicit priority
    decision. Within that bound, aging lets old work outrank fresh high-priority
    work. Original age and task id break ties deterministically.
    """
    rows = conn.execute(
        "SELECT id, assignee, priority, created_at FROM tasks "
        "WHERE status = ? AND claim_lock IS NULL ORDER BY priority DESC, created_at ASC, id ASC",
        (status,),
    ).fetchall()
    if not settings.enabled:
        return rows
    clock = time.time() if now is None else now
    return sorted(rows, key=lambda row: (
        -aged_priority(row["priority"], row["created_at"], clock,
                       aging_seconds=settings.aging_seconds, maximum_bonus=settings.maximum_bonus),
        row["created_at"], row["id"],
    ))


def _report_once(conn, kind: str, key: str, task_id: str, payload: dict, now: int,
                 *, run_id: int | None = None) -> bool:
    from hermes_cli import kanban_db as kb

    inserted = conn.execute(
        "INSERT OR IGNORE INTO dispatch_health_reports (kind, report_key, created_at) VALUES (?, ?, ?)",
        (kind, key, now),
    ).rowcount == 1
    if inserted:
        kb._append_event(conn, task_id, kind, payload, run_id=run_id)
    return inserted


def _stall_reports(conn, settings: SchedulingSettings, now: int) -> list[str]:
    rows = conn.execute(
        "SELECT t.id, t.current_run_id, t.last_heartbeat_at, r.started_at "
        "FROM tasks t JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running' AND r.ended_at IS NULL",
    ).fetchall()
    reported = []
    for row in rows:
        progress = max(row["started_at"], row["last_heartbeat_at"] or row["started_at"])
        if now - progress < settings.stall_seconds:
            continue
        key = f'{row["id"]}:{row["current_run_id"]}:{progress}'
        payload = {"run_id": row["current_run_id"], "last_progress_at": progress,
                   "idle_seconds": now - progress, "threshold_seconds": settings.stall_seconds,
                   "action": "report_only"}
        if _report_once(conn, "dispatch_stalled", key, row["id"], payload, now,
                        run_id=row["current_run_id"]):
            reported.append(row["id"])
    return reported


def _oldest_report(conn, settings: SchedulingSettings, now: int) -> list[dict]:
    rows = conn.execute(
        "SELECT id, status, assignee, created_at FROM tasks "
        "WHERE status NOT IN ('done', 'archived', 'cancelled') "
        "ORDER BY created_at ASC, id ASC LIMIT ?", (settings.oldest_limit,),
    ).fetchall()
    if not rows:
        return []
    oldest = [{"task_id": row["id"], "status": row["status"], "assignee": row["assignee"],
               "created_at": row["created_at"], "age_seconds": max(0, now - row["created_at"])}
              for row in rows]
    # UTC calendar-day key: at most one report per board/day across restarts
    # and competing processes, committed atomically with its task event.
    if _report_once(conn, "dispatch_oldest_open", str(now // 86400), rows[0]["id"],
                    {"observed_at": now, "oldest": oldest, "limit": settings.oldest_limit}, now):
        return oldest
    return []


def report_health(conn, settings: SchedulingSettings, *, dry_run: bool = False,
                  now: int | None = None) -> tuple[list[str], list[dict]]:
    """A stall is evidence, not permission to kill, requeue or preempt a job.

    Existing explicit TTL/runtime controls are unchanged. Deduplication and
    the corresponding task events share the board's write transaction.
    """
    if not settings.enabled or dry_run:
        return [], []
    from hermes_cli import kanban_db as kb

    clock = int(time.time()) if now is None else now
    with kb.write_txn(conn):
        conn.execute("CREATE TABLE IF NOT EXISTS dispatch_health_reports ("
                     "kind TEXT NOT NULL, report_key TEXT NOT NULL, created_at INTEGER NOT NULL, "
                     "PRIMARY KEY (kind, report_key))")
        stalled = _stall_reports(conn, settings, clock)
        oldest = _oldest_report(conn, settings, clock)
    return stalled, oldest
