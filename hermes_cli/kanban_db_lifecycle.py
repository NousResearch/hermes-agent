"""Parent-aware Kanban lifecycle transitions shared by generic and guarded callers."""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Optional


def _unblock_task_in_txn(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    attribution: Optional[dict[str, Any]] = None,
) -> Optional[tuple[int, str]]:
    """Apply unblock inside the caller's write transaction."""
    from hermes_cli import kanban_db as _kb

    now = int(time.time())
    current_status = _kb._task_status(conn, task_id)
    resume_status = (
        _kb._resume_status_from_events(conn, task_id)
        if current_status == "blocked"
        else "ready"
    )
    _kb._reclaim_dangling_run(
        conn,
        task_id,
        statuses=("blocked", "scheduled"),
        now=now,
        note="invariant recovery on unblock",
    )
    landing_status = _kb._landing_status_after_parents(conn, task_id)
    new_status = (
        "review"
        if landing_status == "ready" and resume_status == "review"
        else landing_status
    )
    cur = conn.execute(
        "UPDATE tasks SET status = ?, current_run_id = NULL, "
        "consecutive_failures = 0, last_failure_error = NULL "
        "WHERE id = ? AND status IN ('blocked', 'scheduled')",
        (new_status, task_id),
    )
    if cur.rowcount != 1:
        return None
    payload: dict[str, Any] = {}
    if new_status != "ready" or resume_status != "ready":
        payload.update({"status": new_status, "resume_status": resume_status})
    if attribution:
        payload.update(attribution)
    event_id = _kb._append_event(
        conn,
        task_id,
        "unblocked",
        payload or None,
    )
    return event_id, new_status


def unblock_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Move a blocked or scheduled task to its parent-aware resumable phase."""
    from hermes_cli import kanban_db as _kb

    with _kb.write_txn(conn):
        return _unblock_task_in_txn(conn, task_id) is not None
