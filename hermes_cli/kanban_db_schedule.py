"""Durable timed parking, sharing the ordinary unblock and dispatch gates."""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime


def parse_schedule_at(at: str | None) -> float | None:
    """Require an explicit instant; prose and timezone-less dates are never timers."""
    if at is None:
        return None
    message = "at must be an ISO-8601 timestamp with a timezone (Z or +HH:MM)"
    if not isinstance(at, str):
        raise ValueError(message)
    try:
        instant = datetime.fromisoformat(at)
    except ValueError as exc:
        raise ValueError(message) from exc
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError(message)
    return instant.timestamp()


def schedule_task(
    conn: sqlite3.Connection, task_id: str, *, reason: str | None = None,
    expected_run_id: int | None = None, at: str | None = None,
) -> bool:
    from hermes_cli import kanban_db as kb

    scheduled_at = parse_schedule_at(at)
    with kb.write_txn(conn):
        params = [scheduled_at, task_id]
        sql = """
            UPDATE tasks
               SET status = 'scheduled', scheduled_at = ?,
                   claim_lock = NULL, claim_expires = NULL,
                   worker_pid = NULL, worker_started_at = NULL
             WHERE id = ?
               AND status IN ('todo', 'ready', 'running', 'blocked', 'scheduled')
        """
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params.append(int(expected_run_id))
        if conn.execute(sql, params).rowcount != 1:
            return False
        run_id = kb._end_or_synthesize_run(
            conn, task_id, outcome="scheduled", status="scheduled", summary=reason, synthesize=bool(reason),
        )
        payload = {"reason": reason}
        if scheduled_at is not None:
            payload["scheduled_at"] = scheduled_at
        kb._append_event(conn, task_id, "scheduled", payload, run_id=run_id)
    return True


def wake_due_scheduled(conn: sqlite3.Connection) -> None:
    from hermes_cli import kanban_db as kb

    now = time.time()
    # Read the current deadline under the same writer lock as the transition:
    # a re-schedule must not race a stale due-task snapshot.
    with kb.write_txn(conn):
        due = conn.execute(
            "SELECT id, scheduled_at FROM tasks WHERE status = 'scheduled' "
            "AND scheduled_at <= ? ORDER BY scheduled_at, id", (now,),
        ).fetchall()
        for row in due:
            unblock_task_in_txn(conn, row["id"])
            kb._append_event(conn, row["id"], "schedule_elapsed", {
                "scheduled_at": row["scheduled_at"], "woken_at": now,
            })


def unblock_task_in_txn(conn: sqlite3.Connection, task_id: str) -> bool:
    """Re-gate a parked task; the caller owns the write transaction."""
    from hermes_cli import kanban_db as kb

    now = int(time.time())
    resume_status = (
        kb._resume_status_from_events(conn, task_id)
        if kb._task_status(conn, task_id) == "blocked"
        else "ready"
    )
    kb._reclaim_dangling_run(
        conn, task_id, statuses=("blocked", "scheduled"), now=now,
        note="invariant recovery on unblock",
    )
    # Re-gate on parent completion before restoring the source phase.
    landing_status = kb._landing_status_after_parents(conn, task_id)
    new_status = (
        "review"
        if landing_status == "ready" and resume_status == "review"
        else landing_status
    )
    # ``block_kind``/``block_recurrences`` deliberately survive the unblock:
    # resetting them is the amnesia that let cron-unblock <-> re-block loop
    # unbounded; only complete_task clears them. ``consecutive_failures``
    # (the dispatcher's spawn/crash counter) IS reset — a deliberate unblock
    # is a fresh start for the retry budget.
    cur = conn.execute(
        "UPDATE tasks SET status = ?, current_run_id = NULL, "
        "consecutive_failures = 0, last_failure_error = NULL, scheduled_at = NULL "
        "WHERE id = ? AND status IN ('blocked', 'scheduled')", (new_status, task_id),
    )
    if cur.rowcount != 1:
        return False
    kb._append_event(
        conn, task_id, "unblocked",
        (
            {"status": new_status, "resume_status": resume_status}
            if new_status != "ready" or resume_status != "ready"
            else None
        ),
    )
    return True
