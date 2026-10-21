"""The ``scheduled`` park and its deadline timer: ``schedule_task`` resolves the
wake deadline at park time; the dispatcher tick (``recompute_ready``) wakes due
parks.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking).
"""

from __future__ import annotations

import datetime
import re
import sqlite3
from typing import Any, Optional

# Mirrors the keeper watchdog's UNTIL_RE so legacy reason markers keep parsing.
_SCHEDULED_UNTIL_RE = re.compile(r"SCHEDULED_UNTIL=([0-9T:+.\-Z]+)")


def parse_until_iso(text: str) -> Optional[int]:
    """ISO8601 instant -> epoch seconds; a naive stamp is machine-local.
    ``None`` on any parse failure (callers decide whether that is an error)."""
    try:
        return int(datetime.datetime.fromisoformat(text).timestamp())
    except ValueError:
        return None


def resolve_deadline(reason: Optional[str], until: Optional[str]) -> Optional[int]:
    """Park deadline: explicit ``until`` beats a ``SCHEDULED_UNTIL=<ISO8601>``
    marker in ``reason``. A non-blank ``until`` that does not parse raises
    ``ValueError`` (caller error); a garbage marker degrades to no timer."""
    until = (until or "").strip()
    if until:
        deadline = parse_until_iso(until)
        if deadline is None:
            raise ValueError(
                f"unparseable until {until!r}: expected ISO8601, e.g. 2026-10-14T09:30:00+03:00")
        return deadline
    match = _SCHEDULED_UNTIL_RE.search(reason or "")
    return parse_until_iso(match.group(1)) if match else None


def wake_due_scheduled(conn: sqlite3.Connection, now: int) -> int:
    """Re-gate every ``scheduled`` park whose deadline has passed; returns the
    count. Runs inside the caller's write txn (``recompute_ready``)."""
    from hermes_cli import kanban_db as _kb

    rows = conn.execute(
        "SELECT id, scheduled_until FROM tasks WHERE status = 'scheduled' "
        "AND scheduled_until <= ?", (now,),
    ).fetchall()
    for row in rows:
        task_id = row["id"]
        _kb._reclaim_dangling_run(
            conn, task_id, statuses=("scheduled",), now=now,
            note="invariant recovery on scheduled wake",
        )
        landing = _kb._landing_status_after_parents(conn, task_id)
        conn.execute(
            "UPDATE tasks SET status = ?, scheduled_until = NULL, current_run_id = NULL, "
            "consecutive_failures = 0, last_failure_error = NULL WHERE id = ?", (landing, task_id),
        )
        _kb._append_event(conn, task_id, "unblocked", {
            "reason": "scheduled deadline elapsed", "status": landing,
            "deadline": row["scheduled_until"],
        })
    return len(rows)


def schedule_task(
    conn: sqlite3.Connection, task_id: str, *, reason: Optional[str] = None,
    expected_run_id: Optional[int] = None, until: Optional[str] = None,
) -> bool:
    """Park in ``scheduled`` (waiting on time, not a human; not dispatchable).

    The deadline comes from ``until`` (ISO8601; beats the reason marker) or a
    ``SCHEDULED_UNTIL=<ISO8601>`` marker in ``reason``; with one, the dispatcher
    tick (``recompute_ready``) wakes the park once it passes. Without one,
    ``unblock_task`` is the only exit. Raises ``ValueError`` for a non-blank
    ``until`` that does not parse; a garbage marker just parks without a timer."""
    from hermes_cli import kanban_db as _kb

    deadline = resolve_deadline(reason, until)
    with _kb.write_txn(conn):
        params: list[Any] = [deadline, task_id]
        sql = """
            UPDATE tasks
               SET status       = 'scheduled',
                   claim_lock   = NULL,
                   claim_expires= NULL,
                   worker_pid   = NULL,
                   scheduled_until = ?
             WHERE id = ?
               AND status IN ('todo', 'ready', 'running', 'blocked')
        """
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params.append(int(expected_run_id))
        if conn.execute(sql, params).rowcount != 1:
            return False
        run_id = _kb._end_or_synthesize_run(
            conn, task_id, outcome="scheduled", status="scheduled", summary=reason, synthesize=bool(reason),
        )
        extra = {"scheduled_until": deadline} if deadline is not None else {}
        _kb._append_event(conn, task_id, "scheduled", {"reason": reason, **extra}, run_id=run_id)
        return True
