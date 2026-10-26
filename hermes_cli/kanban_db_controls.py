"""Native block/resume transitions with optional task-event concurrency guards."""
from __future__ import annotations

import sqlite3
import time
from typing import Optional


def _event_matches(conn: sqlite3.Connection, task_id: str, expected_event_id: Optional[int]) -> bool:
    if expected_event_id is None:
        return True
    if type(expected_event_id) is not int or expected_event_id < 1:
        raise ValueError("expected_event_id must be a positive integer")
    row = conn.execute("SELECT MAX(id) FROM task_events WHERE task_id = ?", (task_id,)).fetchone()
    return row[0] == expected_event_id


def _block_parked(conn, task_id, row, reason, kind, expected_run_id, expected_event_id):
    from hermes_cli import kanban_db as kb

    if expected_run_id is not None or row["current_run_id"] is not None:
        return False
    if kind is not None and row["block_kind"] is None:
        conn.execute(
            "UPDATE tasks SET block_kind = ?, block_recurrences = 1 WHERE id = ?",
            (kind, task_id),
        )
        kb._append_event(conn, task_id, "blocked", {
            "kind": kind, "reason": reason, "classified_in_place": True,
        })
        return True
    if expected_event_id is not None and kind == "needs_input":
        # A second current Stop must invalidate an already prepared resume,
        # without retyping the original block or increasing its loop counter.
        source_status = kb._resume_status_from_events(conn, task_id)
        kb._append_event(conn, task_id, "blocked", {
            "kind": row["block_kind"], "reason": reason, "reaffirmed": True,
            "source_status": source_status,
        })
        return True
    return False


def block_task(
    conn: sqlite3.Connection, task_id: str, *, reason: Optional[str] = None,
    kind: Optional[str] = None, expected_run_id: Optional[int] = None,
    expected_event_id: Optional[int] = None,
) -> bool:
    """Block active work; an explicit ``needs_input`` also stops queued work.

    ``expected_event_id`` optionally compares the task's latest event under
    the write transaction. A stale observation returns False without changing
    task/run state. It never substitutes for ``expected_run_id`` ownership.

    An untyped parked block can still be classified in place. A guarded
    ``needs_input`` on an already typed, parked block records a fresh Stop
    event without replacing its classification, counters, or run evidence.
    """
    from hermes_cli import kanban_db as kb

    if kind is not None and kind not in kb.VALID_BLOCK_KINDS:
        raise ValueError(f"block kind must be one of {sorted(kb.VALID_BLOCK_KINDS)} or None")
    with kb.write_txn(conn):
        if not _event_matches(conn, task_id, expected_event_id):
            return False
        cur_row = conn.execute(
            "SELECT status, block_kind, block_recurrences, current_run_id FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if cur_row is None:
            return False
        if cur_row["status"] == "blocked":
            return _block_parked(conn, task_id, cur_row, reason, kind, expected_run_id, expected_event_id)
        if cur_row["status"] in ("todo", "scheduled", "review"):
            # Only an operator Stop can override a dependency/time/review wait;
            # a worker cannot assert a run identity for an unclaimed card.
            if kind != "needs_input" or expected_run_id is not None:
                return False
        source_status = (
            kb._retry_status_for_run(conn, task_id)
            if cur_row["status"] == "running"
            else "review" if cur_row["status"] == "review" else "ready"
        )
        requested_kind = kind
        rekind_reason = None
        if kind == "dependency" and kb._parents_satisfied(conn, task_id):
            kind = "needs_input"
            rekind_reason = "no_open_parent"
        new_status, event_kind, set_sql, params, payload = kb._route_block(
            kind, reason, source_status, prev_kind=kb._row_get(cur_row, "block_kind"),
            prev_recurrences=int(kb._row_get(cur_row, "block_recurrences") or 0),
        )
        if rekind_reason:
            payload["requested_kind"] = requested_kind
            payload["rekind_reason"] = rekind_reason
        sql = f"""
                UPDATE tasks
                   SET status        = '{new_status}',
                       claim_lock    = NULL,
                       claim_expires = NULL,
                       worker_pid    = NULL,
                       {set_sql}
                 WHERE id = ?
                   AND status IN ('running', 'ready', 'todo', 'scheduled', 'review')
                """
        params = (*params, task_id)
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params = (*params, int(expected_run_id))
        if conn.execute(sql, params).rowcount != 1:
            return False
        run_id = kb._end_or_synthesize_run(
            conn, task_id, outcome="blocked", status="blocked", summary=reason, synthesize=bool(reason),
        )
        kb._append_event(conn, task_id, event_kind, payload, run_id=run_id)
        blocked_task = kb.get_task(conn, task_id)
        if kind == "dependency":
            # Preserve the existing dependency-lane hook ordering.
            kb._fire_task_hook("kanban_task_blocked", blocked_task, task_id, run_id, reason=reason)
            return True
    kb._fire_task_hook("kanban_task_blocked", blocked_task, task_id, run_id, reason=reason)
    return True


def _route_block(
    kind: Optional[str], reason: Optional[str], source_status: str, *,
    prev_kind: Optional[str], prev_recurrences: int,
) -> tuple[str, str, str, tuple, dict]:
    """Resolve the existing dependency lane and block-loop policy."""
    from hermes_cli import kanban_db as kb

    payload = {"reason": reason, "kind": kind, "source_status": source_status}
    if kind == "dependency":
        return "todo", "dependency_wait", "block_kind    = ?", (kind,), payload
    recurrences = prev_recurrences + 1 if prev_kind == kind else 1
    set_sql = "block_kind    = ?,\n                       block_recurrences = ?"
    payload = {"reason": reason, "kind": kind, "recurrences": recurrences, "source_status": source_status}
    if recurrences >= kb.BLOCK_RECURRENCE_LIMIT:
        payload["limit"] = kb.BLOCK_RECURRENCE_LIMIT
        return "triage", "block_loop_detected", set_sql, (kind, recurrences), payload
    return "blocked", "blocked", set_sql, (kind, recurrences), payload


def unblock_task(
    conn: sqlite3.Connection, task_id: str, *, expected_event_id: Optional[int] = None,
    request_id: Optional[str] = None,
) -> bool:
    """Resume blocked/scheduled work, optionally only from its observed event.

    Callers preparing an activation retain their original event token across
    retries. Reading a fresh token after a newer Stop is a new resume decision,
    not a safe retry. The guard runs before any dangling-run recovery.

    A caller may label a guarded transition with ``request_id``. The existing
    native event retains that label and its observed predecessor so a lost
    response can be reconciled without mistaking another operator's resume
    for this request. The label is provenance, not permission or a new grant.
    """
    from hermes_cli import kanban_db as kb

    if request_id is not None and (expected_event_id is None or not isinstance(request_id, str)
                                   or not request_id.strip() or len(request_id) > 200):
        raise ValueError("request_id requires an event guard and a nonempty label of at most 200 characters")
    now = int(time.time())
    with kb.write_txn(conn):
        if not _event_matches(conn, task_id, expected_event_id):
            return False
        resume_status = (
            kb._resume_status_from_events(conn, task_id)
            if kb._task_status(conn, task_id) == "blocked"
            else "ready"
        )
        kb._reclaim_dangling_run(
            conn, task_id, statuses=("blocked", "scheduled"), now=now,
            note="invariant recovery on unblock",
        )
        landing_status = kb._landing_status_after_parents(conn, task_id)
        new_status = (
            "review"
            if landing_status == "ready" and resume_status == "review"
            else landing_status
        )
        cur = conn.execute(
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "consecutive_failures = 0, last_failure_error = NULL "
            "WHERE id = ? AND status IN ('blocked', 'scheduled')", (new_status, task_id),
        )
        if cur.rowcount != 1:
            return False
        payload = ({"status": new_status, "resume_status": resume_status}
                   if new_status != "ready" or resume_status != "ready" else None)
        if request_id is not None:
            payload = {**(payload or {}), "request_id": request_id, "expected_event_id": expected_event_id}
        kb._append_event(conn, task_id, "unblocked", payload)
        return True
