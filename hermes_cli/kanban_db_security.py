"""Atomic worker authorization fences for Kanban graph mutations."""
from __future__ import annotations

import sqlite3
import time
from typing import Any, Optional


def _db():
    """Late-bind the facade so this sibling does not create an import cycle."""
    from hermes_cli import kanban_db
    return kanban_db


def unlink_tasks(
    conn: sqlite3.Connection, parent_id: str, child_id: str, *, actor: Optional[str] = None,
    expected_child_run_id: Optional[int] = None, worker_task_id: Optional[str] = None,
    deny_foreign_live_child: bool = False,
) -> bool:
    """Remove an edge, fencing worker writes to a verified live child run."""
    kb = _db()
    with kb.write_txn(conn):
        if worker_task_id is not None and child_id != worker_task_id:
            raise ValueError("worker may only unlink its own task")
        child = conn.execute(
            "SELECT status, current_run_id FROM tasks WHERE id = ?", (child_id,)
        ).fetchone()
        if child is None:
            return False
        if worker_task_id is not None and (
            child["status"] != "running" or child["current_run_id"] != expected_child_run_id
        ):
            raise ValueError("worker child run is absent, stale, mismatched, or ended")
        if (deny_foreign_live_child and worker_task_id is not None
                and child_id != worker_task_id and child["status"] == "running"
                and child["current_run_id"] is not None):
            raise ValueError("refusing to unlink a foreign child with a live run")
        cur = conn.execute(
            "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?", (parent_id, child_id),
        )
        removed = cur.rowcount > 0
        if removed:
            payload = {"parent": parent_id, "child": child_id}
            if actor is not None:
                payload["actor"] = actor
            kb._append_event(conn, child_id, "unlinked", payload, run_id=expected_child_run_id)
    if removed:
        # Re-gate the child now instead of leaving it in todo until the next tick.
        kb.recompute_ready(conn)
    return removed


def unblock_task(
    conn: sqlite3.Connection, task_id: str, *, actor: Optional[str] = None,
    expected_run_id: Optional[int] = None, worker_task_id: Optional[str] = None,
) -> bool:
    """``blocked``/``scheduled`` -> its resumable phase with worker fencing."""
    kb = _db()
    now = int(time.time())
    with kb.write_txn(conn):
        if worker_task_id is not None:
            if task_id != worker_task_id:
                raise ValueError("worker may only unblock its own task")
            live = conn.execute(
                """SELECT t.status, t.current_run_id, r.profile, r.status AS run_status, r.ended_at
                     FROM tasks t LEFT JOIN task_runs r ON r.id=t.current_run_id WHERE t.id=?""",
                (task_id,),
            ).fetchone()
            if (live is None or live["status"] not in {"blocked", "scheduled"} or
                    live["current_run_id"] != expected_run_id or
                    live["run_status"] != "running" or live["ended_at"] is not None or
                    not live["profile"]):
                raise ValueError("worker unblock requires the live current run")
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
        payload: Optional[dict[str, Any]] = (
            {"status": new_status, "resume_status": resume_status}
            if new_status != "ready" or resume_status != "ready" else None
        )
        if actor is not None:
            payload = {**(payload or {}), "actor": actor}
        kb._append_event(conn, task_id, "unblocked", payload, run_id=expected_run_id)
        return True
