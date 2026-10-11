"""Retry policy helpers for durable Kanban runs."""

from __future__ import annotations

import sqlite3
import time
from typing import Optional


def retry_status_for_run(conn: sqlite3.Connection, task_id: str, run_id: Optional[int] = None) -> str:
    """Choose the retry phase, or block automatic replay after a one-shot attempt."""
    from hermes_cli import kanban_db as kb

    task = conn.execute("SELECT max_retries FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if task is not None and task["max_retries"] is not None and int(task["max_retries"]) == 0:
        prior = conn.execute(
            "SELECT 1 FROM task_runs WHERE task_id = ? AND started_at IS NOT NULL LIMIT 1", (task_id,),
        ).fetchone()
        if prior is not None:
            reason = (
                "Automatic replay is disabled (max_retries=0) after a recorded execution attempt. "
                "Operator reconciliation must verify the retained run, worker identity and external effects "
                "before authorizing any new execution; this block does not prove the worker stopped."
            )
            previous = kb._json_dict(kb._row_get(kb._latest_event(conn, task_id, "blocked"), "payload"))
            if previous.get("reason_code") != "no_auto_replay":
                conn.execute(
                    "UPDATE tasks SET block_kind='capability', last_failure_error=? WHERE id=?", (reason, task_id),
                )
                kb._insert_comment(conn, task_id, "dispatcher", reason, int(time.time()))
                kb._append_event(conn, task_id, "blocked", {
                    "kind": "capability", "reason": reason, "reason_code": "no_auto_replay", "sticky": True,
                }, run_id=run_id or kb._current_run_id(conn, task_id))
            return "blocked"
    if run_id is None:
        run_id = kb._current_run_id(conn, task_id)
    if run_id is None:
        return "ready"
    event = kb._latest_event(conn, task_id, "claimed", run_id)
    payload = kb._json_dict(kb._row_get(event, "payload"))
    return "review" if payload.get("source_status") == "review" else "ready"
