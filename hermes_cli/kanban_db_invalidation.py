"""Atomic descendant retraction shared by ancestor reopen and gate verdict edits."""
from __future__ import annotations

import sqlite3
import time
from typing import Any, Optional


def invalidate_descendants(
    conn: sqlite3.Connection, task_id: str, *, author: str, gate_result: Optional[str] = None,
) -> dict[str, Any]:
    """Compose under the caller's transaction; return workers to terminate after commit.

    Done descendants reopen through the same lifecycle as ancestor reopen: retain
    result, completed runs and events, clear completed_at, and audit the retraction.
    Gate edits keep the ancestor done and preserve descendants' failure counters.
    """
    from hermes_cli import kanban_db as kb

    reason = "ancestor_gate_failed" if gate_result is not None else "ancestor_reopened"
    explanation = (
        f"ancestor {task_id} result changed to {gate_result}"
        if gate_result is not None else f"ancestor {task_id} was reopened"
    )
    now = int(time.time())
    invalidated: list[dict[str, Any]] = []
    terminations: list[tuple[Optional[int], Optional[str], Optional[int]]] = []
    with kb.write_txn(conn, allow_nested=True):
        rows = conn.execute(
            """
            WITH RECURSIVE descendants(id) AS (
                SELECT child_id FROM task_links WHERE parent_id = ?
                UNION
                SELECT l.child_id
                FROM task_links l
                JOIN descendants d ON d.id = l.parent_id
            )
            SELECT t.id, t.status, t.current_run_id, t.worker_pid, t.claim_lock, t.worker_started_at
            FROM descendants d
            JOIN tasks t ON t.id = d.id
            ORDER BY t.id
            """,
            (task_id,),
        ).fetchall()
        for row in rows:
            previous_status = row["status"]
            if previous_status not in {"ready", "review", "running", "done"}:
                continue
            resume_status = "ready"
            run_id = None
            if previous_status == "review":
                resume_status = "review"
            elif previous_status == "running":
                resume_status = kb._retry_status_for_run(conn, row["id"], row["current_run_id"])
                terminations.append((row["worker_pid"], row["claim_lock"], row["worker_started_at"]))
                run_id = kb._end_run(
                    conn, row["id"], outcome="reclaimed", status="todo",
                    summary=explanation,
                )
            conn.execute(
                "UPDATE tasks SET status = 'todo', completed_at = NULL, "
                "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
                "current_run_id = NULL, consecutive_failures = CASE WHEN ? THEN 0 "
                "ELSE consecutive_failures END WHERE id = ?", (gate_result is None, row["id"]),
            )
            entry = {
                "id": row["id"], "prior_status": previous_status,
                "new_status": "todo", "resume_status": resume_status,
            }
            kb._append_event(
                conn, row["id"], "descendant_invalidated",
                {"ancestor": task_id, "reason": reason, "result": gate_result,
                 **{k: v for k, v in entry.items() if k != "id"}},
                run_id=run_id,
            )
            # Legacy 'status' event so existing live-feed consumers still see
            # the move without learning the new event kind.
            kb._append_event(
                conn, row["id"], "status",
                {
                    "status": "todo", "reason": reason, "parent": task_id,
                    "previous_status": previous_status, "resume_status": resume_status,
                },
                run_id=run_id,
            )
            kb._insert_comment(
                conn, row["id"], author, f"Invalidated: {explanation}; "
                f"retracted from '{previous_status}' to 'todo' "
                f"(will resume via '{resume_status}').", now,
            )
            invalidated.append(entry)
    return {"invalidated": invalidated, "terminations": terminations}
