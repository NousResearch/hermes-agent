"""Bounded model-facing Kanban edits and archival. No schema migration.

Related upstream proposals: #98836 (field edits), #65372 and #109027 (archive).
This surface deliberately retains workspaces and refuses non-leaf archival.
"""

from __future__ import annotations

import sqlite3
from typing import Any


def edit_fields(
    conn: sqlite3.Connection,
    task_id: str,
    changes: dict[str, Any],
    expected: dict[str, Any],
) -> list[str]:
    """Atomically edit metadata only when the caller's field values are current."""
    from hermes_cli import kanban_db as kb

    allowed = {"title", "body", "priority"}
    if not changes or not set(changes) <= allowed or set(expected) != set(changes):
        raise ValueError("provide changed fields and matching expected values")
    for field, value in changes.items():
        if field == "priority":
            if type(value) is not int:
                raise ValueError("priority must be an integer")
        elif not isinstance(value, str):
            raise ValueError(f"{field} must be a string")
    if "title" in changes and not changes["title"].strip():
        raise ValueError("title cannot be blank")
    changes = dict(changes)
    if "title" in changes:
        changes["title"] = changes["title"].strip()
    with kb.write_txn(conn):
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise ValueError("task not found")
        if row["status"] == "archived":
            raise ValueError("cannot edit an archived task")
        if any(row[k] != expected[k] for k in changes):
            raise ValueError("stale field values; read the task again before editing")
        changed = {k: v for k, v in changes.items() if row[k] != v}
        if changed:
            assignments = ", ".join(k + "=?" for k in changed)
            conn.execute(
                f"UPDATE tasks SET {assignments} WHERE id=?",
                (*changed.values(), task_id),
            )
            kb._append_event(
                conn,
                task_id,
                "edited",
                {"fields": list(changed), "source": "kanban_edit"},
            )
    if changed:
        kb.notify_task_updated(conn, task_id, list(changed))
    return list(changed)


def archive_leaf(
    conn: sqlite3.Connection,
    task_id: str,
    expected_status: str,
    reason: str,
) -> None:
    """Archive a stopped leaf without destroying its workspace or releasing children."""
    from hermes_cli import kanban_db as kb

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("archive reason is required")
    with kb.write_txn(conn):
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise ValueError("task not found")
        if row["status"] != expected_status:
            raise ValueError("stale status; read the task again")
        if (
            row["status"] in {"running", "archived"}
            or row["claim_lock"]
            or row["worker_pid"]
        ):
            raise ValueError(
                "cannot archive an active, claimed, or already archived task"
            )
        if conn.execute(
            "SELECT 1 FROM task_links WHERE parent_id=? LIMIT 1", (task_id,)
        ).fetchone():
            raise ValueError(
                "cannot archive a task with children; resolve the dependency graph separately"
            )
        if conn.execute(
            "SELECT 1 FROM task_runs WHERE task_id=? AND ended_at IS NULL LIMIT 1",
            (task_id,),
        ).fetchone():
            raise ValueError("cannot archive a task with an active run")
        conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (task_id,))
        kb._append_event(
            conn,
            task_id,
            "archived",
            {
                "source": "kanban_archive",
                "reason": reason.strip(),
                "workspace_retained": True,
            },
        )
    # No recompute: a leaf cannot release children. No cleanup: archive is not deletion.
    kb.notify_task_updated(conn, task_id, ["status"])
