"""Task field edits: the assignee write, the title/body/priority/result edit, and the combined
patch that applies both in one transaction (``update_task_fields``).

``assign_locked`` / ``edit_locked`` run inside the CALLER's transaction, so the public entry
points in ``hermes_cli.kanban_db`` (``assign_task``, ``edit_task``) and the combined patch share
one write path. Origin-resident helpers are reached late-bound via ``_kb`` so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Optional

from hermes_cli import kanban_db as _kb


def assign_locked(conn: sqlite3.Connection, task_id: str, profile: Optional[str]) -> bool:
    """Assignee write + ``assigned`` event inside the CALLER's txn; the caller owns
    the commit and the post-commit observer."""
    row = conn.execute(
        "SELECT status, claim_lock, assignee FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if not row:
        return False
    if row["claim_lock"] is not None and row["status"] == "running":
        raise RuntimeError(
            f"cannot reassign {task_id}: currently running (claimed). "
            "Wait for completion or reclaim the stale lock first."
        )
    if row["assignee"] != profile:
        # The failure streak is per task/profile; a new profile starts fresh.
        conn.execute(
            "UPDATE tasks SET assignee = ?, consecutive_failures = 0, "
            "last_failure_error = NULL WHERE id = ?", (profile, task_id),
        )
    else:
        conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", (profile, task_id))
    # ``from`` lets the respawn guard tell a real handoff (dev→closer) from
    # a no-op re-assign or an unassign, which must not lift ``active_pr``.
    _kb._append_event(
        conn, task_id, "assigned", {"assignee": profile, "from": row["assignee"]},
    )
    return True


# Content edits the external API refuses: a finished card's text is a historical record.
_TERMINAL_EDIT_STATES = frozenset({"done", "archived"})


def update_task_fields(
    conn: sqlite3.Connection, task_id: str, *, assign: bool = False, assignee: Optional[str] = None,
    title: Optional[str] = None, body: Optional[str] = None, priority: Optional[int] = None,
    board: Optional[str] = None,
) -> bool:
    """Reassign (when ``assign``) and/or edit ``title``/``body``/``priority`` (``None`` =
    unchanged) in ONE transaction; ``False`` when the task does not exist.

    All-or-nothing: a transition landing between the phases makes a later guard raise
    ``RuntimeError`` and rolls the reassignment back too, so a caller never sees a refusal
    after the assignee was already persisted (and announced). Title/body edits on a
    ``done``/``archived`` task raise. Observers fire once, after the combined commit.
    """
    changed: list[str] = []
    with _kb.write_txn(conn):
        if assign:
            if not assign_locked(conn, task_id, _kb._canonical_assignee(assignee)):
                return False
            changed.append("assignee")
        if (title is not None or body is not None) and _kb._task_status(conn, task_id) in _TERMINAL_EDIT_STATES:
            raise RuntimeError("cannot edit the title/body of a finished task")
        if title is not None or body is not None or priority is not None:
            edited = edit_locked(conn, task_id, title=title, body=body, priority=priority)
            if edited is None:
                return False
            changed += edited
        elif not assign:
            return _kb._task_status(conn, task_id) is not None
    # Observer fires AFTER commit so subscribers see durable state.
    _kb.notify_task_updated(conn, task_id, changed, board=board)
    return True


def edit_locked(
    conn: sqlite3.Connection, task_id: str, *, title: Optional[str] = None,
    body: Optional[str] = None, priority: Optional[int] = None,
    result: Optional[str] = None, summary: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[list[str]]:
    """``edit_task``'s write + events inside the CALLER's txn; the changed field names, or
    ``None`` when nothing was applied. The caller owns the commit and the observer."""
    changed_fields = [
        field for field, value in (("title", title), ("body", body), ("priority", priority))
        if value is not None
    ]
    status = _kb._task_status(conn, task_id)
    if status is None or (result is not None and status != "done"):
        return None
    assignments = []
    params = []
    for field, value in (("title", title), ("body", body), ("priority", priority)):
        if value is not None:
            assignments.append(f"{field} = ?")
            params.append(value)
    if result is not None:
        assignments.append("result = ?")
        params.append(result)
        changed_fields.append("result")
    if not assignments:
        return None
    conn.execute(
        f"UPDATE tasks SET {', '.join(assignments)} WHERE id = ?",
        (*params, task_id),
    )
    if priority is not None:
        _kb._append_event(conn, task_id, "reprioritized", {"priority": priority})
    if result is None:
        non_priority_fields = [field for field in changed_fields if field != "priority"]
        if non_priority_fields:
            _kb._append_event(conn, task_id, "edited", {"fields": non_priority_fields})
    else:
        handoff_summary = summary if summary is not None else result
        changed_fields.append("summary")
        if metadata is not None:
            changed_fields.append("metadata")
        run = conn.execute(
            """
            SELECT id FROM task_runs
             WHERE task_id = ?
               AND outcome = 'completed'
             ORDER BY COALESCE(ended_at, started_at, 0) DESC, id DESC
             LIMIT 1
            """,
            (task_id,),
        ).fetchone()
        if run is None:
            run_id = _kb._synthesize_ended_run(
                conn, task_id, outcome="completed", summary=handoff_summary, metadata=metadata,
            )
        else:
            run_id = int(run["id"])
            conn.execute("UPDATE task_runs SET summary = ? WHERE id = ?", (handoff_summary, run_id))
            if metadata is not None:
                conn.execute(
                    "UPDATE task_runs SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata, ensure_ascii=False), run_id),
                )
        _kb._append_event(
            conn, task_id, "edited",
            {
                "fields": ["result", "summary"] + (["metadata"] if metadata is not None else []),
                "result_len": len(result) if result else 0,
                "summary": _kb._first_line(handoff_summary, 400) or None,
            },
            run_id=run_id,
        )
    return changed_fields
