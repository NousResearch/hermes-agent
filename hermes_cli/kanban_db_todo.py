"""Manual backlog-hold policy for Hermes Kanban todo tasks."""

from __future__ import annotations

import sqlite3
from typing import Any


def has_manual_todo_hold(conn: sqlite3.Connection, task_id: str) -> bool:
    """Return whether the latest lifecycle instruction explicitly requested todo."""
    row = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind IN ("
        "'status', 'promoted', 'dependency_wait', 'unblocked', 'changes_requested', "
        "'review_reopened', 'reclaimed', 'stale', 'timed_out', 'crashed', "
        "'spawn_failed', 'rate_limited', 'gave_up'"
        ") ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if not row or row["kind"] != "status":
        return False
    from hermes_cli.kanban_db import _json_dict

    payload: dict[str, Any] = _json_dict(row["payload"])
    return payload.get("requested_status") == "todo"
