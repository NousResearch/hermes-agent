"""Independent acceptance of work parked for operator attention."""
from __future__ import annotations

import sqlite3


def require_independent_completion(conn: sqlite3.Connection, task_id: str, assignee: str | None) -> str:
    """Check under the board write lock; reassignment must not race acceptance."""
    # Keep these helpers lazy because kanban_db imports this module.
    from hermes_cli.kanban_db import _json_dict, _latest_event, _nonblank_str
    from hermes_cli.profiles import get_active_profile_name

    requested = _latest_event(conn, task_id, "review_requested")
    implementer = (
        _nonblank_str(_json_dict(requested["payload"]).get("implementer"))
        if requested is not None else None
    ) or assignee
    actor = get_active_profile_name()
    if implementer and actor == implementer:
        raise ValueError(
            f"cannot complete {task_id}: triage/blocked acceptance requires a profile "
            "other than the implementer"
        )
    return actor
