"""Review-phase questions about a Kanban task, kept out of the kanban_db facade."""

from __future__ import annotations

import sqlite3
from typing import Optional

from hermes_cli.kanban_db import _retry_status_for_run, get_task


def is_review_completion(
    conn: sqlite3.Connection, task_id: str, *, expected_run_id: Optional[int] = None,
) -> bool:
    """Whether completion belongs to the review phase, including a claimed reviewer run.

    A reviewer claim changes the task row from ``review`` to ``running``. Its
    claimed-event provenance remains the durable distinction from an
    implementation run, so lifecycle gates must consult it rather than the
    current status alone.
    """
    task = get_task(conn, task_id)
    status = getattr(task, "status", None)
    if status == "review":
        return True
    if status is None:
        return False
    return _retry_status_for_run(conn, task_id, expected_run_id) == "review"
