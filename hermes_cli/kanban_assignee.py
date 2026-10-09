"""Creation routing for assignees that have no runnable profile on disk."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Optional

from hermes_cli import kanban_db as kb


def create_assignee_warning(assignee: Optional[str]) -> tuple[Optional[str], list[str]]:
    """Return a triage reason and available profiles when an assignee is unknown.

    An empty inventory cannot establish that a profile is missing.
    """
    if not assignee:
        return None, []
    try:
        profiles = kb.list_profiles_on_disk(require_complete=True)
    except OSError:
        return None, []
    if not profiles or kb._canonical_assignee(assignee) in profiles:
        return None, profiles
    reason = (f"Unknown assignee profile '{assignee}'; task moved to triage. "
              f"Valid profiles: {', '.join(profiles)}")
    return reason, profiles


def record_create_assignee_warning(conn: sqlite3.Connection, task_id: str, reason: Optional[str]) -> None:
    if reason:
        kb.add_comment(conn, task_id, author="kanban", body=reason)


@contextmanager
def create_replay_guard(conn: sqlite3.Connection, key: Optional[str]):
    """Serialize keyed creates and report whether create_task will reuse a task."""
    if not key:
        yield False
        return
    with kb.write_txn(conn, allow_nested=True):
        replay = conn.execute(
            "SELECT 1 FROM tasks WHERE idempotency_key = ? AND status != 'archived' LIMIT 1",
            (key,),
        ).fetchone() is not None
        yield replay
