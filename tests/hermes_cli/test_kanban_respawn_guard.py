"""Focused regression tests for Kanban respawn guard behavior."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _add_pr_comment(conn, task_id: str, created_at: int) -> None:
    conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) "
        "VALUES (?, 'worker', ?, ?)",
        (
            task_id,
            "Opened https://github.com/totemx-AI/subsidysmart/pull/42",
            created_at,
        ),
    )


def test_respawn_guard_recent_pr_without_requeue_is_active(kanban_home):
    """Recent PR evidence with no later requeue keeps the active_pr guard."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="has-pr", assignee="alice")
        _add_pr_comment(conn, task_id, int(time.time()) - 10)

        reason = kb.check_respawn_guard(conn, task_id)

    assert reason == "active_pr"


@pytest.mark.parametrize("event_kind", ["promoted", "unblocked", "status", "reclaimed"])
def test_respawn_guard_active_pr_bypassed_by_later_requeue(
    kanban_home, event_kind
):
    """Every explicit requeue event after PR evidence permits more work."""
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title=f"requeued-after-pr-{event_kind}",
            assignee="alice",
        )
        now = int(time.time())
        _add_pr_comment(conn, task_id, now - 20)
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) VALUES (?, ?, ?)",
            (task_id, event_kind, now - 10),
        )

        reason = kb.check_respawn_guard(conn, task_id)

    assert reason is None


def test_respawn_guard_active_pr_not_bypassed_by_earlier_requeue(kanban_home):
    """A requeue before the PR comment does not supersede newer PR evidence."""
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="requeued-before-pr",
            assignee="alice",
        )
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) "
            "VALUES (?, 'promoted', ?)",
            (task_id, now - 20),
        )
        _add_pr_comment(conn, task_id, now - 10)

        reason = kb.check_respawn_guard(conn, task_id)

    assert reason == "active_pr"


def test_respawn_guard_active_pr_not_bypassed_by_same_timestamp_requeue(
    kanban_home,
):
    """A requeue at the PR comment timestamp is not strictly later evidence."""
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="requeued-at-pr-timestamp",
            assignee="alice",
        )
        created_at = int(time.time()) - 10
        _add_pr_comment(conn, task_id, created_at)
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) "
            "VALUES (?, 'promoted', ?)",
            (task_id, created_at),
        )

        reason = kb.check_respawn_guard(conn, task_id)

    assert reason == "active_pr"
