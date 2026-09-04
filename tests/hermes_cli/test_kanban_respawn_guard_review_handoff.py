"""Regression tests for the respawn guard `active_pr` rule (issue #85663).

A review task that is unblocked back into the ``ready`` lane still carries the
reviewer's verdict comment, which legitimately quotes the PR URL under review.
The ``active_pr`` rule scans recent comments for a GitHub PR URL and, before
this fix, treated that quote as "a worker already opened a duplicate PR" —
stranding the card in ``ready`` with no worker for up to 24h.

The rule must be skipped for any task that has a review-handoff history (a
``review_requested`` event), while still firing for plain implement tasks that
actually have a worker-posted PR URL.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def _stamp_review_handoff(conn, task_id, now):
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, 'review_requested', ?, ?)",
        (task_id, '{"implementer":"worker","reviewer":"rev"}', now),
    )


def _add_comment(conn, task_id, author, body, now):
    conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) "
        "VALUES (?, ?, ?, ?)",
        (task_id, author, body, now),
    )
    conn.commit()


def test_review_handoff_with_pr_verdict_comment_not_guarded(conn):
    """A review-handoff task whose verdict comment quotes the PR URL must NOT
    trip the active_pr guard — that would strand it in ready after unblock."""
    now = int(time.time())
    task_id = kb.create_task(conn, title="review task", assignee="worker")
    _stamp_review_handoff(conn, task_id, now)
    _add_comment(
        conn, task_id, "rev",
        "changes requested: https://github.com/o/r/pull/123", now,
    )
    assert kb.check_respawn_guard(conn, task_id) is None


def test_plain_task_with_worker_pr_comment_is_guarded(conn):
    """A plain implement task with a worker-posted PR URL still trips active_pr."""
    now = int(time.time())
    task_id = kb.create_task(conn, title="impl task", assignee="worker2")
    _add_comment(
        conn, task_id, "worker2",
        "opened https://github.com/o/r/pull/456", now,
    )
    assert kb.check_respawn_guard(conn, task_id) == "active_pr"


def test_review_handoff_without_pr_comment_not_guarded(conn):
    """A review-handoff task with no PR URL in comments is never guarded."""
    now = int(time.time())
    task_id = kb.create_task(conn, title="review task 3", assignee="worker3")
    _stamp_review_handoff(conn, task_id, now)
    assert kb.check_respawn_guard(conn, task_id) is None


def test_plain_task_without_pr_comment_not_guarded(conn):
    """Sanity: a plain implement task with no PR comment is not guarded."""
    task_id = kb.create_task(conn, title="impl 4", assignee="w4")
    assert kb.check_respawn_guard(conn, task_id) is None
