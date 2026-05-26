"""Tests for the human_review status and review/approve/reject transitions.

Cover the four new ``kanban_db`` helpers (``move_to_review``,
``move_to_human_review``, ``approve_task``, ``reject_task``) and the
dispatcher's hands-off behavior for ``human_review`` tasks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claim(conn, tid):
    task = kb.claim_task(conn, tid)
    assert task is not None, "claim_task must succeed in fixture path"
    return task


def test_human_review_is_a_valid_status(kanban_home):
    assert "human_review" in kb.VALID_STATUSES


def test_move_to_review_running_to_review(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        _claim(conn, tid)
        ok = kb.move_to_review(conn, tid, reason="PR opened at https://example/pr/1")
        assert ok
        t = kb.get_task(conn, tid)
        assert t.status == "review"
        events = [e.kind for e in kb.list_events(conn, tid)]
        assert "review_requested" in events


def test_move_to_review_rejects_blocked_task(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        _claim(conn, tid)
        kb.block_task(conn, tid, reason="stuck")
        assert kb.move_to_review(conn, tid, reason="too late") is False


def test_move_to_human_review_review_to_human_review(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        _claim(conn, tid)
        kb.move_to_review(conn, tid, reason="PR opened")
        # Claim as review agent, then escalate to human_review.
        _ = kb.claim_review_task(conn, tid)
        ok = kb.move_to_human_review(
            conn, tid, reason="PR merged; awaiting Sahil",
        )
        assert ok
        t = kb.get_task(conn, tid)
        assert t.status == "human_review"
        events = [e.kind for e in kb.list_events(conn, tid)]
        assert "human_review_requested" in events


def test_move_to_human_review_rejects_blocked(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        _claim(conn, tid)
        kb.block_task(conn, tid, reason="stuck")
        assert kb.move_to_human_review(conn, tid, reason="x") is False


def test_approve_task_human_review_to_done(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        _claim(conn, tid)
        kb.move_to_human_review(conn, tid, reason="ready")
        ok, outcome, pr_url, task = kb.approve_task(conn, tid, reason="LGTM")
        assert ok
        assert outcome == "done"
        assert pr_url is None
        assert task is None
        t = kb.get_task(conn, tid)
        assert t.status == "done"
        assert t.completed_at is not None
        events = [e.kind for e in kb.list_events(conn, tid)]
        assert "approved" in events


def test_approve_task_rejects_non_human_review(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        _claim(conn, tid)
        # Still running — approve_task must refuse.
        ok, outcome, _, __ = kb.approve_task(conn, tid, reason="nope")
        assert not ok
        # And on review — also refuse.
        kb.move_to_review(conn, tid, reason="pr")
        ok2, _, __, ___ = kb.approve_task(conn, tid, reason="nope")
        assert not ok2


def test_reject_task_human_review_to_ready(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        _claim(conn, tid)
        kb.move_to_human_review(conn, tid, reason="ready")
        ok = kb.reject_task(conn, tid, reason="needs tests")
        assert ok
        t = kb.get_task(conn, tid)
        assert t.status == "ready"
        assert t.completed_at is None
        events = [e.kind for e in kb.list_events(conn, tid)]
        assert "rejected" in events


def test_reject_task_review_to_ready(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        _claim(conn, tid)
        kb.move_to_review(conn, tid, reason="pr")
        ok = kb.reject_task(conn, tid, reason="AC not met")
        assert ok
        t = kb.get_task(conn, tid)
        assert t.status == "ready"


def test_reject_task_rejects_running(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        _claim(conn, tid)
        assert kb.reject_task(conn, tid, reason="bad") is False


def test_dispatch_skips_human_review_tasks(kanban_home):
    """The dispatcher must NOT spawn anything for human_review."""
    spawned: list = []

    def fake_spawn(claimed, workspace, board=None):
        spawned.append((claimed.id, claimed.status))
        return 12345

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        _claim(conn, tid)
        kb.move_to_human_review(conn, tid, reason="awaiting human")
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn)
    assert spawned == [], "dispatcher must skip human_review tasks"
    # Nothing should have been spawned for our task in any bucket.
    spawned_ids = [row[0] for row in result.spawned]
    assert tid not in spawned_ids


def test_approve_cas_atomic(kanban_home):
    """Approve must be idempotent: a second call after success is a no-op."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        _claim(conn, tid)
        kb.move_to_human_review(conn, tid, reason="ready")
        ok, outcome, _, __ = kb.approve_task(conn, tid, reason="ok")
        assert ok
        assert outcome == "done"
        # Second call — task is already done; CAS row count == 0, returns not-ok.
        ok2, _, __, ___ = kb.approve_task(conn, tid, reason="ok")
        assert not ok2
