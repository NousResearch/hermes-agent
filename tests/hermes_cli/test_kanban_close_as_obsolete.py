"""Regression tests for the ``close_as_obsolete`` admin verb.

Contract: kanban_db.close_as_obsolete docstring (blocked → done atomic
transition, refusal guards) — cherry-picked from origin/main 1f378065a,
Dre-approved 2026-07-19 ref t_21cc0e2a. These tests fail on pre-verb
code (AttributeError on import) and pin the three refusal paths plus the
happy path.
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


def _make_blocked_task(conn, title="obsolete-me"):
    task_id = kb.create_task(conn, title=title)
    assert kb.block_task(conn, task_id, reason="superseded")
    return task_id


def test_blocked_card_closes_as_done(kanban_home):
    conn = kb.connect()
    try:
        task_id = _make_blocked_task(conn)
        assert kb.close_as_obsolete(
            conn, task_id, reason="superseded by t_new", closed_by="test-admin"
        ) is True
        row = conn.execute(
            "SELECT status, result, claim_lock, block_kind FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "done"
        assert row["result"] == "obsolete: superseded by t_new"
        assert row["claim_lock"] is None
        assert row["block_kind"] is None
    finally:
        conn.close()


def test_non_blocked_card_returns_false(kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="still-ready")
        assert kb.close_as_obsolete(
            conn, task_id, reason="nope", closed_by="test-admin"
        ) is False
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        assert row["status"] != "done"
    finally:
        conn.close()


def test_missing_card_returns_false(kanban_home):
    conn = kb.connect()
    try:
        assert kb.close_as_obsolete(
            conn, "t_does_not_exist", reason="nope", closed_by="test-admin"
        ) is False
    finally:
        conn.close()


def test_stale_claim_lock_refused(kanban_home):
    conn = kb.connect()
    try:
        task_id = _make_blocked_task(conn)
        # Simulate the stale-lock scenario from the docstring: a blocked
        # card that still carries a claim_lock from a non-block_tool path.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_lock = ? WHERE id = ?",
                ("host:stale-worker", task_id),
            )
        with pytest.raises(kb.CardHasActiveClaimError) as exc:
            kb.close_as_obsolete(
                conn, task_id, reason="nope", closed_by="test-admin"
            )
        assert exc.value.claim_lock == "host:stale-worker"
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        assert row["status"] == "blocked"
    finally:
        conn.close()


def test_open_children_refused_until_terminal(kanban_home):
    conn = kb.connect()
    try:
        parent_id = _make_blocked_task(conn, title="parent")
        child_id = kb.create_task(conn, title="child")
        kb.link_tasks(conn, parent_id, child_id)

        with pytest.raises(kb.CardHasOpenChildrenError) as exc:
            kb.close_as_obsolete(
                conn, parent_id, reason="nope", closed_by="test-admin"
            )
        assert child_id in exc.value.open_children

        # Any terminal child status unblocks the close.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'cancelled' WHERE id = ?",
                (child_id,),
            )
        assert kb.close_as_obsolete(
            conn, parent_id, reason="children resolved", closed_by="test-admin"
        ) is True
    finally:
        conn.close()


def test_completed_event_carries_obsolete_marker(kanban_home):
    conn = kb.connect()
    try:
        task_id = _make_blocked_task(conn)
        assert kb.close_as_obsolete(
            conn, task_id, reason="moot", closed_by="test-admin"
        )
        run = conn.execute(
            "SELECT metadata, outcome FROM task_runs WHERE task_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert run is not None
        assert run["outcome"] == "completed"
        assert '"obsolete": true' in (run["metadata"] or "").lower()
    finally:
        conn.close()
