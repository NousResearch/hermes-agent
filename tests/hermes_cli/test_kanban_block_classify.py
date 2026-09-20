"""Tests for block_task() classification of already-blocked cards.

Reproduces issue #117363: the circuit breaker parks cards untyped, and
block_task() refuses to classify them because the WHERE clause only
matches running/ready.
"""
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_classify_already_blocked_untyped_card(kanban_home):
    """A card parked blocked with block_kind=NULL should accept a kind."""
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="breaker-parked card")
        # Simulate the circuit breaker: set status=blocked without block_kind.
        conn.execute(
            "UPDATE tasks SET status = 'blocked', block_kind = NULL WHERE id = ?",
            (task_id,),
        )
        assert kb.get_task(conn, task_id).block_kind is None

        result = kb.block_task(conn, task_id, kind="needs_input", reason="supervisor classifying breaker block")
        assert result is True

        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"

        events = [e for e in kb.list_events(conn, task_id) if e.kind == "block_classified"]
        assert len(events) == 1
        assert events[-1].payload["kind"] == "needs_input"
        assert events[-1].payload["previous_kind"] is None


def test_classify_already_blocked_typed_card(kanban_home):
    """Re-classifying a typed blocked card should update block_kind."""
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="typed block")
        kb.block_task(conn, task_id, kind="transient", reason="flaky")

        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.block_kind == "transient"

        result = kb.block_task(conn, task_id, kind="needs_input", reason="escalated")
        assert result is True

        task = kb.get_task(conn, task_id)
        assert task.block_kind == "needs_input"

        events = [e for e in kb.list_events(conn, task_id) if e.kind == "block_classified"]
        assert len(events) == 1
        assert events[-1].payload["previous_kind"] == "transient"


def test_classify_already_blocked_same_kind_is_idempotent(kanban_home):
    """Classifying with the same kind should be a no-op (idempotent)."""
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="already classified")
        kb.block_task(conn, task_id, kind="needs_input", reason="initial")

        result = kb.block_task(conn, task_id, kind="needs_input", reason="duplicate")
        assert result is True

        events = [e for e in kb.list_events(conn, task_id) if e.kind == "block_classified"]
        assert len(events) == 0, "idempotent classify must not emit an event"


def test_block_task_still_blocks_running_card(kanban_home):
    """Normal blocking of a running card should still work."""
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="normal card")
        kb.claim_task(conn, task_id, worker_pid=12345)

        result = kb.block_task(conn, task_id, kind="needs_input", reason="waiting for input")
        assert result is True

        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"


def test_classify_blocked_without_kind_returns_false(kanban_home):
    """block_task() on an already-blocked card without kind should return False."""
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="blocked card")
        kb.block_task(conn, task_id, kind="transient", reason="flaky")

        # Without kind, this should fail (no-op on already blocked)
        result = kb.block_task(conn, task_id, reason="no kind given")
        assert result is False
