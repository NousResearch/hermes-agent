"""Repeated unanswered human questions stay visible in Blocked."""
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn, kb.create_task(conn, title="Needs a decision")


def test_needs_input_at_limit_stays_blocked(board):
    conn, tid = board
    for recurrence in range(1, 6):
        if recurrence > 1:
            assert kb.unblock_task(conn, tid)
        assert kb.block_task(conn, tid, kind="needs_input", reason="Which option?")
        task = kb.get_task(conn, tid)
        assert (task.status, task.block_kind, task.block_recurrences) == ("blocked", "needs_input", recurrence)
        events = kb.list_events(conn, tid)
        routing = [e for e in events if e.kind in {"blocked", "block_loop_detected", "dependency_wait"}][-1]
        assert routing.kind == "blocked"
        assert routing.payload["recurrences"] == recurrence
        if recurrence >= kb.BLOCK_RECURRENCE_LIMIT:
            assert routing.payload["loop_detected"] is True
            assert routing.payload["limit"] == kb.BLOCK_RECURRENCE_LIMIT
        assert not [e for e in events if e.kind == "block_loop_detected"]


@pytest.mark.parametrize("kind", ["transient", "capability", None])
def test_other_kinds_at_limit_still_route_to_triage(board, kind):
    conn, tid = board
    assert kb.block_task(conn, tid, kind=kind, reason="Same failure")
    assert kb.unblock_task(conn, tid)
    assert kb.block_task(conn, tid, kind=kind, reason="Same failure")
    task = kb.get_task(conn, tid)
    assert (task.status, task.block_kind, task.block_recurrences) == ("triage", kind, 2)
    event = kb.list_events(conn, tid)[-1]
    assert event.kind == "block_loop_detected"
    assert event.payload["recurrences"] == event.payload["limit"] == 2


def test_needs_input_after_different_kind_restarts_count(board):
    conn, tid = board
    assert kb.block_task(conn, tid, kind="capability", reason="Missing tool")
    assert kb.unblock_task(conn, tid)
    assert kb.block_task(conn, tid, kind="needs_input", reason="Which option?")
    assert kb.get_task(conn, tid).block_recurrences == 1
    assert "loop_detected" not in kb.list_events(conn, tid)[-1].payload
