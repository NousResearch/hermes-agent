"""Regression test for #115493: /undo and /retry on a session with an unanswered user turn.

When a turn ends without an assistant reply (e.g. non-retryable provider error or interrupt),
the unanswered user row remains durable. The subsequent user turn triggers pre-request
alternation repair which merges adjacent user messages in memory. Rewind operations must
load durable messages with alternation repair to keep warm and durable user turn projections
aligned, and soft-delete the merged sequence cleanly.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.agent_runtime_helpers import repair_message_sequence_with_cursor
from hermes_cli.cli_session_mixin import CLISessionMixin
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_rewind.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


def test_undo_and_retry_after_unanswered_user_turn(db):
    """Regression test for #115493: session history does not diverge on rewind after repair."""
    sid = "unanswered-turn-session"
    db.create_session(sid, source="cli")

    # Turn 1: user ask that fails with no assistant reply
    db.append_message(sid, "user", "first failed ask")

    # Turn 2: next user ask in the same session
    db.append_message(sid, "user", "second ask")

    # In memory, pre-request alternation repair merges the two user rows
    warm = [
        {"role": "user", "content": "first failed ask"},
        {"role": "user", "content": "second ask"},
    ]
    repairs = repair_message_sequence_with_cursor(None, warm)
    assert repairs == 1
    assert len(warm) == 1
    assert warm[0]["content"] == "first failed ask\n\nsecond ask"

    # Turn 2 succeeds with an assistant answer
    db.append_message(sid, "assistant", "answer to second ask")
    warm.append({"role": "assistant", "content": "answer to second ask"})

    # Setup CLI mixin with warm history
    cli = CLISessionMixin.__new__(CLISessionMixin)
    cli._session_db, cli.session_id, cli.conversation_history = db, sid, list(warm)
    cli.agent = SimpleNamespace(
        _session_messages=cli.conversation_history,
        _last_flushed_db_idx=len(cli.conversation_history),
        _db_flush_scan_prefix=list(cli.conversation_history),
    )
    cli._prefill_input_buffer = MagicMock()

    # Retry should succeed and return the merged retry text
    retried = cli.retry_last()
    assert retried == "first failed ask\n\nsecond ask"
    assert cli.conversation_history == []
    assert db.get_active_message_ids(sid) == []


def test_multi_turn_undo_across_merged_unanswered_turn(db):
    """Subsequent turns can be undone, followed by undoing the merged turn."""
    sid = "multi-turn-unanswered"
    db.create_session(sid, source="cli")

    # Turn 1: normal turn
    db.append_message(sid, "user", "q1")
    db.append_message(sid, "assistant", "a1")

    # Turn 2: failed turn (unanswered user row)
    db.append_message(sid, "user", "q2_failed")

    # Turn 3: next user ask
    db.append_message(sid, "user", "q3")

    warm = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2_failed"},
        {"role": "user", "content": "q3"},
    ]
    repair_message_sequence_with_cursor(None, warm)
    db.append_message(sid, "assistant", "a3")
    warm.append({"role": "assistant", "content": "a3"})

    # Turn 4: normal turn
    db.append_message(sid, "user", "q4")
    db.append_message(sid, "assistant", "a4")
    warm.append({"role": "user", "content": "q4"})
    warm.append({"role": "assistant", "content": "a4"})

    cli = CLISessionMixin.__new__(CLISessionMixin)
    cli._session_db, cli.session_id, cli.conversation_history, cli.agent = db, sid, list(warm), None
    cli._prefill_input_buffer = MagicMock()

    # 1. Undo turn 4
    cli.undo_last(1)
    assert len(cli.conversation_history) == 4
    assert [m["content"] for m in cli.conversation_history if m["role"] == "user"] == [
        "q1", "q2_failed\n\nq3"
    ]

    # 2. Undo turn 3 (the merged turn)
    cli.undo_last(1)
    assert len(cli.conversation_history) == 2
    assert [m["content"] for m in cli.conversation_history if m["role"] == "user"] == ["q1"]

    # 3. Undo turn 1
    cli.undo_last(1)
    assert len(cli.conversation_history) == 0
    assert db.get_active_message_ids(sid) == []


def test_undo_on_resumed_session_with_unanswered_turn(db):
    """When a session with an unanswered turn is resumed, undo still functions without divergence."""
    sid = "resumed-unanswered"
    db.create_session(sid, source="cli")

    db.append_message(sid, "user", "q1_failed")
    db.append_message(sid, "user", "q2")
    db.append_message(sid, "assistant", "a2")

    model_history, _ = db.get_resume_conversations(sid)
    assert len([m for m in model_history if m["role"] == "user"]) == 1

    cli = CLISessionMixin.__new__(CLISessionMixin)
    cli._session_db, cli.session_id, cli.conversation_history, cli.agent = db, sid, list(model_history), None
    cli._prefill_input_buffer = MagicMock()

    cli.undo_last(1)
    assert len(cli.conversation_history) == 0
    assert db.get_active_message_ids(sid) == []
