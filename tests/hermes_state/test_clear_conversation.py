"""Invariant coverage for the explicit canonical-chat clear (#110229)."""

import os

import pytest

from hermes_state import SessionDB
from hermes_state_errors import SessionTurnLeaseLostError


def test_clear_conversation_keeps_named_lineage_and_fences_active_work(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="desktop", model="model-a", model_config={"provider": "test"})
    db.set_session_title("root", "Bot Chat")
    db.append_message("root", "user", "before compression")
    db.end_session("root", "compression")
    db.create_session("tip", source="desktop", parent_session_id="root", model="model-a")
    db.append_message("tip", "assistant", "summary and current tail", _compressed_summary=True)
    assert db.set_session_title("tip", "Bot Chat")
    db.create_session("other", source="desktop")
    db.append_message("other", "user", "keep me")

    holder = f"pid={os.getpid()}:turn=active"
    assert db.try_acquire_session_turn_lease("tip", holder, ttl_seconds=30)
    with pytest.raises(SessionTurnLeaseLostError):
        db.clear_conversation_by_title("Bot Chat")
    db.release_session_turn_lease("tip", holder)

    cleared = db.clear_conversation_by_title("Bot Chat")

    assert cleared == {
        "root_id": "root",
        "resolved_id": "tip",
        "lineage_ids": ["root", "tip"],
        "messages_cleared": 2,
    }
    assert db.get_session_by_title("Bot Chat")["id"] == "tip"
    assert db.get_compression_tip("root") == "tip"
    assert db.get_session("root")["model_config"] == '{"provider": "test"}'
    assert db.get_messages_as_conversation("tip", include_ancestors=True, include_compacted=True) == []
    assert [message["content"] for message in db.get_messages("other")] == ["keep me"]
