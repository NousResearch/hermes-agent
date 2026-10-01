"""Persisted message provenance survives gateway replay, but never reaches a provider."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.turn_context import build_api_messages
from gateway.run import _build_gateway_agent_history, _build_replay_entry
from hermes_state import SessionDB


@pytest.mark.parametrize("timestamps", [False, True])
def test_persisted_metadata_survives_reopen_and_stays_off_wire(tmp_path, timestamps):
    path = tmp_path / "state.db"
    sid = "replay-provenance"
    metadata = {
        "user": {"notification_category": "diagnostic", "context_source": {"receipt": "test-receipt", "source_id": 10}},
        "assistant": {"context_source": {"receipt": "test-receipt", "source_id": 12}},
    }
    db = SessionDB(db_path=path)
    try:
        db.create_session(sid, source="telegram", model="test-model")
        for role, content in (("user", "question"), ("assistant", "answer")):
            db.append_message(sid, role, content, timestamp=1234567890.0, display_metadata=metadata[role])
    finally:
        db.close()

    reopened = SessionDB(db_path=path)
    try:
        history = reopened.get_messages_as_conversation(sid)
    finally:
        reopened.close()
    replay, observed = _build_gateway_agent_history(history, inject_timestamps=timestamps)
    assert observed is None
    assert [msg["display_metadata"] for msg in replay] == [metadata["user"], metadata["assistant"]]
    before = deepcopy(replay)

    agent = SimpleNamespace(
        _copy_reasoning_content_for_api=lambda *_: None,
        _should_sanitize_tool_calls=lambda: False,
        ephemeral_system_prompt=None,
        _current_turn_timestamp=1234567891.0,
    )
    wire, _ = build_api_messages(
        agent, replay, current_turn_user_idx=None, ext_prefetch_cache=None,
        plugin_user_context=None, moa_config=None, active_system_prompt="system",
    )
    assert all("display_metadata" not in msg for msg in wire)
    assert replay == before
    assert [msg["content"] for msg in wire[1:]] == [msg["content"] for msg in replay]


@pytest.mark.parametrize("metadata", [None, "not-a-map", ["not-a-map"]])
def test_replay_ignores_invalid_display_metadata(metadata):
    msg = {"role": "user", "content": "question", "display_metadata": metadata}
    replay = _build_replay_entry("user", "question", msg)
    assert "display_metadata" not in replay
    assert replay["role"] == "user" and replay["content"] == "question"
