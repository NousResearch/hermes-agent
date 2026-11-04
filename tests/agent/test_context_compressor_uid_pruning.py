"""Compression must prune folded UID slots at the same positions as tool calls."""

import json

import pytest

from agent.agent_runtime_helpers import _merge_consecutive_assistants
from agent.context_compressor import ContextCompressor
from agent.message_metadata import tool_call_uid_from_history
from hermes_state import SessionDB


@pytest.mark.parametrize("mapped_first", [True, False], ids=["mapped-first", "legacy-first"])
@pytest.mark.parametrize("survivor_index", [0, 1], ids=["keep-first", "keep-second"])
def test_compression_preserves_mixed_era_survivor_uid(mapped_first, survivor_index, tmp_path):
    uid = "b" * 32
    turns = []
    for index in range(2):
        turn = {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"call_x|fc_{index + 1}", "type": "function",
                            "function": {"name": "t", "arguments": "{}"}}],
        }
        if (index == 0) == mapped_first:
            turn["_tool_call_uids"] = {"call_x": uid}
        turns.append(turn)
    merged, repairs = _merge_consecutive_assistants(turns)
    assert repairs == 1
    survivor_id = f"call_x|fc_{survivor_index + 1}"
    result = {"role": "tool", "tool_call_id": f"fc_{survivor_index + 1}", "content": "r"}
    # A later user turn makes the incomplete assistant historical, not in flight.
    history = [merged[0], result, {"role": "user", "content": "continue"}]
    compressor = object.__new__(ContextCompressor)
    compressor.quiet_mode = True
    sanitized = compressor._sanitize_tool_pairs(history)
    assistant = sanitized[0]
    assert [tc["id"] for tc in assistant["tool_calls"]] == [survivor_id]
    expected_uid = uid if (survivor_index == 0) == mapped_first else None
    assert tool_call_uid_from_history(sanitized, 1, {}) == expected_uid
    assert assistant["_tool_call_uids"] == {"call_x": [expected_uid]}

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("parent", source="cli")
        db.append_message("parent", "user", "original")
        assert db.try_acquire_compression_lock("parent", "owner", ttl_seconds=60)
        db.publish_compression_child(
            parent_session_id="parent", child_session_id="child", source="cli",
            messages=[{"role": "user", "content": "summary"}, *sanitized],
            compression_lock_holder="owner",
        )
        stored = db.get_messages("child")
        assert json.loads(stored[1]["tool_call_uids"]) == {"call_x": [expected_uid]}
        assert stored[2]["tool_call_uid"] == expected_uid
    finally:
        db.close()
    db = SessionDB(db_path=db_path)
    try:
        restored = db.get_messages_as_conversation("child", repair_alternation=True)
        assert restored[1]["_tool_call_uids"] == {"call_x": [expected_uid]}
        assert restored[2].get("_tool_call_uid") == expected_uid
        assert tool_call_uid_from_history(restored, 2, {}) == expected_uid
    finally:
        db.close()
