"""Compression must prune folded UID slots at the same positions as tool calls."""

import pytest

from agent.agent_runtime_helpers import _merge_consecutive_assistants
from agent.context_compressor import ContextCompressor
from agent.message_metadata import tool_call_uid_from_history


@pytest.mark.parametrize("mapped_first", [True, False], ids=["mapped-first", "legacy-first"])
@pytest.mark.parametrize("survivor_index", [0, 1], ids=["keep-first", "keep-second"])
def test_compression_preserves_mixed_era_survivor_uid(mapped_first, survivor_index):
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
