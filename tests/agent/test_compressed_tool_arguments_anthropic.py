import json

from agent.anthropic_adapter import _convert_assistant_message
from agent.tool_argument_integrity import (
    INCOMPLETE_TOOL_ARGUMENTS_KEY,
    neutralize_completed_incomplete_tool_calls,
)


def tc(call_id, marker):
    args = {INCOMPLETE_TOOL_ARGUMENTS_KEY: {"version": 1}} if marker else {"path": "ok"}
    return {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": json.dumps(args)}}


def test_anthropic_affected_turn_discards_signed_sidecar_and_keeps_complete_call():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tc("bad", True), tc("good", False)],
            "anthropic_content_blocks": [
                {"type": "thinking", "thinking": "private", "signature": "sig"},
                {"type": "tool_use", "id": "bad", "name": "read_file", "input": {}},
                {"type": "tool_use", "id": "good", "name": "read_file", "input": {}},
            ],
        },
        {"role": "tool", "tool_call_id": "bad", "content": "blocked"},
        {"role": "tool", "tool_call_id": "good", "content": "ok"},
    ]
    wire = neutralize_completed_incomplete_tool_calls(messages)
    converted = _convert_assistant_message(wire[0])
    blocks = converted["content"]
    assert not any(b.get("type") == "thinking" for b in blocks)
    assert "anthropic_content_blocks" not in wire[0]
    assert [b.get("id") for b in blocks if b.get("type") == "tool_use"] == ["good"]
    assert any("compressed historical tool call" in b.get("text", "") for b in blocks)
    assert [m.get("tool_call_id") for m in wire if m.get("role") == "tool"] == ["good"]
