"""A steer delivered as a user turn after tool results must serialize to a
valid Anthropic turn: one user message holding the tool_result block(s)
followed by the steer text block. Guards the _merge_consecutive_roles path."""
from agent.anthropic_adapter import convert_messages_to_anthropic


def test_steer_user_turn_merges_into_tool_result_turn():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc1", "type": "function",
             "function": {"name": "terminal", "arguments": "{}"}}
        ]},
        {"role": "tool", "content": "tool output", "tool_call_id": "tc1",
         "name": "terminal", "tool_name": "terminal"},
        {"role": "user", "content": "[The user sent this mid-task via /steer]\nstop and ask first"},
    ]
    _system, result = convert_messages_to_anthropic(messages)
    # No two consecutive user messages survive.
    roles = [m["role"] for m in result]
    for i in range(1, len(roles)):
        assert not (roles[i] == "user" and roles[i - 1] == "user")
    # The steer text rides in the same user turn as the tool_result block.
    user_turns = [m for m in result if m["role"] == "user"]
    merged = user_turns[-1]
    assert isinstance(merged["content"], list)
    types = [b.get("type") for b in merged["content"]]
    assert "tool_result" in types
    assert any(
        b.get("type") == "text" and "stop and ask first" in b.get("text", "")
        for b in merged["content"]
    )
