"""Regression coverage for duplicate Anthropic tool blocks in replayed turns."""

from agent.anthropic_adapter import convert_messages_to_anthropic


def test_duplicate_tool_use_and_tool_result_ids_are_collapsed_per_turn():
    """Replay/merge duplication must not produce an invalid Anthropic payload."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}},
                {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": "first",
        },
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": "duplicate",
        },
    ]

    _, result = convert_messages_to_anthropic(messages)
    assistant = next(message for message in result if message["role"] == "assistant")
    user = next(
        message
        for message in result
        if message["role"] == "user"
        and any(block.get("type") == "tool_result" for block in message["content"])
    )

    assert [block["id"] for block in assistant["content"] if block["type"] == "tool_use"] == ["t1"]
    assert [
        block["tool_use_id"] for block in user["content"] if block["type"] == "tool_result"
    ] == ["t1"]
