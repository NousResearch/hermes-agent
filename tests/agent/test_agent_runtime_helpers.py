"""Focused transcript sanitizer contracts for runtime tool calls."""

from agent.agent_runtime_helpers import sanitize_api_messages


def test_sanitize_pairs_result_names_with_the_local_assistant_turn():
    messages = [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "reused-call-id",
                    "type": "function",
                    "function": {"name": "first_tool", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "reused-call-id",
            "name": "provider_first_tool",
            "content": "first result",
        },
        {"role": "user", "content": "second"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "reused-call-id",
                    "type": "function",
                    "function": {"name": "second_tool", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "reused-call-id",
            "name": "provider_second_tool",
            "content": "second result",
        },
    ]

    sanitized = sanitize_api_messages(messages)
    results = [message for message in sanitized if message.get("role") == "tool"]

    assert [message["name"] for message in results] == [
        "first_tool",
        "second_tool",
    ]
