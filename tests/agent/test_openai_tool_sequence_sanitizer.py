from agent.chat_completion_helpers import _sanitize_openai_tool_message_sequence


def _assistant_tool_call(call_id: str, name: str = "terminal") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def _tool_result(call_id: str, content: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_drops_orphan_tool_after_user_turn():
    messages = [
        {"role": "user", "content": "run"},
        {"role": "assistant", "tool_calls": [_assistant_tool_call("call_x")]},
        _tool_result("call_x"),
        {"role": "user", "content": "next"},
        _tool_result("call_y"),
    ]

    sanitized = _sanitize_openai_tool_message_sequence(messages)

    assert [m.get("role") for m in sanitized] == ["user", "assistant", "tool", "user"]
    assert "call_x" in [m.get("tool_call_id") for m in sanitized]
    assert "call_y" not in [m.get("tool_call_id") for m in sanitized]


def test_nominal_parallel_tool_block_is_unchanged():
    messages = [
        {"role": "user", "content": "run"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _assistant_tool_call("call_a"),
                _assistant_tool_call("call_b"),
            ],
        },
        _tool_result("call_b"),
        _tool_result("call_a"),
        {"role": "assistant", "content": "done"},
    ]

    assert _sanitize_openai_tool_message_sequence(messages) is messages


def test_strips_assistant_tool_calls_without_contiguous_result():
    messages = [
        {"role": "user", "content": "run"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_assistant_tool_call("call_missing")],
        },
        {"role": "user", "content": "later"},
    ]

    sanitized = _sanitize_openai_tool_message_sequence(messages)

    assert sanitized[1]["role"] == "assistant"
    assert "tool_calls" not in sanitized[1]
    assert sanitized[1]["content"] == ""


def test_filters_unanswered_calls_but_keeps_answered_calls():
    messages = [
        {"role": "user", "content": "run"},
        {
            "role": "assistant",
            "tool_calls": [
                _assistant_tool_call("call_keep"),
                _assistant_tool_call("call_missing"),
            ],
        },
        _tool_result("call_keep"),
        {"role": "user", "content": "later"},
    ]

    sanitized = _sanitize_openai_tool_message_sequence(messages)

    assert [tc["id"] for tc in sanitized[1]["tool_calls"]] == ["call_keep"]
    assert sanitized[2]["tool_call_id"] == "call_keep"


def test_strips_tool_call_without_id():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
    ]

    sanitized = _sanitize_openai_tool_message_sequence(messages)

    assert "tool_calls" not in sanitized[0]
    assert sanitized[0]["content"] == ""


def test_deduplicates_duplicate_declared_tool_call_ids():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                _assistant_tool_call("call_same"),
                _assistant_tool_call("call_same"),
            ],
        },
        _tool_result("call_same"),
    ]

    sanitized = _sanitize_openai_tool_message_sequence(messages)

    assert len(sanitized[0]["tool_calls"]) == 1
    assert sanitized[0]["tool_calls"][0]["id"] == "call_same"
    assert sanitized[1]["tool_call_id"] == "call_same"


def test_matches_responses_call_id_when_id_differs():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "fc_123",
                    "call_id": "call_123",
                    "response_item_id": "fc_123",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        _tool_result("call_123"),
    ]

    sanitized = _sanitize_openai_tool_message_sequence(messages)

    assert sanitized is not messages
    assert sanitized[0]["tool_calls"][0]["id"] == "call_123"
    assert sanitized[0]["tool_calls"][0]["call_id"] == "call_123"
    assert sanitized[1]["tool_call_id"] == "call_123"
