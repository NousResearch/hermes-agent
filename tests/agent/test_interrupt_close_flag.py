from agent.message_sanitization import close_interrupted_tool_sequence


def _tool_tail_messages():
    return [
        {"role": "user", "content": "finish the migration"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "halfway done"},
    ]


def test_interrupt_close_tail_is_structurally_flagged():
    messages = _tool_tail_messages()

    assert close_interrupted_tool_sequence(messages) is True

    assert messages[-1] == {
        "role": "assistant",
        "content": "Operation interrupted.",
        "_interrupt_close": True,
        "finish_reason": "interrupt_close",
    }


def test_plain_operation_interrupted_assistant_turn_is_distinguishable():
    normal_assistant_turn = {"role": "assistant", "content": "Operation interrupted."}

    assert normal_assistant_turn.get("_interrupt_close") is not True


def _assert_api_valid(messages):
    for index, message in enumerate(messages):
        if index:
            assert message.get("role") != messages[index - 1].get("role")

        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue

        expected_ids = [
            call.get("id")
            for call in message.get("tool_calls", [])
            if isinstance(call, dict) and call.get("id")
        ]
        following = messages[index + 1:index + 1 + len(expected_ids)]
        assert len(following) == len(expected_ids)
        assert [row.get("role") for row in following] == ["tool"] * len(expected_ids)
        assert [row.get("tool_call_id") for row in following] == expected_ids


def test_tool_tail_close_is_api_valid_and_flagged():
    messages = _tool_tail_messages()

    close_interrupted_tool_sequence(messages)

    assert messages[-1]["_interrupt_close"] is True
    _assert_api_valid(messages)


def test_dangling_assistant_tool_call_tail_is_removed_and_flagged():
    messages = [
        {"role": "user", "content": "finish the migration"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }
            ],
        },
    ]

    assert close_interrupted_tool_sequence(messages) is True

    assert messages == [
        {"role": "user", "content": "finish the migration"},
        {
            "role": "assistant",
            "content": "Operation interrupted.",
            "_interrupt_close": True,
            "finish_reason": "interrupt_close",
        },
    ]
    _assert_api_valid(messages)


def test_interrupted_assistant_text_tail_is_marked_and_api_valid():
    messages = [
        {"role": "user", "content": "summarize the repo"},
        {"role": "assistant", "content": "I checked the gateway and"},
    ]

    assert close_interrupted_tool_sequence(messages, interrupted_assistant_tail=True) is True

    assert messages[-1] == {
        "role": "assistant",
        "content": "I checked the gateway and",
        "_interrupt_close": True,
        "finish_reason": "interrupt_close",
    }
    _assert_api_valid(messages)
