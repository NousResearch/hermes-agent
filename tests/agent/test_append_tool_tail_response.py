"""Delivered non-interrupted failure responses remain visible and durable."""

from agent.message_sanitization import append_tool_tail_response


def _tool_tail():
    return [
        {"role": "user", "content": "edit the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "patch", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok edited"},
    ]


def test_delivered_failure_is_persisted_once_without_inventing_a_response():
    messages = _tool_tail()
    original = [dict(m) for m in messages]
    assert append_tool_tail_response(messages, "") is False
    assert messages == original
    assert append_tool_tail_response(messages, "Response truncated.") is True
    assert messages[-1]["content"] == "Response truncated."
    assert isinstance(messages[-1]["timestamp"], float)
    assert append_tool_tail_response(messages, "Response truncated.") is False
    assert messages[:-1] == original


def test_assistant_tail_is_left_untouched():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "partial reply"},
    ]
    before = [dict(m) for m in messages]
    assert append_tool_tail_response(messages, "interrupted") is False
    assert messages == before


def test_user_tail_is_left_untouched():
    messages = [{"role": "user", "content": "hi"}]
    assert append_tool_tail_response(messages, "Response truncated.") is False
    assert len(messages) == 1
