from agent.action_stall import (
    ACTION_STALL_CONTINUATION_PREFIX,
    build_action_stall_continuation,
    latest_user_message_is_stall_continuation,
)


def test_build_action_stall_continuation_has_stable_prefix():
    text = build_action_stall_continuation()
    assert text.startswith(ACTION_STALL_CONTINUATION_PREFIX)
    assert "Execute the required tool calls" in text


def test_latest_user_message_detects_stall_after_no_tool_assistant():
    messages = [
        {"role": "user", "content": "Inspect the repo."},
        {"role": "assistant", "content": "I'll inspect the repo now."},
        {"role": "user", "content": build_action_stall_continuation()},
    ]

    assert latest_user_message_is_stall_continuation(messages) is True


def test_latest_user_message_does_not_force_after_assistant_tool_calls():
    messages = [
        {"role": "user", "content": "Run date."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "terminal", "content": "ok"},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": build_action_stall_continuation()},
    ]

    assert latest_user_message_is_stall_continuation(messages) is False


def test_latest_user_message_does_not_force_when_tool_result_is_in_segment():
    messages = [
        {"role": "user", "content": "Run date."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "terminal", "content": "ok"},
        {"role": "user", "content": build_action_stall_continuation()},
    ]

    assert latest_user_message_is_stall_continuation(messages) is False


def test_latest_user_message_ignores_non_marker_user_message():
    messages = [
        {"role": "user", "content": "Inspect the repo."},
        {"role": "assistant", "content": "I'll inspect it."},
        {"role": "user", "content": "continue please"},
    ]

    assert latest_user_message_is_stall_continuation(messages) is False


def test_marker_only_forces_when_it_is_the_request_tail():
    messages = [
        {"role": "user", "content": "Inspect the repo."},
        {"role": "assistant", "content": "I'll inspect it."},
        {"role": "user", "content": build_action_stall_continuation()},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "terminal", "content": "ok"},
    ]

    assert latest_user_message_is_stall_continuation(messages) is False
