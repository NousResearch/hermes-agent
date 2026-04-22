from run_agent import AIAgent


def test_strip_recovery_artifacts_removes_empty_nudge_pair():
    messages = [
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"ok\":true}"},
        {"role": "assistant", "content": "(empty)"},
        {
            "role": "user",
            "content": "You just executed tool calls but returned an empty response. Please process the tool results above and continue with the task.",
        },
        {"role": "assistant", "content": "Here is the actual follow-up."},
    ]

    cleaned = AIAgent._strip_recovery_artifacts(messages)

    assert cleaned == [
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"ok\":true}"},
        {"role": "assistant", "content": "Here is the actual follow-up."},
    ]


def test_strip_recovery_artifacts_removes_standalone_nudge():
    messages = [
        {"role": "user", "content": "Normal user request"},
        {
            "role": "user",
            "content": "You just executed tool calls but returned an empty response. Please process the tool results above and continue with the task.",
        },
        {"role": "assistant", "content": "Recovered."},
    ]

    cleaned = AIAgent._strip_recovery_artifacts(messages)

    assert cleaned == [
        {"role": "user", "content": "Normal user request"},
        {"role": "assistant", "content": "Recovered."},
    ]
