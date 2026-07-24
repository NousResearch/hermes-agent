from gateway.run import _proxy_text_history


def test_proxy_text_history_drops_assistant_tool_call_analysis_content():
    history = [
        {"role": "user", "content": "start roadmap docs"},
        {
            "role": "assistant",
            "content": "Need maybe do architecture audit? stale internal plan",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "todo", "arguments": "{}"},
                }
            ],
            "finish_reason": "tool_calls",
        },
        {"role": "tool", "content": "{}", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "Roadmap docs are the active task.", "finish_reason": "stop"},
        {"role": "assistant", "content": "", "finish_reason": "stop"},
    ]

    assert _proxy_text_history(history) == [
        {"role": "user", "content": "start roadmap docs"},
        {"role": "assistant", "content": "Roadmap docs are the active task."},
    ]


def test_proxy_text_history_drops_tool_call_rows_even_without_finish_reason():
    history = [
        {
            "role": "assistant",
            "content": "Need verify status.",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "terminal"}}],
        },
        {"role": "assistant", "content": "Visible final answer."},
    ]

    assert _proxy_text_history(history) == [
        {"role": "assistant", "content": "Visible final answer."},
    ]
