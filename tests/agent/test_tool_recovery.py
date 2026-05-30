from agent.tool_recovery import empty_tool_result_recovery_response


def test_empty_tool_result_recovery_returns_none_without_recent_tools():
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": ""},
    ]

    assert empty_tool_result_recovery_response(messages) is None


def test_empty_tool_result_recovery_formats_latest_tool_results():
    messages = [
        {"role": "user", "content": "run checks"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": "shell"}}],
        },
        {"role": "tool", "name": "shell", "content": {"ok": True, "count": 2}},
    ]

    response = empty_tool_result_recovery_response(messages)

    assert response is not None
    assert "latest tool results" in response
    assert "`shell`" in response
    assert '"ok": true' in response
    assert '"count": 2' in response


def test_empty_tool_result_recovery_limits_to_recent_tool_block():
    messages = [
        {"role": "tool", "name": "old", "content": "old result"},
        {"role": "user", "content": "new task"},
        {"role": "tool", "name": "new", "content": "new result"},
    ]

    response = empty_tool_result_recovery_response(messages)

    assert response is not None
    assert "new result" in response
    assert "old result" not in response
