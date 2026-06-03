from agent.eval_lab.capture import capture_tool_call


def test_capture_tool_call_records_redacted_args_duration_and_result():
    record = capture_tool_call(
        tool_name="terminal",
        tool_args={"command": "echo ok", "api_token": "secret-value"},
        duration_ms=12,
        status="ok",
    )

    assert record.tool_name == "terminal"
    assert record.tool_args_redacted["api_token"] == "[REDACTED]"
    assert record.duration_ms == 12
    assert record.error is None


def test_capture_tool_call_preserves_error_without_credentials():
    record = capture_tool_call(
        tool_name="web",
        tool_args={"authorization": "Bearer abcdefghijklmnop"},
        duration_ms=5,
        status="error",
        error="failed with Bearer abcdefghijklmnop",
    )

    assert record.tool_args_redacted["authorization"] == "[REDACTED]"
    assert record.error == "failed with [REDACTED]"
