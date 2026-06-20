from agent import runtime_status
from model_tools import _emit_post_tool_call_hook


def test_emit_post_tool_call_hook_records_runtime_tool_status_without_plugins():
    runtime_status.clear_session("s-model-tools")

    _emit_post_tool_call_hook(
        function_name="terminal",
        function_args={"command": "false"},
        result={"error": "boom"},
        session_id="s-model-tools",
        tool_call_id="tc-1",
        duration_ms=42,
    )

    snap = runtime_status.snapshot("s-model-tools")
    assert snap["recent_tool"]["name"] == "terminal"
    assert snap["recent_tool"]["status"] == "error"
    assert snap["recent_tool"]["duration_ms"] == 42
    assert snap["recent_tool"]["error_message"] == "boom"
