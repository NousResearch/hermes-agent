from types import SimpleNamespace
from unittest.mock import patch

from agent.agent_runtime_helpers import invoke_tool


def test_agent_progress_callback_reaches_registry_dispatch():
    callback = object()
    captured = {}

    def fake_handle_function_call(name, args, task_id, **kwargs):
        captured.update(kwargs)
        return '{"ok":true}'

    agent = SimpleNamespace(
        session_id="session-1",
        valid_tool_names={"web_search"},
        enabled_toolsets=None,
        disabled_toolsets=None,
        tool_progress_callback=callback,
        _memory_manager=None,
        _current_turn_id="turn-1",
        _current_api_request_id="request-1",
    )
    fake_run_agent = SimpleNamespace(handle_function_call=fake_handle_function_call)

    with patch("agent.agent_runtime_helpers._ra", return_value=fake_run_agent):
        result = invoke_tool(
            agent,
            "web_search",
            {"q": "test"},
            "task-1",
            pre_tool_block_checked=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )

    assert result == '{"ok":true}'
    assert captured["progress_callback"] is callback