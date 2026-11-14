"""Sequential and concurrent tool execution: Ctrl-C emits a cancelled ``post_tool_call`` hook.

Split out of ``tests/agent/test_run_agent.py`` (code-health size ratchet); shares its fixtures.
"""

import contextlib
import json
from unittest.mock import patch

import pytest

from tests.agent import test_run_agent as _base
from tests.agent.test_run_agent import _mock_assistant_msg, _mock_tool_call

# Reuse the facade's fixtures by name (an import would shadow-redefine ``agent`` — F811).
agent = _base.agent
_mock_plugin_discovery = _base._mock_plugin_discovery


class TestExecuteToolCallsInterruptHook:
    @pytest.mark.parametrize("concurrent", [False, True], ids=["sequential", "concurrent"])
    def test_keyboard_interrupt_emits_cancelled_post_tool_hook(self, agent, monkeypatch, concurrent):
        calls = [_mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id=f"c{i}") for i in (1, 2)]
        mock_msg = _mock_assistant_msg(content="", tool_calls=calls if concurrent else calls[:1])
        messages = []
        hook_calls = []
        agent.session_id = "session-1"
        agent._current_turn_id = "turn-1"
        agent._current_api_request_id = "api-1"

        def _capture_hook(hook_name, **kwargs):
            hook_calls.append((hook_name, kwargs))
            return []

        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _capture_hook)
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)

        with (
            patch("model_tools.handle_function_call", side_effect=KeyboardInterrupt),
            patch("run_agent._set_interrupt"),
            patch("agent.interrupt_control._set_interrupt"),
            # The concurrent worker absorbs Ctrl-C into a cancelled result; sequential re-raises.
            contextlib.nullcontext() if concurrent else pytest.raises(KeyboardInterrupt),
        ):
            run = agent._execute_tool_calls_concurrent if concurrent else agent._execute_tool_calls_sequential
            run(mock_msg, messages, "task-1")

        post_calls = sorted((kwargs for name, kwargs in hook_calls if name == "post_tool_call"), key=lambda kw: kw["tool_call_id"])
        assert len(post_calls) == len(mock_msg.tool_calls)
        assert post_calls[0]["tool_name"] == "web_search"
        assert post_calls[0]["tool_call_id"] == "c1"
        assert post_calls[0]["session_id"] == "session-1"
        assert post_calls[0]["turn_id"] == "turn-1"
        assert post_calls[0]["api_request_id"] == "api-1"
        assert post_calls[0]["status"] == "cancelled"
        assert post_calls[0]["error_type"] == "keyboard_interrupt"
        assert json.loads(post_calls[0]["result"])["status"] == "cancelled"
        # The hook sees the user-interrupt attribution, not the generic "Turn interrupted".
        assert post_calls[0]["error_message"] == "Tool execution cancelled. User interrupt"
