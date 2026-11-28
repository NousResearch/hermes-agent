"""The agent's sequential and concurrent registry dispatch forward the platform clarify callback.

Kept out of ``tests/agent/test_run_agent.py`` (code-health size ratchet); shares its fixtures.
"""

from unittest.mock import patch

import pytest

from tests.agent import test_run_agent as _base
from tests.agent.test_run_agent import _mock_assistant_msg, _mock_tool_call

# Reuse the facade's fixtures by name (an import would shadow-redefine ``agent`` — F811).
agent = _base.agent
_mock_plugin_discovery = _base._mock_plugin_discovery


class TestClarifyCallbackForwarding:
    def test_concurrent_forwards_clarify_callback_to_registry_dispatch(self, agent):
        """Concurrent registry dispatch must preserve the platform callback."""
        tc1 = _mock_tool_call(name="web_search", arguments='{"q":"alpha"}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{"q":"beta"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        captured = {}

        def platform_callback(questions):
            return {"answers": {"q0": "USER_RESPONSE"}, "outcome": "submitted"}

        def fake_handle(name, args, task_id, **kwargs):
            captured[kwargs["tool_call_id"]] = kwargs.get("clarify_callback")
            return "ok"

        agent.clarify_callback = platform_callback
        with patch("model_tools.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert captured == {"c1": platform_callback, "c2": platform_callback}

    @pytest.mark.parametrize("quiet_mode", [False, True])
    def test_sequential_forwards_clarify_callback(self, agent, quiet_mode):
        def sentinel(*a, **k):
            return None

        agent.quiet_mode = quiet_mode
        agent.clarify_callback = sentinel
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        with patch("model_tools.handle_function_call", return_value="search result") as mock_hfc:
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")
        assert mock_hfc.call_args.kwargs["clarify_callback"] is sentinel
