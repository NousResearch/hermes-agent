import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import uuid
import logging

from run_agent import AIAgent

def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]

def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )

def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model")

def test_execute_code_only_turn_refunds_iteration_counter():
    """Verify that when a turn only invokes execute_code, the api_call_count is refunded."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("execute_code")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False

        # First call: returns a tool call to execute_code
        tc = _mock_tool_call(name="execute_code", arguments="{}", call_id="c1")
        resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
        # Second call: stops
        resp2 = _mock_response(content="All done!", finish_reason="stop")

        agent.client.chat.completions.create.side_effect = [resp1, resp2]

        with (
            patch("run_agent.handle_function_call", return_value="execution success") as mock_handle,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("run python code")

        assert result["final_response"] == "All done!"
        # Since the first turn (calling only execute_code) is free and refunded,
        # the total api_calls should be 1 (for the second API call that finished).
        assert result["api_calls"] == 1
        assert agent._api_call_count == 1
