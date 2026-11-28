"""A turn that ends on three invalid tool names releases its task resources.

That exit returns a partial result straight from tool-call validation, so it never reaches
finalize_turn. Without its own cleanup, a terminal or browser session opened earlier in the turn
stayed alive until the idle reaper found it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tests.agent.test_run_agent import _make_tool_defs, _mock_response, _mock_tool_call


@pytest.fixture
def agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.compression_enabled = False
    a.save_trajectories = False
    return a


def test_three_invalid_tool_names_clean_up_task_resources(agent):
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls",
                       tool_calls=[_mock_tool_call(name="no_such_tool", arguments="{}", call_id=f"c{i}")])
        for i in range(3)
    ]
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources") as cleanup,
    ):
        result = agent.run_conversation("do something", task_id="task-42")

    assert result["partial"] is True
    assert "invalid tool call" in result["final_response"]
    cleanup.assert_called_with("task-42")


def test_truncated_tool_args_exit_still_cleans_up(agent):
    agent.valid_tool_names.add("write_file")
    # A router that rewrites ``length`` to ``tool_calls`` hands over cut-off args.
    cut = _mock_tool_call(name="write_file", arguments='{"path":"a.md","content":"part', call_id="c1")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[cut]),
    ]
    with (
        patch("model_tools.handle_function_call", return_value='{"success":true}'),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources") as cleanup,
    ):
        result = agent.run_conversation("write it", task_id="task-7")

    assert result["partial"] is True
    cleanup.assert_called_with("task-7")
