"""Regression tests for conversation loop fallback state management."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _tool_defs(*names):
    """Helper: create minimal tool definitions for given names."""
    return [
        {
            "type": "function", "function": {
                "name": name,
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            }
        }
        for name in names
    ]


def _tool_call(name, call_id):
    """Helper: create a minimal tool call object."""
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _response(*, content, finish_reason, tool_calls=None):
    """Helper: create a minimal API response object."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_model_decides_after_each_tool_round_without_tool_name_classification():
    """Tool names and adjacent narration never decide whether a turn is final.

    Every tool receipt returns to the same model.  Even after an empty response,
    recovery remains model-authored instead of promoting an earlier tool-turn
    narration based on a runtime category such as "housekeeping".
    """
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs("todo", "web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1/",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"todo", "web_search"}
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        # Turn 1: model-authored narration plus a tool call.
        _response(
            content="I'll begin the work.",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("todo", "todo1")],
        ),
        # Turn 2: another tool call, with no narration.
        _response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("web_search", "search1")],
        ),
        # Turn 3: empty response enters the generic model-recovery path.
        _response(content="", finish_reason="stop"),
        # Turn 4: the model authors the terminal response.
        _response(content="Recovered after nudge.", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the full task")

    assert result["final_response"] == "Recovered after nudge."
    assert result["api_calls"] == 4, (
        f"Expected 4 API calls (including nudge), got: {result['api_calls']}. "
        "Every tool receipt and the empty recovery turn must return to the model."
    )
    assert result["turn_exit_reason"].startswith("text_response"), (
        f"Expected text_response exit, got: {result['turn_exit_reason']}. "
        f"This indicates the wrong fallback path was taken."
    )


def test_tool_turn_narration_is_not_promoted_to_final_by_tool_name():
    """A tool-call turn cannot become final through a runtime tool-name list."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs("memory")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1/",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"memory"}
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        # Turn 1: narration accompanies a tool call but is not terminal.
        _response(
            content="You're welcome!",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("memory", "mem1")],
        ),
        # Turn 2: the same model sees the receipt and authors the final answer.
        _response(content="Memory saved and verified.", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("save this")

    assert result["final_response"] == "Memory saved and verified."
    assert result["api_calls"] == 2
    assert result["turn_exit_reason"].startswith("text_response")
