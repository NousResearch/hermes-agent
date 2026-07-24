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


def test_substantive_tool_only_turn_invalidates_older_housekeeping_fallback():
    """
    Regression test for #63860.

    A cached `_last_content_with_tools` response from a housekeeping-only turn
    must not survive a later substantive tool-only turn. When the model returns
    an empty response after the substantive tool turn, the system should enter
    the post-tool nudge path, not use the stale housekeeping fallback.

    Production impact: scheduled cron jobs could return early without
    completing their actual work (e.g., daily report job returning a
    housekeeping message instead of producing the report artifact).

    Test sequence:
    1. Content + todo (housekeeping) → sets fallback, marks as all-housekeeping
    2. Empty content + web_search (substantive) → should CLEAR old fallback
    3. Empty content, no tool calls → should enter post-tool nudge, not use old fallback
    4. Content "Recovered after nudge." → should be returned as final response

    Before the fix:
    - Step 2 would not clear the fallback state (no visible content)
    - Step 3 would incorrectly use the housekeeping fallback from step 1
    - API calls would stop at 3, never reaching the nudge response

    After the fix:
    - Step 2 classifies tools and clears the fallback because web_search is substantive
    - Step 3 enters the post-tool nudge path (no stale housekeeping fallback available)
    - Step 4 returns the nudge response as the final answer
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
        # Turn 1: Content + housekeeping tool
        _response(
            content="I'll begin the work.",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("todo", "todo1")],
        ),
        # Turn 2: Empty content + substantive tool (should clear stale fallback)
        _response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("web_search", "search1")],
        ),
        # Turn 3: Empty response (should enter nudge path, not use stale fallback)
        _response(content="", finish_reason="stop"),
        # Turn 4: Nudge response
        _response(content="Recovered after nudge.", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the full task")

    assert result["final_response"] == "Recovered after nudge.", (
        f"Expected nudge recovery response, got: {result['final_response']}. "
        f"This indicates the stale housekeeping fallback was incorrectly used."
    )
    assert result["api_calls"] == 4, (
        f"Expected 4 API calls (including nudge), got: {result['api_calls']}. "
        f"This indicates the conversation exited early without retrying."
    )
    assert result["turn_exit_reason"].startswith("text_response"), (
        f"Expected text_response exit, got: {result['turn_exit_reason']}. "
        f"This indicates the wrong fallback path was taken."
    )


def test_housekeeping_only_turn_still_sets_fallback():
    """Regression: pure housekeeping turns (content + only housekeeping tools)
    must still set the fallback so the post-response mute path works.  This
    verifies the fix doesn't break the original use case the fallback was
    designed for.

    Since #65600 the shortcut is deferred past one nudge retry: the first
    empty completion gets a nudge, and only a SECOND consecutive empty is
    treated as "the model is done" and served the fallback content.
    """
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
        # Turn 1: Content + housekeeping tool (should set fallback)
        _response(
            content="You're welcome!",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("memory", "mem1")],
        ),
        # Turn 2: Empty response (should spend the one nudge retry)
        _response(content="", finish_reason="stop"),
        # Turn 3: Empty again (nudge spent — NOW use the housekeeping fallback)
        _response(content="", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("save this")

    assert result["final_response"] == "You're welcome!", (
        f"Expected housekeeping fallback content, got: {result['final_response']}. "
        f"Pure housekeeping turns should still set the fallback."
    )
    assert "fallback_prior_turn_content" in result.get("turn_exit_reason", ""), (
        f"Expected fallback_prior_turn_content exit, got: {result['turn_exit_reason']}."
    )
    assert result["api_calls"] == 3, (
        f"Expected 3 API calls (housekeeping turn, empty, nudged empty), "
        f"got: {result['api_calls']}."
    )


def test_housekeeping_empty_follow_up_gets_one_nudge_before_fallback():
    """
    Regression test for #65600.

    An empty completion right after a housekeeping-only tool turn is not
    always "the model has nothing more to say" — weak/quantized local models
    intermittently choke and return empty mid-task.  The shortcut used to
    fire immediately, recycling the prior turn's narration as the final
    response and silently ending the turn.

    Test sequence:
    1. Content + skill_manage (housekeeping) → sets fallback
    2. Empty completion → must get the standard post-tool nudge, NOT the
       immediate fallback shortcut
    3. Model recovers with real content → that content is the final response

    Before the fix: step 2 took the shortcut, the final response was the
    recycled step-1 narration, and the model never got a chance to finish
    (2 API calls, reason=fallback_prior_turn_content).
    """
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs("skill_manage")),
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
    agent.valid_tool_names = {"skill_manage"}
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        # Turn 1: Mid-task narration + housekeeping tool (sets fallback)
        _response(
            content="Reviewing the skill library now.",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("skill_manage", "skill1")],
        ),
        # Turn 2: Model chokes — genuinely empty, but NOT done
        _response(content="", finish_reason="stop"),
        # Turn 3: Nudge lands, model finishes the task for real
        _response(content="Skill library updated with two new entries.", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("review the skill library")

    assert result["final_response"] == "Skill library updated with two new entries.", (
        f"Expected the post-nudge recovery response, got: {result['final_response']}. "
        f"The housekeeping fallback shortcut fired before the nudge retry."
    )
    assert result["api_calls"] == 3, (
        f"Expected 3 API calls (housekeeping turn, choke, nudge recovery), "
        f"got: {result['api_calls']}. 2 means the shortcut bypassed the nudge."
    )
    assert result["turn_exit_reason"].startswith("text_response"), (
        f"Expected text_response exit, got: {result['turn_exit_reason']}."
    )