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
        # Turn 2: Empty response (should use the housekeeping fallback)
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


def test_kanban_worker_housekeeping_fallback_does_not_bypass_terminal_tool_guard(monkeypatch):
    """Regression: for a kanban worker (HERMES_KANBAN_TASK set) that has not
    called kanban_complete/kanban_block yet, an empty follow-up after a
    housekeeping-only turn (e.g. `todo`) must NOT take the
    fallback_prior_turn_content shortcut — that shortcut `break`s the turn
    loop immediately, before the kanban-stop guard further down ever runs,
    so a worker that only narrates ("I'll update the todo list") and then
    goes silent would exit clean (rc=0) without a terminal board tool. The
    dispatcher records that as protocol_violation.

    Instead, the empty follow-up must fall through to the post-tool-call
    nudge path (real retry), giving the model a chance to actually call
    kanban_complete before the turn ends.
    """
    monkeypatch.setenv("HERMES_KANBAN_TASK", "board-1:card-42")

    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs("memory", "kanban_complete")),
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
    agent.valid_tool_names = {"memory", "kanban_complete"}
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        # Turn 1: Content + housekeeping tool (would set the fallback).
        _response(
            content="I'll update the board now.",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("memory", "mem1")],
        ),
        # Turn 2: Empty response. Pre-fix: fallback_prior_turn_content fires
        # here and the turn ends clean with no kanban_complete ever called.
        _response(content="", finish_reason="stop"),
        # Turn 3: post-tool-nudge retry reaches the model, which now calls
        # the terminal tool.
        _response(
            content="Calling kanban_complete now.",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("kanban_complete", "kc1")],
        ),
        # Turn 4: final answer, after the terminal tool was recorded.
        _response(content="Board updated.", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the board task")

    assert result["final_response"] == "Board updated.", (
        f"Expected the turn to continue past the empty response and finish "
        f"after kanban_complete was called, got: {result['final_response']}. "
        f"This indicates the housekeeping-fallback shortcut incorrectly "
        f"exited the turn before the kanban-stop guard could run."
    )
    assert result["api_calls"] == 4, (
        f"Expected 4 API calls (including the post-tool nudge retry and the "
        f"kanban_complete turn), got: {result['api_calls']}. A count of 2 "
        f"would mean the shortcut still bypassed the terminal-tool guard."
    )
    assert result.get("turn_exit_reason", "").startswith("text_response"), (
        f"Expected a text_response exit after kanban_complete was recorded, "
        f"got: {result['turn_exit_reason']}."
    )