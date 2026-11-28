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
        patch("model_tools.get_tool_definitions", return_value=_tool_defs("todo", "web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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
        patch("model_tools.handle_function_call", return_value="ok"),
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


def test_bare_tool_marker_is_not_reused_as_final_response():
    """
    Regression test for #78148.

    A provider/local template can emit a bare bracketed token (e.g. "[memory]")
    as assistant content alongside a tool call. That token is protocol
    scaffolding, not an answer. If it gets cached as `_last_content_with_tools`
    and the following turn is empty, the post-tool fallback replays it as the
    final response — and because it then enters the persisted transcript,
    later context compaction preserves it, letting the model repeat the
    marker in subsequent turns.

    Test sequence:
    1. Content "[memory]" + skill_manage (housekeeping) tool call → the bare
       marker must be discarded, not cached as a fallback.
    2. Empty content, no tool calls → enters the post-tool nudge path since
       no fallback is available.
    3. Content "Recovered after nudge." → returned as the final response.

    Before the fix:
    - Step 1 cached "[memory]" as `_last_content_with_tools`.
    - Step 2 reused it via the empty-response fallback, so the conversation
      never reached step 3 and "[memory]" leaked into the persisted history.

    After the fix:
    - Step 1 strips the bare marker before it is cached or persisted.
    - Step 2 has no fallback available and enters the nudge path instead.
    - Step 3 returns the nudge response as the final answer.
    """
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_defs("skill_manage")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"skill_manage"}
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        # Turn 1: Bare "[memory]" marker + housekeeping tool call.
        _response(
            content="[memory]",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("skill_manage", "skill1")],
        ),
        # Turn 2: Empty response (should enter nudge path, not reuse "[memory]").
        _response(content="", finish_reason="stop"),
        # Turn 3: Nudge response
        _response(content="Recovered after nudge.", finish_reason="stop"),
    ]

    with (
        patch("model_tools.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the full task")

    assert result["final_response"] != "[memory]", (
        "The bare tool-call marker leaked through as the final response — "
        "it should have been discarded before caching/persistence."
    )
    assert result["final_response"] == "Recovered after nudge.", (
        f"Expected nudge recovery response, got: {result['final_response']}."
    )
    assert result["api_calls"] == 3, (
        f"Expected 3 API calls (including nudge), got: {result['api_calls']}."
    )


def test_last_resort_fallback_surfaces_prior_content_after_exhaustion():
    """When the model narrates alongside a *substantive* tool call and then goes silent
    through the nudge, the prefill ladder, every retry and the fallback hop, that
    narration must be surfaced as the final response instead of "(empty)".

    The housekeeping-only reuse path (`_last_content_tools_all_housekeeping`) covers a
    different class — there the model was done.  Here it was mid-task, so the cached
    narration is the only visible text the user ever saw; dropping it to "(empty)" loses
    content the model already produced.

    The exit reason is the already-wired ``fallback_prior_turn_content``, so the explainer
    catalog and the turn finalizer handle it with no new plumbing.
    """
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_defs("terminal")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"terminal"}

    calls = {"n": 0}

    def _fake_create(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Narration + substantive tool call → cached, marked NOT all-housekeeping.
            return _response(
                content="Let me check the config...",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "term1")],
            )
        # Every later turn is empty: nudge → thinking prefill → retries → exhaustion.
        return _response(content="", finish_reason="stop")

    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = _fake_create

    with (
        patch("model_tools.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("check the server config")

    assert result["final_response"] == "Let me check the config...", (
        "Expected the prior-turn narration to be surfaced instead of \"(empty)\", "
        f"got: {result['final_response']!r}"
    )
    assert result["turn_exit_reason"] == "fallback_prior_turn_content", (
        f"Expected fallback_prior_turn_content, got: {result['turn_exit_reason']!r}"
    )
    # The narration was streamed as interim commentary: the client must settle the text
    # it already painted rather than render the same content twice.
    assert result["response_previewed"] is True
    assert result["response_reused"] is True
    assert calls["n"] >= 2, "the empty follow-up turns must have been attempted"


def test_bare_marker_beside_substantive_tool_is_not_resurfaced_at_exhaustion():
    """Over-eager-fallback guard (interaction with #78148).

    A bare bracketed token (``[memory]``) beside a tool call is protocol scaffolding,
    not an answer — ``turn_tool_round`` strips it so it is never cached.  When the
    follow-up turns are all empty, the last-resort path must therefore fall through to
    the ``empty_response_exhausted`` sentinel rather than resurrect the marker.
    """
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_defs("terminal")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"terminal"}

    calls = {"n": 0}

    def _fake_create(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Bare marker + substantive tool → the marker is discarded, not cached.
            return _response(
                content="[memory]",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("terminal", "term1")],
            )
        return _response(content="", finish_reason="stop")

    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = _fake_create

    with (
        patch("model_tools.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the full task")

    assert result["final_response"] != "[memory]", (
        "the bare tool-call marker leaked through the last-resort fallback"
    )
    assert result["turn_exit_reason"] == "empty_response_exhausted", (
        f"Expected empty_response_exhausted, got: {result['turn_exit_reason']!r}"
    )
    assert calls["n"] >= 2, "the empty follow-up turns must have been attempted"

