"""Regression test for issue #63860.

A cached ``_last_content_with_tools`` from a prior housekeeping-only turn
(content + todo/memory/etc.) must not survive a later substantive tool-only
turn (content="" + web_search/terminal/etc.). Previously the substantive
turn neither updated nor cleared the old fallback candidate, so when the
model returned empty after the tool result, the conversation loop emitted
the old narration ("I'll begin the work.") as the final answer instead of
entering the post-tool nudge path — wasting the full tool round.

The fix: when a turn has at least one substantive (non-housekeeping) tool
call and no usable content, clear ``_last_content_with_tools`` and
``_last_content_tools_all_housekeeping`` so the downstream fallback check
at line ~4893 skips the prior-turn shortcut and enters the nudge path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from run_agent import AIAgent


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _tool_call(name: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _response(
    *, content: str | None, finish_reason: str, tool_calls: list | None = None
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


class TestSubstantiveToolOnlyTurnClearsStaleHousekeepingFallback:
    """Issue #63860 — regression guard."""

    def test_substantive_tool_only_turn_invalidates_older_housekeeping(self):
        """Turn sequence:

          1. content + todo (housekeeping)    → saves fallback content
          2. content="" + web_search (subst.)  → should INVALIDATE old fallback
          3. content="" + stop                 → should enter nudge, not use stale content
          4. content="Recovered after nudge."  → final answer
        """
        with (
            patch("run_agent.get_tool_definitions", return_value=_tool_defs("todo", "web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
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
            _response(
                content="I'll begin the work.",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("todo", "todo1")],
            ),
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("web_search", "search1")],
            ),
            _response(content="", finish_reason="stop"),
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
            f"Expected nudge recovery, got: {result['final_response']!r}"
        )
        assert result["api_calls"] == 4, (
            f"Expected 4 API calls (3 + 1 nudge follow-up), got {result['api_calls']}"
        )
        assert result["turn_exit_reason"].startswith("text_response"), (
            f"Expected text_response exit, got: {result['turn_exit_reason']}"
        )

    def test_housekeeping_only_turn_still_sets_fallback_content(self):
        """Pure housekeeping turns with content should still set the fallback.

        A turn with content + only housekeeping tools (e.g. "You're welcome!"
        + memory save) should behave exactly as before.
        """
        with (
            patch("run_agent.get_tool_definitions", return_value=_tool_defs("memory", "todo")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent.valid_tool_names = {"memory", "todo"}
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="You're welcome!",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("memory", "mem1")],
            ),
            _response(content="", finish_reason="stop"),
        ]

        with (
            patch("run_agent.handle_function_call", return_value="ok"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("thanks")

        # Housekeeping-only fallback should still fire — the "You're welcome!"
        # answer + memory save is the exact use case the fallback was designed for.
        assert result["final_response"] == "You're welcome!", (
            f"Expected housekeeping fallback to fire, got: {result['final_response']!r}"
        )
        assert result["api_calls"] == 2, (
            f"Expected 2 API calls, got {result['api_calls']}"
        )
