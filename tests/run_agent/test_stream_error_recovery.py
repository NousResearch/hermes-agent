"""Regression tests for provider stream-error (``finish_reason="error"``) recovery.

Some providers (observed: gemini-2.5-flash via OpenRouter, 2026-07) abort a
streamed generation and ship the partial turn with the non-standard
``finish_reason="error"``. In the observed field cases the stream died exactly
at the tool-call boundary: the narration arrived intact ("Here's the synopsis
coming right up!"), ``tool_calls`` stayed empty, and the tool never ran.
Before the fix, ``error`` had no branch — ``content_filter``, ``length`` and
``incomplete`` each did — so the errored generation fell through to the
ordinary text exit and was recorded as the model having finished talking: no
retry, no warning. On a long-running deployment this presented as the agent
"stopping partway through with no explanation" (measured at a 5.4% turn death
rate during the worst window).

The fix routes ``finish_reason == "error"`` with no tool calls through the
same bounded re-prompt used for dropped tool calls (3 consecutive stalls,
budget shared and reset after any genuine turn end), with its own nudge
wording since an aborted stream may also cut mid-text with no tool intent.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_run_agent's fixture)
    so we can stage a stream-error response + continuation pair on
    ``.chat.completions.create``."""
    from run_agent import AIAgent
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
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
        return agent


def _stream_error_response(content: str):
    """A response whose stream the provider aborted: partial narration in
    content, no tool calls, ``finish_reason="error"``."""
    from tests.run_agent.test_run_agent import _mock_assistant_msg
    return SimpleNamespace(
        id="chatcmpl-stream-error",
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content=content, tool_calls=None),
            finish_reason="error",
        )],
        usage=None,
    )


class TestStreamErrorRecovery:
    def test_stream_error_reprompts_instead_of_exiting(self, loop_agent):
        """finish_reason=error with no tool calls must re-prompt the model to
        continue rather than recording the aborted generation as the final
        answer."""
        from tests.run_agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _stream_error_response(
                "I'll fetch the video metadata next. Here's the synopsis "
                "coming right up!"
            ),
            _mock_response(content="Synopsis: a short film about a train.",
                           finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("summarize the video")

        assert loop_agent.client.chat.completions.create.call_count == 2, (
            "A stream-error turn must trigger a re-prompt (second API call), "
            "not exit the loop with the aborted generation as the answer."
        )

        # The nudge must tell the model its turn was cut off and to continue.
        second_call = loop_agent.client.chat.completions.create.call_args_list[1]
        msgs = second_call.kwargs.get("messages") or second_call.args[0].get("messages")
        last_user = next(
            (m for m in reversed(msgs) if m.get("role") == "user"), None,
        )
        assert last_user is not None
        assert "cut off" in (last_user.get("content") or "").lower(), (
            "The nudge must say the previous turn was cut off, not accuse the "
            "model of omitting a tool call it may never have intended."
        )
        assert "Synopsis" in result["final_response"]

    def test_persistent_stream_errors_are_bounded(self, loop_agent):
        """If the provider keeps aborting, the recovery must give up after a
        bounded number of consecutive stalls instead of looping forever."""
        from tests.run_agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _stream_error_response("Working on it.") for _ in range(9)
        ] + [_mock_response(content="done", finish_reason="stop")]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("summarize the video")

        # 1 initial call + exactly 3 bounded re-prompts = 4 total. == pins
        # both halves: the recovery fired (not 1) and the bound held (not 9).
        assert loop_agent.client.chat.completions.create.call_count == 4, (
            "Consecutive stream errors must be re-prompted exactly 3 times, "
            "then bounded (no infinite loop, no silent immediate exit)."
        )
        assert result is not None

    def test_stream_error_with_tool_calls_executes_normally(self, loop_agent):
        """An errored stream that DID ship a tool call must take the ordinary
        tool-execution branch, not the re-prompt — the guard's
        ``not assistant_message.tool_calls`` conjunct is load-bearing."""
        from tests.run_agent.test_run_agent import (
            _mock_response,
            _mock_tool_call,
        )

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(
                content="Searching now.",
                finish_reason="error",
                tool_calls=[_mock_tool_call(name="web_search")],
            ),
            _mock_response(content="Found it.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("search for it")

        # The tool call must be answered with a tool-role result, and no
        # re-prompt nudge may be injected for a turn that actually acted.
        second_call = loop_agent.client.chat.completions.create.call_args_list[1]
        msgs = second_call.kwargs.get("messages") or second_call.args[0].get("messages")
        assert any(m.get("role") == "tool" for m in msgs if isinstance(m, dict)), (
            "A tool call on an errored stream must still be executed."
        )
        assert not any(
            isinstance(m, dict) and m.get("_dropped_toolcall_nudge")
            for m in msgs
        ), "No re-prompt nudge may fire when the turn shipped a real call."
        assert "Found it." in result["final_response"]

    def test_stall_budget_resets_at_turn_entry(self, loop_agent):
        """A counter leaked by a prior turn that exited through a path
        bypassing the genuine-turn-end reset must not starve this turn's
        recovery — turn entry zeroes the consecutive-stall budget."""
        from tests.run_agent.test_run_agent import _mock_response

        # Simulate the leak: a previous turn left the budget fully spent.
        loop_agent._dropped_toolcall_retries = 3

        loop_agent.client.chat.completions.create.side_effect = [
            _stream_error_response("Fetching the file now."),
            _mock_response(content="File contents attached.",
                           finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("fetch the file")

        assert loop_agent.client.chat.completions.create.call_count == 2, (
            "A stale stall budget from a prior turn must not suppress "
            "recovery on a fresh turn."
        )
        assert "File contents attached." in result["final_response"]

    def test_error_nudge_pair_is_ephemeral_scaffolding(self, loop_agent):
        """The stream-error re-prompt pair must ride the same ephemeral
        scaffolding flag as the dropped-tool-call nudge, so persistence never
        writes it to the durable transcript."""
        from tests.run_agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _stream_error_response("I'll save the brief to memory now."),
            _mock_response(content="Saved.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("save the brief")

        assert result["completed"] is True
        leftover = [
            m for m in result["messages"]
            if isinstance(m, dict) and m.get("_dropped_toolcall_nudge")
        ]
        assert not leftover, (
            "The re-prompt pair must be stripped at finalization, not kept "
            "in the returned transcript."
        )
