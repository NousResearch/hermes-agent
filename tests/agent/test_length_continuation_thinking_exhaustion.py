"""Regression tests for thinking-only length truncations.

GLM-5.3-flash on ollama with reasoning_effort=max can burn the ENTIRE output cap
on reasoning delivered in a separate field and return finish_reason="length"
with NO visible content (verified live: max_tokens=4096 → completion_tokens=4096,
reasoning ~18.5KB, content empty).

The self-heal contract: the thinking channel carries the answer, so it is
RESCUED — the reasoning text is returned as the final response and thinking
stays configured. The old one-shot reasoning-off override (which muted the next
call and dropped the reasoning on the floor) is gone: the next call never runs
with ``{"enabled": False, "effort": "none"}`` after a thinking-only truncation.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import FINISH_REASON_LENGTH


class _AgentStandIn:
    """Minimal agent surface _reasoning_config_for_wire needs."""

    def __init__(self, reasoning_config):
        self.reasoning_config = reasoning_config


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def _thinking_only_length_response(reasoning=None):
    """finish_reason='length' with reasoning but zero visible content — the
    live GLM-5.3-flash-on-ollama shape (normal response id, NOT the
    partial-stream stub)."""
    from tests.agent.test_run_agent import _mock_assistant_msg

    return SimpleNamespace(
        id="chatcmpl-thinking-exhausted",
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content="", reasoning=reasoning),
            finish_reason=FINISH_REASON_LENGTH,
        )],
        usage=None,
    )


def _full_response(content):
    from tests.agent.test_run_agent import _mock_response

    return _mock_response(content=content, finish_reason="stop")


def _truncated_text_response(content):
    from tests.agent.test_run_agent import _mock_response

    return _mock_response(content=content, finish_reason=FINISH_REASON_LENGTH)


def _run(agent, message, history=None):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message, conversation_history=history)


def _no_empty_assistant_rows(messages):
    return [
        m for m in messages
        if m.get("role") == "assistant"
        and not (m.get("content") or "").strip()
        and not m.get("tool_calls")
    ]


class TestThinkingOnlyTruncation:
    def test_retry_after_reasoningless_truncation_completes(self, loop_agent):
        """A truncated response with NO reasoning text at all falls through to the
        continuation ladder — and the continuation runs with thinking still ON."""
        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_length_response(),
            _full_response("Here is the full answer."),
        ]
        result = _run(loop_agent, "write me a long report")

        assert result["completed"] is True
        assert "full answer" in (result["final_response"] or "")
        assert _no_empty_assistant_rows(result["messages"]) == [], (
            "An empty (thinking-only) truncated response must never be "
            "appended to the transcript."
        )

        calls = loop_agent.client.chat.completions.create.call_args_list
        assert len(calls) == 2
        # Continuation retry boosts the output cap (2^1 × 4096 base floor).
        assert calls[1].kwargs.get("max_tokens") == 8192, (
            "The continuation retry must request a larger output budget than "
            "the request that truncated."
        )

    def test_full_ceiling_with_empty_fragments_still_settles(self, loop_agent):
        """All four attempts thinking-only (no reasoning captured): the turn must
        exit through the ceiling with an actionable final_response and no
        poisoned transcript."""
        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_length_response() for _ in range(4)
        ]
        result = _run(loop_agent, "write me a long report")

        assert result["completed"] is False
        assert result["partial"] is True
        assert result["final_response"], (
            "An all-empty ceiling exit must still surface a user-facing "
            "message instead of an invisible None."
        )
        assert _no_empty_assistant_rows(result["messages"]) == []

    def test_mixed_fragments_keep_visible_text(self, loop_agent):
        """A visible fragment followed by a thinking-only one: the visible
        text must be stitched, the empty one skipped."""
        loop_agent.client.chat.completions.create.side_effect = [
            _truncated_text_response("visible part one. "),
            _thinking_only_length_response(),
            _full_response("and the ending."),
        ]
        result = _run(loop_agent, "write me a long report")

        assert result["completed"] is True
        assert "visible part one." in (result["final_response"] or "")
        assert "and the ending." in (result["final_response"] or "")
        assert _no_empty_assistant_rows(result["messages"]) == []


class TestReasoningRescue:
    def test_thinking_only_truncation_rescues_reasoning_as_the_answer(self, loop_agent):
        """Reasoning present, content empty: the reasoning text IS the answer —
        the turn ends with it, no continuation call is burned, and the next
        request keeps the configured reasoning (never a disable)."""
        loop_agent.reasoning_config = {"enabled": True, "effort": "high"}
        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_length_response(reasoning="The answer is 42, because the math says so."),
        ]
        result = _run(loop_agent, "write me a long report")

        assert "The answer is 42" in (result["final_response"] or "")
        # The rescued reasoning stays in the assistant row's reasoning field.
        assistant_rows = [m for m in result["messages"] if m.get("role") == "assistant"]
        assert any("The answer is 42" in (m.get("reasoning") or "") for m in assistant_rows)

    def test_rescue_keeps_thinking_on_the_wire(self, loop_agent):
        """The request AFTER a rescued thinking-only truncation must carry the
        configured reasoning — never ``{"enabled": False, "effort": "none"}``."""
        loop_agent.reasoning_config = {"enabled": True, "effort": "high"}
        loop_agent._supports_reasoning_extra_body = lambda: True
        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_length_response(reasoning="Rescued reasoning."),
            _full_response("Fresh turn answer."),
        ]
        _run(loop_agent, "write me a long report")
        _run(loop_agent, "and now something else")

        wire = [
            (c.kwargs.get("extra_body") or {}).get("reasoning")
            for c in loop_agent.client.chat.completions.create.call_args_list
        ]
        assert wire[0] == {"enabled": True, "effort": "high"}, wire
        assert all(w != {"enabled": False, "effort": "none"} for w in wire), (
            f"no request may go out with reasoning disabled; got {wire!r}"
        )


class TestThinkingStaysOnTheWire:
    def test_continuation_request_keeps_reasoning_configured(self, loop_agent):
        """After a reasoningless thinking-only truncation the continuation call
        runs with the user's configured reasoning — no reasoning-off override."""
        loop_agent.reasoning_config = {"enabled": True, "effort": "high"}
        loop_agent._supports_reasoning_extra_body = lambda: True
        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_length_response(),
            _full_response("Here is the full answer."),
        ]
        result = _run(loop_agent, "write me a long report")
        assert result["completed"] is True

        calls = loop_agent.client.chat.completions.create.call_args_list
        assert len(calls) == 2
        second = (calls[1].kwargs.get("extra_body") or {}).get("reasoning")
        assert second == {"enabled": True, "effort": "high"}, (
            f"continuation must keep thinking configured, got {second!r}"
        )

    def test_reasoning_config_is_stable_across_the_retry_sequence(self, loop_agent):
        """Prompt-cache invariant: the reasoning parameter is part of the
        provider's cache key on config-sensitive providers. Every request of the
        continuation sequence must go out with the same configured reasoning,
        and the system prompt must be byte-identical so no request ever
        rebuilds the prefix."""
        loop_agent.reasoning_config = {"enabled": True, "effort": "high"}
        loop_agent._supports_reasoning_extra_body = lambda: True
        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_length_response(),
            _truncated_text_response("PART ONE of the answer"),
            _full_response(" and PART TWO, done."),
        ]
        result = _run(loop_agent, "write me a long report")
        assert result["completed"] is True
        assert "PART ONE" in result["final_response"]
        assert "PART TWO" in result["final_response"]

        calls = loop_agent.client.chat.completions.create.call_args_list
        assert len(calls) == 3
        wire = [
            (c.kwargs.get("extra_body") or {}).get("reasoning") for c in calls
        ]
        assert wire == [{"enabled": True, "effort": "high"}] * 3, wire
        system_prompts = {
            c.kwargs["messages"][0]["content"] for c in calls
            if c.kwargs["messages"][0].get("role") == "system"
        }
        assert len(system_prompts) == 1, (
            "system prompt must be byte-identical across the retry sequence "
            "(no override may change request parameters beyond the prefix)"
        )