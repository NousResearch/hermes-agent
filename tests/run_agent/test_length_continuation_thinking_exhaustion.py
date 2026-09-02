"""Regression tests for thinking-only length truncations.

GLM-5.3-flash on ollama-cloud with reasoning_effort=high can burn the ENTIRE
output cap on reasoning delivered in a separate field and return
finish_reason="length" with NO visible content (verified live: max_tokens=4096
→ completion_tokens=4096, reasoning ~18.5KB, content empty).

The old continuation flow handled this badly:
  1. the empty response was appended as an interim assistant fragment,
     poisoning the transcript until the pre-call sanitizer "healed" it
     (observed 3+ healings per turn);
  2. every continuation re-ran with thinking ON, re-deriving — and re-burning
     — the whole thinking budget against a growing context, so 4 attempts
     still produced nothing and the turn died with
     "Response remained truncated after 4 continuation attempts".

The fix: skip empty interim fragments, and issue the continuation with a
one-shot reasoning-off override so the budget goes to writing the answer.
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


class TestReasoningOffOneShotOverride:
    def test_flag_consumed_exactly_once(self):
        from agent.chat_completion_helpers import _reasoning_config_for_wire

        agent = _AgentStandIn({"enabled": True, "effort": "high"})
        # Without the flag the reasoning config passes through untouched.
        assert _reasoning_config_for_wire(agent) == {
            "enabled": True,
            "effort": "high",
        }

        agent._ephemeral_reasoning_off = True
        cfg = _reasoning_config_for_wire(agent)
        assert cfg["enabled"] is False
        assert cfg["effort"] == "none"
        assert agent._ephemeral_reasoning_off is False, (
            "The one-shot override must be consumed by the first call."
        )

        # Subsequent calls keep the user's own reasoning config.
        assert _reasoning_config_for_wire(agent) == {
            "enabled": True,
            "effort": "high",
        }

    def test_flag_with_no_user_reasoning_config(self):
        from agent.chat_completion_helpers import _reasoning_config_for_wire

        agent = _AgentStandIn(None)
        agent._ephemeral_reasoning_off = True
        cfg = _reasoning_config_for_wire(agent)
        assert cfg == {"enabled": False, "effort": "none"}


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
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


def _thinking_only_length_response():
    """finish_reason='length' with reasoning but zero visible content — the
    live GLM-5.3-flash-on-ollama-cloud shape (normal response id, NOT the
    partial-stream stub)."""
    from tests.run_agent.test_run_agent import _mock_assistant_msg

    return SimpleNamespace(
        id="chatcmpl-thinking-exhausted",
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content=""),
            finish_reason=FINISH_REASON_LENGTH,
        )],
        usage=None,
    )


def _full_response(content):
    from tests.run_agent.test_run_agent import _mock_response

    return _mock_response(content=content, finish_reason="stop")


def _truncated_text_response(content):
    from tests.run_agent.test_run_agent import _mock_response

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
    def test_retry_after_thinking_only_truncation_completes(self, loop_agent):
        """One thinking-only truncation, then a normal answer: the retry must
        drop thinking (one-shot), boost the output cap, and finish the turn."""
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
        assert loop_agent._ephemeral_reasoning_off is False, (
            "The one-shot reasoning-off override must be consumed by the "
            "continuation call."
        )

    def test_thinking_only_truncation_sets_reasoning_off(self, loop_agent):
        from tests.run_agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_length_response(),
            _mock_response(
                content="done", finish_reason=FINISH_REASON_LENGTH
            ),
            _full_response("finally complete."),
        ]
        _run(loop_agent, "write me a long report")

        calls = loop_agent.client.chat.completions.create.call_args_list
        assert len(calls) == 3
        # The thinking-only fragment set the flag; it was consumed by the
        # next call, and the SECOND truncated fragment (which had visible
        # text) does not set it again — so the third call sees thinking ON.
        assert loop_agent._ephemeral_reasoning_off is False

    def test_full_ceiling_with_empty_fragments_still_settles(self, loop_agent):
        """All four attempts thinking-only: the turn must exit through the
        ceiling with an actionable final_response, no poisoned transcript,
        and no leaked reasoning-off flag."""
        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_length_response() for _ in range(4)
        ]
        result = _run(loop_agent, "write me a long report")

        assert result["completed"] is False
        assert result["partial"] is True
        assert "truncated after 4 continuation attempts" in (result.get("error") or "")
        assert result["final_response"], (
            "An all-empty ceiling exit must still surface a user-facing "
            "message instead of an invisible None."
        )
        assert "reasoning" in (result["final_response"] or "").lower()
        assert _no_empty_assistant_rows(result["messages"]) == []
        assert loop_agent._ephemeral_reasoning_off is False, (
            "The ceiling exit must clear the pending one-shot override so the "
            "next turn does not silently lose thinking."
        )

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

class TestReasoningOffReachesTheWire:
    def test_continuation_request_carries_reasoning_off_on_the_wire(self, loop_agent):
        """The flag is only useful if the continuation REQUEST goes out with
        thinking disabled — assert the OpenRouter extra_body, not the flag."""
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
        first = (calls[0].kwargs.get("extra_body") or {}).get("reasoning")
        second = (calls[1].kwargs.get("extra_body") or {}).get("reasoning")
        assert first == {"enabled": True, "effort": "high"}, first
        assert second is not None and second.get("enabled") is False, (
            f"continuation must be sent with thinking off, got {second!r}"
        )

    def test_stale_flag_does_not_leak_into_next_turn(self, loop_agent):
        """A flag armed by a previous turn that never reached build_api_kwargs
        (interrupt/error between arm and consume) must not silently strip
        thinking from the next turn's first request."""
        loop_agent.reasoning_config = {"enabled": True, "effort": "high"}
        loop_agent._supports_reasoning_extra_body = lambda: True
        loop_agent._ephemeral_reasoning_off = True  # stale from a prior turn
        loop_agent.client.chat.completions.create.side_effect = [
            _full_response("fresh turn answer."),
        ]
        result = _run(loop_agent, "hello")
        assert result["completed"] is True
        calls = loop_agent.client.chat.completions.create.call_args_list
        first = (calls[0].kwargs.get("extra_body") or {}).get("reasoning")
        assert first == {"enabled": True, "effort": "high"}, first
