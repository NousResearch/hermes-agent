"""Regression tests for thinking-only length truncations.

GLM-5.3-flash on ollama-cloud with reasoning_effort=high can burn the ENTIRE
output cap on reasoning delivered in a separate field and return
finish_reason="length" with NO visible content (verified live: max_tokens=4096
→ completion_tokens=4096, reasoning ~18.5KB, content empty).

Continuations skip empty interim fragments and increase the output cap within
the existing bounded retry ladder. They preserve the session's reasoning
settings rather than forcing a cold-cache reasoning-off request. If the model
still produces no visible answer, the ceiling returns an actionable result.
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


class TestReasoningRejection:
    def test_rejected_disable_resends_users_config_verbatim(self):
        """A rejected disable is omitted, but an enabled session config is preserved."""
        from agent.chat_completion_helpers import _reasoning_config_for_wire

        agent = _AgentStandIn({"enabled": True, "effort": "high"})
        agent._reasoning_disable_rejected = True
        assert _reasoning_config_for_wire(agent) == {"enabled": True, "effort": "high"}

        agent.reasoning_config = {"enabled": False}
        assert _reasoning_config_for_wire(agent) is None

    def test_mandatory_reasoning_route_steps_a_disable_up_to_the_floor(self):
        """"Reasoning is mandatory ... cannot be disabled" understands the field: the session's
        disable becomes the floor effort (closest to what the user asked for), while a config that
        already reasons still goes out verbatim (cache key preserved)."""
        from agent.auxiliary_reasoning_floor import REASONING_FLOOR_EFFORT
        from agent.chat_completion_helpers import _reasoning_config_for_wire

        agent = _AgentStandIn({"enabled": False})
        agent._reasoning_disable_rejected = True
        agent._reasoning_floor_required = True
        assert _reasoning_config_for_wire(agent) == {"enabled": True, "effort": REASONING_FLOOR_EFFORT}

        agent.reasoning_config = {"enabled": True, "effort": "high"}
        assert _reasoning_config_for_wire(agent) == {"enabled": True, "effort": "high"}


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
            reasoning_config={"enabled": True, "effort": "high"},
            max_tokens=4096,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        a._supports_reasoning_extra_body = lambda: True
        return a


def _thinking_only_length_response():
    """finish_reason='length' with reasoning but zero visible content — the
    live GLM-5.3-flash-on-ollama-cloud shape (normal response id, NOT the
    partial-stream stub)."""
    from tests.agent.test_run_agent import _mock_assistant_msg

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
    def test_retry_after_thinking_only_truncation_completes(self, loop_agent):
        """One thinking-only truncation, then a normal answer: the retry must
        preserve thinking, boost the output cap, and finish the turn."""
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
        assert all(c.kwargs["extra_body"]["reasoning"] == {"enabled": True, "effort": "high"}
                   for c in calls)


    def test_full_ceiling_with_empty_fragments_still_settles(self, loop_agent):
        """All four attempts thinking-only: the turn must exit through the
        ceiling with an actionable final_response, no poisoned transcript,
        and unchanged reasoning on every attempt."""
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
        calls = loop_agent.client.chat.completions.create.call_args_list
        assert len(calls) == 4
        assert [c.kwargs.get("max_tokens") for c in calls[1:]] == [8192, 16384, 32768]
        assert all(c.kwargs["extra_body"]["reasoning"] == {"enabled": True, "effort": "high"}
                   for c in calls)
        assert loop_agent.reasoning_config == {"enabled": True, "effort": "high"}

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

class TestReasoningPreservedOnTheWire:
    def test_continuation_request_preserves_reasoning_on_the_wire(self, loop_agent):
        """Assert the OpenRouter request body, not an internal continuation flag."""
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
        assert second == first

    def test_reasoning_and_prefix_stay_stable_across_mixed_continuations(self, loop_agent):
        """Empty and visible-text continuations keep the configured reasoning and prefix."""
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
        assert wire[0] == {"enabled": True, "effort": "high"}, wire
        assert wire == [{"enabled": True, "effort": "high"}] * 3, wire
        assert [c.kwargs.get("max_tokens") for c in calls[1:]] == [8192, 16384]
        system_prompts = {
            c.kwargs["messages"][0]["content"] for c in calls
            if c.kwargs["messages"][0].get("role") == "system"
        }
        assert len(system_prompts) == 1, (
            "system prompt must be byte-identical across the retry sequence "
            "(continuation must not rebuild the prefix)"
        )
