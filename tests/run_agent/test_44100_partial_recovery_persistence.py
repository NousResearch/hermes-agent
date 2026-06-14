"""Regression tests for issue #44100 — delivered responses must be persisted.

The partial-stream recovery path (empty final message but content already
streamed to the user) used to set ``final_response`` and break WITHOUT
appending an assistant message to ``messages``. ``_persist_session`` then
wrote no assistant row for the turn, so the session DB accumulated
consecutive user messages and the model re-answered all of them on the
next turn.

Pins the contract: when partial-stream recovery fires, the recovered text
is appended to ``messages`` as a real assistant turn before the loop exits,
so it reaches the session DB and the transcript keeps role alternation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_run_agent's fixture)."""
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
        a.tool_delay = 0
        a.compression_enabled = False
        a.save_trajectories = False
        return a


class TestPartialStreamRecoveryPersistence:
    def test_recovered_response_is_appended_as_assistant_message(self, loop_agent):
        """Empty final message + streamed content → recovery must leave the
        recovered text in ``messages`` so _persist_session writes it (#44100)."""
        from tests.run_agent.test_run_agent import _mock_response

        # The aggregated final message is empty, but text was already
        # streamed (and delivered) to the platform. The accumulator is
        # reset before each API call, so populate it from inside the
        # mocked call — when stream deltas would actually arrive.
        def _create(*_a, **_kw):
            loop_agent._record_streamed_assistant_text(
                "Here is the streamed answer."
            )
            return _mock_response(content="", finish_reason="stop")

        loop_agent.client.chat.completions.create.side_effect = _create

        persisted = {}

        def _capture(messages, conversation_history=None):
            persisted["messages"] = list(messages)

        with (
            patch.object(loop_agent, "_persist_session", side_effect=_capture),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("hi")

        assert result["final_response"] == "Here is the streamed answer."
        assert result["turn_exit_reason"] == "partial_stream_recovery"

        messages = persisted["messages"]
        last = messages[-1]
        assert last.get("role") == "assistant", (
            "Partial-stream recovery must append the recovered text as an "
            "assistant message — otherwise the turn is never persisted and "
            "the model re-answers old messages next turn (#44100)."
        )
        assert last.get("content") == "Here is the streamed answer."

    def test_normal_turn_unaffected(self, loop_agent):
        """Control: a normal text response still ends with one assistant
        message carrying the response."""
        from tests.run_agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content="Plain answer.", finish_reason="stop"),
        ]

        persisted = {}

        def _capture(messages, conversation_history=None):
            persisted["messages"] = list(messages)

        with (
            patch.object(loop_agent, "_persist_session", side_effect=_capture),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("hi")

        assert result["final_response"] == "Plain answer."
        assistants = [
            m for m in persisted["messages"]
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        assert len(assistants) == 1
        assert assistants[0].get("content") == "Plain answer."
