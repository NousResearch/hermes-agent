"""Regression tests for the truncation-exhaustion fallback escalation.

Mirrors the #32421 content-filter fallback escalation: a primary that hits
``finish_reason=length`` on all 4 continuation attempts is in runaway
generation -- the same failure mode the continuation retries exist to
paper over -- so hammering the SAME backend past the ceiling just re-hits
the same loop. Before this fix, the give-up path at the ceiling returned
"Response remained truncated after 4 continuation attempts" unconditionally
and never consulted the configured fallback chain, even when a healthy
fallback provider was available to finish the turn.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import PARTIAL_STREAM_STUB_ID, FINISH_REASON_LENGTH


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


def _stub(content):
    from tests.run_agent.test_run_agent import _mock_assistant_msg
    return SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content=content),
            finish_reason=FINISH_REASON_LENGTH,
        )],
        usage=None,
    )


def _run(agent, message, history=None):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message, conversation_history=history)


class TestTruncationExhaustionActivatesFallback:
    """A primary stuck emitting finish_reason=length across all 4
    continuation attempts must escalate to the fallback chain instead of
    giving up -- exactly like the content-filter escalation does."""

    def test_fallback_completes_the_turn(self, loop_agent):
        from tests.run_agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _stub("part one "), _stub("part two "),
            _stub("part three "), _stub("part four."),
            _mock_response(content="Done on the fallback provider.", finish_reason="stop"),
        ]
        loop_agent._fallback_chain = [
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.7"},
        ]
        loop_agent._fallback_index = 0
        fb_calls = {"n": 0}

        def _fake_activate(reason=None):
            fb_calls["n"] += 1
            loop_agent._fallback_index = len(loop_agent._fallback_chain)
            return True

        with patch.object(loop_agent, "_try_activate_fallback", side_effect=_fake_activate):
            result = _run(loop_agent, "write me a very long report")

        assert fb_calls["n"] == 1, (
            "Fallback must activate exactly once, right after the 4th "
            "continuation attempt is exhausted -- not before, not more "
            "than once."
        )
        assert loop_agent.client.chat.completions.create.call_count == 5, (
            "4 truncated continuation attempts against the primary, then "
            "exactly one fresh request against the (now-activated) fallback."
        )
        assert result["completed"] is True
        assert result["final_response"] == "Done on the fallback provider.", (
            "The fallback's response must win outright -- the truncated "
            "partial fragments from the primary must not be stitched onto it."
        )
        assert not result.get("error")
        assert result.get("partial") is not True

    def test_no_fallback_configured_preserves_existing_give_up(self, loop_agent):
        """Empty fallback chain: the existing ceiling behavior is
        unchanged -- the turn ends with the stitched partial and the
        original error string."""
        loop_agent.client.chat.completions.create.side_effect = [
            _stub("part one "), _stub("part two "),
            _stub("part three "), _stub("part four."),
        ]
        loop_agent._fallback_chain = []
        loop_agent._fallback_index = 0

        with patch.object(loop_agent, "_try_activate_fallback") as mock_activate:
            result = _run(loop_agent, "write me a very long report")

        mock_activate.assert_not_called()
        assert loop_agent.client.chat.completions.create.call_count == 4
        assert result["completed"] is False
        assert result["partial"] is True
        assert result["error"] == "Response remained truncated after 4 continuation attempts"
        assert "part one" in result["final_response"]
        assert "part four" in result["final_response"]
