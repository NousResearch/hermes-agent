"""Consecutive-compaction fixed point + continuation-nudge loop (#109682).

Incident: once the compaction block hit its fixed point, every turn repeated the same
cycle — compaction → ``max_tokens`` truncation → continuation nudge → compaction at
~the same size — with nothing new entering the transcript (4 consecutive compactions
at the same size, 29 truncation continuations in one session).

Near-identical consecutive sizes alone are a normal steady state (any compaction
target lands in the same band), so the loop signature is the NUDGE: two consecutive
compactions inside the ±2% band with only continuation nudges in between and no fresh
user turn. Two protections, both locked here:

* ``ContextCompressor.record_compaction_size`` arms the fixed point, and
  ``should_compress_info`` then defers auto-compaction with reason ``"fixed_point"``;
* ``agent.turn_truncation._continue_text`` stops nudging while armed — the turn ends
  with the partial response and an explanatory notice instead of feeding the loop.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model", threshold_percent=0.85, protect_first_n=1, protect_last_n=1,
            quiet_mode=True,
        )


def _arm(compressor: ContextCompressor, size: int = 51_710) -> None:
    """Replay the incident: two near-identical compactions with a nudge in between."""
    compressor.record_compaction_size(size)
    compressor.note_continuation_nudge()
    compressor.record_compaction_size(size)


class TestFixedPointDetection:
    def test_near_identical_sizes_alone_are_not_a_fixed_point(self):
        """A steady-state compaction target lands in the same band pass after pass —
        size similarity without a nudged interval is not evidence of a loop."""
        compressor = _compressor()
        compressor.record_compaction_size(51_000)
        compressor.record_compaction_size(51_200)
        assert compressor.compaction_fixed_point_reached() is False

    def test_nudge_between_near_identical_compactions_arms(self):
        """The incident's loop: same-size passes whose only new row was the nudge."""
        compressor = _compressor()
        _arm(compressor)
        assert compressor.compaction_fixed_point_reached() is True

    def test_four_consecutive_same_size_passes_stay_armed(self):
        """Passes that keep landing at the same size keep the verdict armed."""
        compressor = _compressor()
        for _ in range(4):
            compressor.record_compaction_size(51_710)
            compressor.note_continuation_nudge()
        assert compressor.compaction_fixed_point_reached() is True

    def test_outside_tolerance_does_not_arm(self):
        """A pass that actually moved the size is progress, not a fixed point."""
        compressor = _compressor()
        compressor.record_compaction_size(51_710)
        compressor.note_continuation_nudge()
        compressor.record_compaction_size(60_000)
        assert compressor.compaction_fixed_point_reached() is False

    def test_armed_gate_defers_auto_compaction(self):
        """Re-running the same compaction cannot change the transcript: defer it."""
        compressor = _compressor()
        _arm(compressor)
        should, reason = compressor.should_compress_info(prompt_tokens=300_000)
        assert (should, reason) == (False, "fixed_point")
        assert compressor._compression_block_reason() == "fixed_point"

    def test_moved_size_clears_the_fixed_point(self):
        compressor = _compressor()
        _arm(compressor)
        compressor.record_compaction_size(80_000)
        assert compressor.compaction_fixed_point_reached() is False
        assert compressor.should_compress_info(prompt_tokens=300_000) == (True, None)

    def test_fresh_user_turn_clears_the_fixed_point(self):
        """New input ends the equilibrium — auto-compaction and nudges are live again."""
        compressor = _compressor()
        _arm(compressor)
        compressor.note_fresh_user_turn()
        assert compressor.compaction_fixed_point_reached() is False
        assert compressor.should_compress_info(prompt_tokens=300_000) == (True, None)

    def test_manual_compress_clears_the_fixed_point(self):
        """``/compress`` is explicit user intent (and may re-scope): never refused."""
        compressor = _compressor()
        _arm(compressor)
        compressor._begin_compress_attempt(current_tokens=300_000, force=True)
        assert compressor.compaction_fixed_point_reached() is False

    def test_session_reset_clears_the_fixed_point(self):
        compressor = _compressor()
        _arm(compressor)
        compressor._reset_session_compaction_state()
        assert compressor.compaction_fixed_point_reached() is False


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent


def _run(agent, message, history=None):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message, conversation_history=history)


class TestNudgeLoopStops:
    def test_fixed_point_mid_turn_stops_the_continuation_loop(self, loop_agent):
        """Compaction arms the fixed point mid-turn and the response truncates: the
        nudge must not fire, or the next iteration re-runs the same cycle forever."""
        from tests.agent.test_run_agent import _mock_response

        compressor = loop_agent.context_compressor
        calls = []

        def responder(*args, **kwargs):
            calls.append(len(calls) + 1)
            if len(calls) == 1:
                _arm(compressor)  # the compaction boundary landing inside this turn
                return _mock_response(content="batch re-read part one ", finish_reason="length")
            return _mock_response(content="should never be requested", finish_reason="stop")

        with patch.object(loop_agent, "_vprint"):
            loop_agent.client.chat.completions.create.side_effect = responder
            result = _run(loop_agent, "keep going")

        assert calls == [1], "a fixed-point truncation must not feed another request"
        assert [m for m in result["messages"] if m.get("_length_continuation_nudge")] == []
        assert result["completed"] is False
        assert "batch re-read part one" in (result["final_response"] or "")
        assert "fixed point" in (result["final_response"] or "")

    def test_same_truncation_nudges_without_a_fixed_point(self, loop_agent):
        """Control: the identical truncation on an unarmed session still gets its
        one continuation — the guard must not break the normal path."""
        from tests.agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content="batch re-read part one ", finish_reason="length"),
            _mock_response(content="and the rest.", finish_reason="stop"),
        ]
        result = _run(loop_agent, "keep going")

        assert loop_agent.client.chat.completions.create.call_count == 2
        assert result["completed"] is True
        assert "and the rest." in (result["final_response"] or "")

    def test_fresh_turn_clears_the_fixed_point_before_nudging(self, loop_agent):
        """A new user turn IS new input: the armed verdict must not outlive turn start."""
        from tests.agent.test_run_agent import _mock_response

        _arm(loop_agent.context_compressor)  # armed before the turn starts
        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content="batch re-read part one ", finish_reason="length"),
            _mock_response(content="and the rest.", finish_reason="stop"),
        ]
        result = _run(loop_agent, "keep going")

        assert loop_agent.client.chat.completions.create.call_count == 2
        assert result["completed"] is True
        assert loop_agent.context_compressor.compaction_fixed_point_reached() is False


class TestOverflowDeferral:
    def test_fixed_point_no_op_reads_as_a_transient_block(self, loop_agent):
        """A fixed-point no-op is a DEFER (like the structural backoff), never proof the
        session is incompressible: the overflow path must soft-defer, not auto-reset."""
        from agent.conversation_compression import (
            _automatic_compression_gate_blocks, compression_blocked_transiently,
        )

        _arm(loop_agent.context_compressor)
        assert _automatic_compression_gate_blocks(loop_agent, bypass_cooldown=False) is True
        assert compression_blocked_transiently(loop_agent) is True
