"""Provider-confirmed growth must recover before the anti-thrash timer expires."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import (
    _restore_compressor_attempt_state,
    _snapshot_compressor_attempt_state,
    compress_context,
)


class _Agent:
    def __init__(self, compressor):
        self.context_compressor = compressor
        self.session_id = "ineffective-growth-test"
        self.platform = "cli"
        self.model = "test/model"
        self.provider = "test-provider"
        self.tools = []
        self._compression_feasibility_checked = True
        self.compression_in_place = False
        self._memory_manager = self._session_db = self._cached_system_prompt = None
        self._todo_store = self

    def format_for_injection(self):
        return ""

    def _emit_status(self, _message):
        pass

    def _emit_warning(self, _message):
        pass

    def _invalidate_system_prompt(self):
        self._cached_system_prompt = None

    def _build_system_prompt(self, system_message):
        return system_message

    def commit_memory_session(self, _messages):
        pass


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        cc = ContextCompressor(
            model="test/model", provider="test-provider", quiet_mode=True,
            config_context_length=100_000,
        )
    cc.tail_token_budget = 10
    return cc


def _messages():
    return [{"role": "system", "content": "system prompt"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} " * 200}
        for i in range(20)
    ]


def _strike(cc, tokens):
    cc.record_completed_compaction(used_fallback=False)
    cc.update_from_response({"prompt_tokens": tokens})


def test_real_growth_rearms_one_probe_without_trusting_rough_estimates():
    cc = _compressor()
    cc.threshold_tokens = 32_768
    with patch("agent.context_compressor.time.time", return_value=1000):
        _strike(cc, 35_059)
        _strike(cc, 34_105)
        assert cc.should_compress_info(34_105) == (False, "ineffective")
        # A different estimator is not evidence of new compressible material.
        assert not cc.should_compress(90_000)
        cc.update_from_response({"prompt_tokens": 35_000})
        assert not cc.should_compress(35_000)
        cc.update_from_response({"prompt_tokens": 55_205})
        assert cc.should_compress(55_205)
        assert cc._ineffective_compression_count == 1
        # A probe that stays over target re-trips immediately; repeating the same
        # high-water prompt cannot repeatedly earn another growth probe.
        _strike(cc, 54_000)
        cc.update_from_response({"prompt_tokens": 55_205})
        assert not cc.should_compress(55_205)
        cc.update_from_response({"prompt_tokens": 61_000})
        assert cc.should_compress(61_000)
        # Independent cooldown still gates the recovered compressor.
        cc._summary_failure_cooldown_until = float("inf")
        assert not cc.should_compress(61_000)
        cc._summary_failure_cooldown_until = 0
        _strike(cc, 20_000)
        assert cc._ineffective_compression_count == 0
        assert cc._ineffective_growth_tokens == 0
        cc.on_session_reset()
        assert not cc._ineffective_overflow_probe_used


@pytest.mark.parametrize("structural_block", [False, True])
def test_overflow_gets_one_real_attempt_through_outer_commit(structural_block):
    cc = _compressor()
    cc.threshold_tokens = 32_768
    agent = _Agent(cc)
    live = _messages()
    original = deepcopy(live)
    with patch("agent.context_compressor.time.time", return_value=1000):
        _strike(cc, 35_059)
        _strike(cc, 34_105)
        assert not cc.should_compress(34_105)
        if structural_block:
            cc._structural_no_op_backoff_until = float("inf")
        with patch.object(cc, "_generate_summary", return_value="Recovered summary") as summary:
            result, _ = compress_context(
                agent, live, "system prompt", approx_tokens=110_000,
                bypass_cooldown=True, trigger="overflow",
            )
            assert live == original
            if structural_block:
                summary.assert_not_called()
                assert result is live
                assert not cc._ineffective_overflow_probe_used
                return
            summary.assert_called_once()
            assert len(result) < len(live)
            assert cc._verify_compaction_cleared_threshold
            cc.update_from_response({"prompt_tokens": 54_000})
            repeated, _ = compress_context(
                agent, result, "system prompt", approx_tokens=110_000,
                bypass_cooldown=True, trigger="overflow",
            )
            assert repeated is result
            summary.assert_called_once()
            # Failed/cancelled attempts must retain both probe limits on rollback.
            snapshot = _snapshot_compressor_attempt_state(cc)
            cc.on_session_reset()
            _restore_compressor_attempt_state(cc, snapshot)
            assert cc._ineffective_overflow_probe_used
            assert cc._ineffective_growth_tokens >= 54_000
            assert not cc.should_compress(54_000)
