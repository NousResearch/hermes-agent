"""Compression feasibility clamp is bounded by the capped summarizer input (#126767).

``_lower_threshold_to_aux_context`` used to drag the session's compaction trigger down to the
aux model's window whenever the window was below the main threshold — but the summarizer never
receives the whole transcript (its prompt is capped at ``_SUMMARY_INPUT_MAX_CHARS``), so a
window that fits the capped input must keep the policy trigger untouched. Only a window too
small for the capped input still clamps.
"""

from types import SimpleNamespace

from agent.conversation_compression import (
    _SUMMARY_OUTPUT_ALLOWANCE_TOKENS,
    _capped_summary_input_tokens,
    _lower_threshold_to_aux_context,
)


def _stub_agent(threshold_tokens: int = 231_200, context_length: int = 272_000):
    compressor = SimpleNamespace(
        threshold_tokens=threshold_tokens,
        context_length=context_length,
        tail_mode=None,
        summary_target_ratio=None,
    )
    agent = SimpleNamespace(
        context_compressor=compressor,
        model="gpt-6-astra",
        provider="openai-codex",
        _emit_diagnostic_status=lambda msg: None,
    )
    return agent


def _call(agent, aux_context: int):
    _lower_threshold_to_aux_context(
        agent, aux_model="qwen-local", aux_context=aux_context,
        aux_provider="lmstudio", aux_base_url="http://127.0.0.1:1234/v1")


NEEDED = _capped_summary_input_tokens() + _SUMMARY_OUTPUT_ALLOWANCE_TOKENS


class TestFeasibilityClampBoundedByInputCap:
    def test_cap_estimate_is_conservative(self):
        # 160K chars at 3 chars/token, ceil — above the naive 4 chars/token estimate.
        assert _capped_summary_input_tokens() == 53_334
        assert NEEDED == 53_334 + _SUMMARY_OUTPUT_ALLOWANCE_TOKENS

    def test_window_that_fits_capped_input_keeps_threshold(self):
        # The observed regression: a 131K local window dragged a 231K trigger down although the
        # summary prompt is only ~19K tokens.
        agent = _stub_agent(threshold_tokens=231_200)
        _call(agent, aux_context=131_072)
        assert agent.context_compressor.threshold_tokens == 231_200
        assert not hasattr(agent.context_compressor, "_aux_context_ceiling")

    def test_boundary_at_need_skips_the_clamp(self):
        agent = _stub_agent(threshold_tokens=231_200)
        _call(agent, aux_context=NEEDED)
        assert agent.context_compressor.threshold_tokens == 231_200

    def test_window_below_need_still_clamps(self):
        # A genuinely too-small window keeps the old behavior (clamp to its context).
        agent = _stub_agent(threshold_tokens=231_200, context_length=272_000)
        _call(agent, aux_context=NEEDED - 1)
        assert agent.context_compressor.threshold_tokens == NEEDED - 1
        assert agent.context_compressor._aux_context_ceiling == NEEDED - 1

    def test_tiny_window_clamps_and_notices(self):
        agent = _stub_agent(threshold_tokens=231_200, context_length=272_000)
        _call(agent, aux_context=16_384)
        assert agent.context_compressor.threshold_tokens == 16_384
        assert agent.context_compressor.threshold_percent == 16_384 / 272_000
        assert "Auto-lowered" in agent._compression_warning
