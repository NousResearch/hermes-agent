"""Aux feasibility probe must honour the configured threshold_tokens cap.

Regression for https://github.com/NousResearch/hermes-agent/issues/117093:
_lower_threshold_to_aux_context() assigned compressor.threshold_tokens = aux_context
directly, bypassing the cap every other writer applies through
_apply_threshold_tokens_cap(). With an aux summariser window larger than the cap,
the live compaction trigger silently became the uncapped value and compaction
fired late.

Contract: after the probe, threshold_tokens <= threshold_tokens_cap, while the
probe still lowers the trigger when the aux window is under the cap.
"""

from types import SimpleNamespace

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import _lower_threshold_to_aux_context

CAP = 256_000


def _make_agent(*, main_context=1_000_000, cap=CAP):
    compressor = ContextCompressor(
        "test-main-model",
        config_context_length=main_context,
        threshold_percent=0.50,
        threshold_tokens_cap=cap,
        quiet_mode=True,
    )
    return SimpleNamespace(
        context_compressor=compressor,
        _last_feasibility_notice=None,
        _compression_warning=None,
        _emit_diagnostic_status=lambda msg: None,
        model="test-main-model",
        provider="test",
    )


def test_aux_probe_honours_threshold_tokens_cap():
    """An aux window above the cap must not raise the trigger above the cap."""
    agent = _make_agent()
    assert agent.context_compressor.threshold_tokens == CAP  # capped baseline

    _lower_threshold_to_aux_context(
        agent, aux_model="aux-model", aux_context=500_000,
        aux_provider="test", aux_base_url="",
    )

    assert agent.context_compressor.threshold_tokens == CAP


def test_aux_probe_still_lowers_below_cap():
    """The probe's purpose — lowering to a small aux window — keeps working."""
    agent = _make_agent()

    _lower_threshold_to_aux_context(
        agent, aux_model="aux-model", aux_context=80_000,
        aux_provider="test", aux_base_url="",
    )

    assert agent.context_compressor.threshold_tokens == 80_000
