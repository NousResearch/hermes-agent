"""Tests for the gateway/agent boot-time 'Compression configured:' log line.

Investigating an in-the-wild log read like:

    Preflight compression: ~67,042 tokens >= 64,000 threshold
    (model gpt-5.5, ctx 272,000)

64,000 == 0.20 * 320,000 — the OLD default. With the live config of
0.30 * 272,000 = 81,600 the line should have said 81,600. We want a
dedicated boot log that prints the resolved compression settings so a
stale-config / stale-runtime mismatch is unambiguous in agent.log.

Also covers the underlying ContextCompressor math: with threshold=0.30
and context_length=272_000 the configured trigger MUST be 81_600
tokens (above the 64_000 MINIMUM_CONTEXT_LENGTH floor).
"""

import logging
from unittest.mock import patch

import pytest

from agent.conversation_compression import format_compression_boot_log
from agent.context_compressor import ContextCompressor


# ── ContextCompressor math: threshold actually reads config ───────────


def test_threshold_tokens_respects_config_threshold_above_floor():
    """0.30 * 272K = 81,600 — above the 64K floor, so floor must not win."""
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=272_000,
    ):
        comp = ContextCompressor(
            model="gpt-5.5",
            threshold_percent=0.30,
            protect_first_n=3,
            protect_last_n=8,
            summary_target_ratio=0.10,
            quiet_mode=True,
        )
    assert comp.context_length == 272_000
    assert comp.threshold_tokens == 81_600


def test_threshold_floor_kicks_in_below_minimum():
    """0.10 * 200K = 20K → floored to 64K minimum."""
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=200_000,
    ):
        comp = ContextCompressor(
            model="x",
            threshold_percent=0.10,
            quiet_mode=True,
        )
    assert comp.threshold_tokens == 64_000


# ── Boot log formatter ────────────────────────────────────────────────


def test_format_boot_log_contains_resolved_values():
    """The formatted INFO line must surface every configured knob so a
    stale-config / stale-runtime mismatch is visible at a glance."""
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=272_000,
    ):
        comp = ContextCompressor(
            model="gpt-5.5",
            threshold_percent=0.30,
            protect_first_n=3,
            protect_last_n=8,
            summary_target_ratio=0.10,
            quiet_mode=True,
        )
    line = format_compression_boot_log(comp, enabled=True)
    assert "Compression configured" in line
    assert "threshold=30%" in line
    assert "target=10%" in line
    assert "protect_first=3" in line
    assert "protect_last=8" in line
    assert "context_len=272000" in line
    assert "trigger_at=81600 tokens" in line


def test_format_boot_log_marks_disabled():
    """When compression is disabled, the line says so."""
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=200_000,
    ):
        comp = ContextCompressor(model="x", quiet_mode=True)
    line = format_compression_boot_log(comp, enabled=False)
    assert "disabled" in line.lower()


def test_format_boot_log_emitted_via_logger(caplog):
    """Confirm the helper logs at INFO so it lands in agent.log."""
    from agent.conversation_compression import log_compression_configured

    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=272_000,
    ):
        comp = ContextCompressor(
            model="gpt-5.5",
            threshold_percent=0.30,
            protect_first_n=3,
            protect_last_n=8,
            summary_target_ratio=0.10,
            quiet_mode=True,
        )

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        log_compression_configured(comp, enabled=True)
    matches = [r for r in caplog.records if "Compression configured" in r.getMessage()]
    assert matches, f"expected 'Compression configured' INFO log, got {[r.getMessage() for r in caplog.records]}"
    assert "trigger_at=81600 tokens" in matches[-1].getMessage()
