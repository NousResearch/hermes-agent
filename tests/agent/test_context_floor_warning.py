"""Tests for the small-context threshold floor behavior (#91007).

The floor itself is by design; the bug is that applying it was silent. These
tests pin: (a) the raise-only floor math, (b) the new warning when the floor
overrides a configured value, (c) silence when the configured value already
meets the floor or the window is large.
"""

import logging

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _SMALL_CTX_THRESHOLD_PERCENT,
    _SMALL_CTX_WINDOW_LIMIT,
)


class TestSmallContextFloorWarning:
    def test_floor_still_raises_low_values(self):
        eff = ContextCompressor._effective_threshold_percent(262_144, 0.6)
        assert eff == _SMALL_CTX_THRESHOLD_PERCENT

    def test_higher_configured_value_wins(self):
        eff = ContextCompressor._effective_threshold_percent(262_144, 0.85)
        assert eff == 0.85

    def test_large_context_keeps_configured_value(self):
        eff = ContextCompressor._effective_threshold_percent(600_000, 0.6)
        assert eff == 0.6

    def test_override_emits_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
            eff = ContextCompressor._effective_threshold_percent(262_144, 0.6)
        assert eff == _SMALL_CTX_THRESHOLD_PERCENT
        assert any(
            "raised" in r.message and "compression.threshold" in r.message
            for r in caplog.records
        ), "the silent clamp must now log a warning naming the setting"

    def test_no_warning_when_config_meets_floor(self, caplog):
        with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
            eff = ContextCompressor._effective_threshold_percent(262_144, 0.75)
        assert eff == 0.75
        assert not any(
            "raised" in r.message for r in caplog.records
        ), "no override happened — nothing to warn about"

    def test_no_warning_on_large_context(self, caplog):
        with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
            eff = ContextCompressor._effective_threshold_percent(600_000, 0.6)
        assert eff == 0.6
        assert not any("raised" in r.message for r in caplog.records)
