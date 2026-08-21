"""Regression tests for the Telegram text-batch adaptive-delay fast-path
and _env_float_clamped helper introduced by PR #10388 (Telegram latency
tuning).

The fast-path lets short replies stream near-instantly while keeping the
configured cap as the upper bound, so an operator who tightens the cap
gets the lower number on every tier.

The env-clamped helper guarantees float env vars never produce NaN/Inf
or out-of-bounds values that could break asyncio.sleep().
"""

from __future__ import annotations

import math

import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def adapter():
    """Build a TelegramAdapter shell without going through __init__'s
    network-touching setup. Just need the class for static-method access
    and the instance for instance-method tests."""
    return TelegramAdapter.__new__(TelegramAdapter)


class TestEnvFloatClamped:
    """_env_float_clamped is the fence around every float env var the
    adapter reads — must reject NaN/Inf and honor min/max bounds."""

    def test_rejects_nan(self, monkeypatch):
        monkeypatch.setenv("HERMES_TEST_VAR", "nan")
        result = TelegramAdapter._env_float_clamped("HERMES_TEST_VAR", 0.5)
        assert math.isfinite(result)
        assert result == 0.5

    def test_clamps_below_min(self, monkeypatch):
        monkeypatch.setenv("HERMES_TEST_VAR", "0.01")
        assert TelegramAdapter._env_float_clamped(
            "HERMES_TEST_VAR", 0.5, min_value=0.1,
        ) == 0.1


class TestAdaptiveTextBatchTiers:
    """The fast-path tiers cap delay for short / medium messages.  Tier
    constants must compose with the configured cap (operators who set a
    lower cap get the lower number on every tier)."""

    def test_class_constants_are_sensible(self):
        """Sanity check that the tier constants form a non-overlapping
        ascending ladder."""
        assert TelegramAdapter._TEXT_BATCH_FAST_LEN < TelegramAdapter._TEXT_BATCH_SHORT_LEN
        assert TelegramAdapter._TEXT_BATCH_FAST_DELAY_S < TelegramAdapter._TEXT_BATCH_SHORT_DELAY_S
        assert TelegramAdapter._TEXT_BATCH_FAST_DELAY_S > 0
        assert TelegramAdapter._TEXT_BATCH_SHORT_DELAY_S > 0

    def test_fast_tier_uses_min_with_configured_cap(self, adapter):
        """A short message picks the lower of the fast-tier delay and
        the operator's configured cap when extra is unset."""
        adapter._text_batch_delay_from_config = False
        adapter._text_batch_delay_seconds = 0.6
        adapter._text_batch_split_delay_seconds = 1.0
        delay = adapter._text_batch_quiet_seconds(total_len=10, last_chunk_len=10)
        assert delay == TelegramAdapter._TEXT_BATCH_FAST_DELAY_S

        adapter._text_batch_delay_seconds = 0.10
        delay = adapter._text_batch_quiet_seconds(total_len=10, last_chunk_len=10)
        assert delay == 0.10

    def test_explicit_config_extra_skips_adaptive_caps(self, adapter):
        """platforms.telegram.extra.text_batch_delay_seconds is the quiet
        period even for short bubbles (file + follow-up text)."""
        adapter._text_batch_delay_from_config = True
        adapter._text_batch_delay_seconds = 3.0
        adapter._text_batch_split_delay_seconds = 6.0
        delay = adapter._text_batch_quiet_seconds(total_len=20, last_chunk_len=20)
        assert delay == 3.0

    def test_split_chunk_still_uses_split_delay_when_extra_set(self, adapter):
        adapter._text_batch_delay_from_config = True
        adapter._text_batch_delay_seconds = 3.0
        adapter._text_batch_split_delay_seconds = 6.0
        delay = adapter._text_batch_quiet_seconds(
            total_len=4100, last_chunk_len=TelegramAdapter._SPLIT_THRESHOLD,
        )
        assert delay == 6.0

    def test_extra_has_reads_platform_config(self, adapter):
        from gateway.config import PlatformConfig

        adapter.config = PlatformConfig(
            enabled=True, token="x", extra={"text_batch_delay_seconds": 3.0},
        )
        assert adapter._extra_has("text_batch_delay_seconds") is True
        assert adapter._extra_has("media_batch_delay_seconds") is False
        assert adapter._coerce_float_extra("text_batch_delay_seconds", 0.3) == 3.0
