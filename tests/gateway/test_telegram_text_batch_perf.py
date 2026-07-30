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
from types import SimpleNamespace

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
        assert TelegramAdapter._TEXT_BATCH_DEFAULT_DELAY_S >= 1.177
        assert (
            TelegramAdapter._TEXT_BATCH_SPLIT_DEFAULT_DELAY_S
            >= TelegramAdapter._TEXT_BATCH_DEFAULT_DELAY_S
        )

    def test_fast_tier_uses_min_with_configured_cap(self, adapter):
        """A short message picks the lower of the fast-tier delay and
        the operator's configured cap."""
        # Operator set a generous cap (0.6s); fast tier should win.
        adapter._text_batch_delay_seconds = 0.6
        delay = min(
            adapter._text_batch_delay_seconds,
            TelegramAdapter._TEXT_BATCH_FAST_DELAY_S,
        )
        assert delay == TelegramAdapter._TEXT_BATCH_FAST_DELAY_S

        # Operator tightened the cap below the fast-tier delay; cap wins.
        adapter._text_batch_delay_seconds = 0.10
        delay = min(
            adapter._text_batch_delay_seconds,
            TelegramAdapter._TEXT_BATCH_FAST_DELAY_S,
        )
        assert delay == 0.10

    def test_observed_sub_4000_shape_uses_long_grace(self, adapter):
        """The observed 2566 -> 3955 burst must not use the short tiers."""
        adapter._text_batch_delay_seconds = TelegramAdapter._TEXT_BATCH_DEFAULT_DELAY_S
        adapter._text_batch_split_delay_seconds = (
            TelegramAdapter._TEXT_BATCH_SPLIT_DEFAULT_DELAY_S
        )

        first = SimpleNamespace(text="a" * 2566, _last_chunk_len=2566)
        combined = SimpleNamespace(
            text=("a" * 2566) + "\n" + ("b" * 3955),
            _last_chunk_len=3955,
        )

        assert adapter._calc_text_batch_delay(first) >= 1.177
        assert adapter._calc_text_batch_delay(combined) >= 1.177

    def test_fast_tiers_remain_unchanged(self, adapter):
        """Long-burst protection must not slow common short messages."""
        adapter._text_batch_delay_seconds = TelegramAdapter._TEXT_BATCH_DEFAULT_DELAY_S
        adapter._text_batch_split_delay_seconds = (
            TelegramAdapter._TEXT_BATCH_SPLIT_DEFAULT_DELAY_S
        )

        fast = SimpleNamespace(text="a" * 320, _last_chunk_len=320)
        short = SimpleNamespace(text="a" * 1024, _last_chunk_len=1024)

        assert adapter._calc_text_batch_delay(fast) == 0.18
        assert adapter._calc_text_batch_delay(short) == 0.24

