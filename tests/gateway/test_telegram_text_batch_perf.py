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


from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


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


class TestTelegramTextBatchDefaults:
    """Production defaults retain short tiers and merge jittered long text."""

    def test_default_delay_table(self, monkeypatch):
        """Production defaults preserve fast tiers and widen long-text tiers."""
        monkeypatch.delenv("HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS", raising=False)
        monkeypatch.delenv("HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS", raising=False)
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))

        pending = [
            SimpleNamespace(text="a" * 320, _last_chunk_len=320),
            SimpleNamespace(text="b" * 1024, _last_chunk_len=1024),
            SimpleNamespace(text="c" * 1025, _last_chunk_len=1025),
            SimpleNamespace(
                text="d" * TelegramAdapter._SPLIT_THRESHOLD,
                _last_chunk_len=TelegramAdapter._SPLIT_THRESHOLD,
            ),
        ]

        assert [adapter._text_batch_delay_for(event) for event in pending] == [
            0.18,
            0.24,
            1.5,
            2.0,
        ]

    def test_operator_override_below_fast_tier_wins(self, monkeypatch):
        """An operator cap below a fast tier remains the effective delay."""
        monkeypatch.setenv("HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS", "0.1")
        monkeypatch.delenv("HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS", raising=False)
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
        pending = SimpleNamespace(text="short", _last_chunk_len=5)

        assert adapter._text_batch_delay_for(pending) == 0.1
