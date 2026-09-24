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


