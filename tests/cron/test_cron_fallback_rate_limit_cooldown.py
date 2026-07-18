"""Regression tests for cron fallback waiting out the live adapter's
rate-limit circuit breaker (issue #66928).

The live adapter and the standalone fallback path each hold their own
``WeixinAdapter`` instance — the standalone path has no way to see that
the live adapter just tripped its circuit breaker, so without
coordination it sends immediately, hits the same iLink -2, and opens
its own breaker, double-failing the job. ``_deliver_result`` waits out
the live adapter's cooldown before invoking the standalone path.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from cron import scheduler


class _FakeAdapter:
    """Minimal stand-in for WeixinAdapter — only the cooldown attribute matters."""

    def __init__(self, cooldown_until: float) -> None:
        self._rate_limit_circuit_until = cooldown_until


class TestWaitForLiveAdapterRateLimitCooldown:
    def test_no_adapter_returns_zero_without_sleeping(self):
        with patch.object(scheduler.time, "sleep") as sleep_mock:
            result = scheduler._wait_for_live_adapter_rate_limit_cooldown(None)
        assert result == 0.0
        sleep_mock.assert_not_called()

    def test_closed_breaker_returns_zero_without_sleeping(self):
        adapter = _FakeAdapter(cooldown_until=0.0)
        with patch.object(scheduler.time, "sleep") as sleep_mock:
            result = scheduler._wait_for_live_adapter_rate_limit_cooldown(adapter)
        assert result == 0.0
        sleep_mock.assert_not_called()

    def test_missing_attribute_returns_zero_without_sleeping(self):
        # Adapters that don't even carry a breaker (Telegram, Slack, …) must
        # not block the fallback path.
        class _NoBreaker:
            pass

        with patch.object(scheduler.time, "sleep") as sleep_mock:
            result = scheduler._wait_for_live_adapter_rate_limit_cooldown(_NoBreaker())
        assert result == 0.0
        sleep_mock.assert_not_called()

    def test_open_breaker_sleeps_for_remaining_plus_buffer(self):
        # Breaker scheduled to expire in 5.0s — helper should sleep 5.1s.
        open_until = time.monotonic() + 5.0
        adapter = _FakeAdapter(cooldown_until=open_until)

        with patch.object(scheduler.time, "sleep") as sleep_mock:
            result = scheduler._wait_for_live_adapter_rate_limit_cooldown(
                adapter, job_id="job_test_001"
            )

        assert result == pytest.approx(5.1, abs=0.05)
        sleep_mock.assert_called_once()
        # The single sleep duration must match the returned duration.
        assert sleep_mock.call_args.args[0] == pytest.approx(5.1, abs=0.05)

    def test_already_expired_breaker_returns_zero(self):
        # Breaker scheduled in the past — must NOT sleep (already safe).
        adapter = _FakeAdapter(cooldown_until=time.monotonic() - 1.0)

        with patch.object(scheduler.time, "sleep") as sleep_mock:
            result = scheduler._wait_for_live_adapter_rate_limit_cooldown(adapter)

        assert result == 0.0
        sleep_mock.assert_not_called()

    def test_non_numeric_breaker_value_returns_zero(self):
        # Defensive: a mis-typed adapter attribute must not crash cron.
        class _WeirdAdapter:
            _rate_limit_circuit_until = "not-a-number"

        with patch.object(scheduler.time, "sleep") as sleep_mock:
            result = scheduler._wait_for_live_adapter_rate_limit_cooldown(_WeirdAdapter())

        assert result == 0.0
        sleep_mock.assert_not_called()