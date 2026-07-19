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


class TestDeliverResultWaitsBeforeStandaloneFallback:
    """End-to-end: when live delivery fails, ``_deliver_result`` MUST wait out
    the live adapter's rate-limit cooldown before invoking the standalone
    fallback. Without this wait the standalone adapter — built fresh with its
    own closed breaker — re-triggers the iLink -2 immediately and opens its
    own breaker, double-failing the job (#66928).

    Mirrors the integration pattern in
    ``tests/cron/test_scheduler.py::TestDeliverResultTimeoutCancelsFuture``
    so the assertion model is familiar to anyone who has debugged that path.
    """

    def test_live_failure_with_open_breaker_sleeps_then_falls_back(self):
        from concurrent.futures import Future
        from unittest.mock import AsyncMock, MagicMock

        from gateway.config import Platform

        from cron.scheduler import _deliver_result

        # Live adapter carries an OPEN circuit breaker (10s remaining).
        # Use a generous headroom so the assertion tolerates the few-ms
        # elapsed between setting the cooldown and reading it inside the
        # fallback path (mock setup runs real time.monotonic() once).
        adapter = AsyncMock()
        adapter._rate_limit_circuit_until = time.monotonic() + 10.0

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        loop = MagicMock()
        loop.is_running.return_value = True

        # Live delivery raises a real exception so the fallback branch runs.
        captured_future = Future()
        captured_future.result = MagicMock(side_effect=RuntimeError("adapter exploded"))

        def fake_run_coro(coro, _loop):
            coro.close()
            return captured_future

        job = {
            "id": "rl-cooldown-job",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }

        standalone_send = AsyncMock(return_value={"success": True})

        sleep_calls = []

        def record_sleep(duration):
            sleep_calls.append(duration)

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro), \
             patch("tools.send_message_tool._send_to_platform", new=standalone_send), \
             patch.object(scheduler.time, "sleep", side_effect=record_sleep):
            _deliver_result(
                job,
                "Hello world",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        # 1. The wait ran (10.0s remaining + 0.1s buffer ≈ 10.1s, give or
        #    take the few-ms drift between breaker-set and breaker-read).
        assert len(sleep_calls) == 1, (
            f"expected exactly one cooldown wait, got {sleep_calls!r}"
        )
        assert sleep_calls[0] == pytest.approx(10.1, abs=0.2), (
            f"sleep duration {sleep_calls[0]} should match remaining + buffer"
        )
        # 2. The standalone fallback still ran (the wait does NOT swallow
        #    the delivery — it just sequences it after the cooldown).
        standalone_send.assert_awaited_once()

    def test_live_failure_with_closed_breaker_skips_wait(self):
        """A live adapter without an open breaker must NOT block the fallback.

        Negative control for the integration contract — the helper is a
        no-op when the breaker is closed, and ``_deliver_result`` must
        honour that. Without the contract a healthy adapter would incur
        a pointless sleep on every failed live send.
        """
        from concurrent.futures import Future
        from unittest.mock import AsyncMock, MagicMock

        from gateway.config import Platform

        from cron.scheduler import _deliver_result

        adapter = AsyncMock()
        adapter._rate_limit_circuit_until = 0.0  # closed

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        loop = MagicMock()
        loop.is_running.return_value = True

        captured_future = Future()
        captured_future.result = MagicMock(side_effect=RuntimeError("adapter exploded"))

        def fake_run_coro(coro, _loop):
            coro.close()
            return captured_future

        job = {
            "id": "rl-closed-job",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }

        standalone_send = AsyncMock(return_value={"success": True})

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro), \
             patch("tools.send_message_tool._send_to_platform", new=standalone_send), \
             patch.object(scheduler.time, "sleep") as sleep_mock:
            _deliver_result(
                job,
                "Hello world",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        sleep_mock.assert_not_called()
        standalone_send.assert_awaited_once()