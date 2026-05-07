"""Tests for the FD watchdog and resource observability module."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from gateway.fd_watchdog import (
    collect_metrics,
    get_cached_client_count,
    get_fd_count,
    get_fd_limit,
    get_thread_count,
    log_metrics,
    should_restart,
)


class TestMetricCollection:
    """Verify individual metric collectors return sane values."""

    def test_fd_count_positive(self):
        count = get_fd_count()
        assert count > 0, "Process must have at least stdin/stdout/stderr open"

    def test_fd_limit_positive(self):
        limit = get_fd_limit()
        assert limit > 0

    def test_thread_count_positive(self):
        count = get_thread_count()
        assert count >= 1, "At least the main thread must be running"

    def test_cached_client_count_non_negative(self):
        count = get_cached_client_count()
        assert count >= 0

    def test_collect_metrics_has_all_keys(self):
        m = collect_metrics()
        for key in ("fd_count", "fd_limit", "fd_usage_pct",
                     "thread_count", "cached_clients", "sockets", "close_wait"):
            assert key in m, f"Missing key: {key}"


class TestShouldRestart:
    """Verify threshold logic for restart decisions."""

    def test_healthy_returns_none(self):
        metrics = {
            "fd_count": 50,
            "fd_limit": 1024,
            "fd_usage_pct": 4.9,
            "close_wait": 50,
        }
        assert should_restart(metrics) is None

    def test_fd_threshold_triggers(self):
        metrics = {
            "fd_count": 750,
            "fd_limit": 1024,
            "fd_usage_pct": 73.2,
            "close_wait": 0,
        }
        reason = should_restart(metrics, fd_pct_threshold=70)
        assert reason is not None
        assert "73.2%" in reason

    def test_close_wait_threshold_triggers(self):
        metrics = {
            "fd_count": 50,
            "fd_limit": 1024,
            "fd_usage_pct": 4.9,
            "close_wait": 200,
        }
        reason = should_restart(metrics, close_wait_threshold=150)
        assert reason is not None
        assert "CLOSE_WAIT" in reason

    def test_custom_thresholds(self):
        metrics = {
            "fd_count": 50,
            "fd_limit": 100,
            "fd_usage_pct": 50.0,
            "close_wait": 10,
        }
        # Default thresholds (70%, 30) — should be fine
        assert should_restart(metrics) is None
        # Tighter thresholds
        assert should_restart(metrics, fd_pct_threshold=40) is not None
        assert should_restart(metrics, close_wait_threshold=5) is not None


class TestLogMetrics:
    """Verify log_metrics runs without error and returns metrics."""

    def test_returns_metrics_dict(self):
        m = log_metrics()
        assert isinstance(m, dict)
        assert "fd_count" in m

    def test_accepts_precomputed_metrics(self):
        fake = {
            "fd_count": 42,
            "fd_limit": 256,
            "fd_usage_pct": 16.4,
            "thread_count": 3,
            "cached_clients": 1,
            "close_wait": 0,
            "sockets": {"ESTABLISHED": 5},
        }
        result = log_metrics(fake)
        assert result is fake


class TestWatchdogLoop:
    """Verify the async watchdog loop behavior."""

    @pytest.mark.asyncio
    async def test_watchdog_cancellation(self):
        """Watchdog should exit cleanly on cancellation."""
        from gateway.fd_watchdog import fd_watchdog_loop

        runner = MagicMock()
        task = asyncio.create_task(fd_watchdog_loop(runner, interval=0.1))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected

    @pytest.mark.asyncio
    async def test_watchdog_triggers_stop_on_threshold(self):
        """When metrics breach threshold, watchdog should call runner.stop()."""
        from gateway.fd_watchdog import fd_watchdog_loop

        stop_called = asyncio.Event()

        class FakeRunner:
            async def stop(self):
                stop_called.set()

        runner = FakeRunner()

        # Mock collect_metrics to return high FD usage
        fake_metrics = {
            "fd_count": 900,
            "fd_limit": 1024,
            "fd_usage_pct": 87.9,
            "thread_count": 5,
            "cached_clients": 3,
            "close_wait": 50,
            "sockets": {"CLOSE_WAIT": 50, "ESTABLISHED": 10},
        }

        with patch("gateway.fd_watchdog.collect_metrics", return_value=fake_metrics):
            with patch("gateway.fd_watchdog._try_cleanup", return_value=0):
                task = asyncio.create_task(
                    fd_watchdog_loop(runner, interval=0.1,
                                     fd_pct_threshold=70, close_wait_threshold=30)
                )
                # Give it time to fire
                await asyncio.sleep(0.3)
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        assert stop_called.is_set(), "runner.stop() should have been called"
