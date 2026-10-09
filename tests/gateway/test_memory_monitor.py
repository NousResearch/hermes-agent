"""Tests for gateway.memory_monitor — periodic process memory logging.

Ported from cline/cline#10343.  The module logs a structured
``[MEMORY] rss=...MB ...`` line periodically so long-running gateway
leaks show up as a time series in agent.log / gateway.log.
"""

from __future__ import annotations

import logging
import time

import pytest

from gateway import memory_monitor as mm


@pytest.fixture(autouse=True)
def _ensure_monitor_stopped():
    """Every test starts from a clean state and leaves one behind."""
    mm.stop_memory_monitoring(timeout=1.0)
    yield
    mm.stop_memory_monitoring(timeout=1.0)


def test_log_memory_usage_has_grep_friendly_format(caplog):
    caplog.set_level(logging.INFO, logger="gateway.memory_monitor")
    mm.log_memory_usage()
    msg = caplog.records[-1].getMessage()
    # Grep-friendly contract: line starts with [MEMORY] and carries RSS
    # (or 'unavailable'), GC counts, thread count, uptime.
    assert msg.startswith("[MEMORY]"), msg
    assert "rss=" in msg
    assert "gc=" in msg
    assert "threads=" in msg
    assert "uptime=" in msg


def test_start_logs_baseline_and_returns_true(caplog):
    caplog.set_level(logging.INFO, logger="gateway.memory_monitor")
    # Large interval so the background timer never fires during the test —
    # we're only checking the synchronous baseline behavior here.
    started = mm.start_memory_monitoring(interval_seconds=3600.0)
    assert started is True
    assert mm.is_running() is True

    messages = [r.getMessage() for r in caplog.records]
    assert any("[MEMORY] baseline " in m for m in messages), messages
    assert any("Periodic memory monitoring started" in m for m in messages), messages


def test_double_start_is_noop():
    assert mm.start_memory_monitoring(interval_seconds=3600.0) is True
    assert mm.start_memory_monitoring(interval_seconds=3600.0) is False
    assert mm.is_running() is True


def test_stop_logs_shutdown_snapshot(caplog):
    mm.start_memory_monitoring(interval_seconds=3600.0)
    caplog.clear()
    caplog.set_level(logging.INFO, logger="gateway.memory_monitor")
    mm.stop_memory_monitoring(timeout=1.0)
    assert mm.is_running() is False

    messages = [r.getMessage() for r in caplog.records]
    assert any("[MEMORY] shutdown " in m for m in messages), messages
    assert any("Periodic memory monitoring stopped" in m for m in messages), messages


