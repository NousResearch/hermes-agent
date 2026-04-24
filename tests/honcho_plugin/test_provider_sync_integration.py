"""Integration tests for the sync worker's integration into HonchoMemoryProvider.

Layer 1 (fire-and-forget): sync_turn must return in < 20ms even when the
Honcho backend would block for seconds.

Layer 3 (breaker + backlog): when the breaker trips open, sync_turn tasks
land in the provider's in-memory backlog instead of running.  When the
breaker closes (via probe recovery), the backlog drains on the next
sync_turn call.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.sync_worker import SyncTask


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _make_provider() -> HonchoMemoryProvider:
    provider = HonchoMemoryProvider()
    provider._manager = MagicMock()
    session = MagicMock()
    provider._manager.get_or_create.return_value = session
    provider._session_key = "agent:main:test"
    provider._cron_skipped = False
    provider._config = MagicMock(message_max_chars=25000)
    return provider


class TestLayer1FireAndForget:
    def test_sync_turn_returns_immediately_with_slow_backend(self):
        """sync_turn must not block even if the backend flush takes seconds."""
        provider = _make_provider()

        # Make the flush block for up to 2s.
        flush_started = threading.Event()
        release_flush = threading.Event()

        def slow_flush(_session):
            flush_started.set()
            release_flush.wait(timeout=3.0)

        provider._manager._flush_session.side_effect = slow_flush

        try:
            t0 = time.monotonic()
            provider.sync_turn("hello", "world")
            elapsed = time.monotonic() - t0
            assert elapsed < 0.1, f"sync_turn blocked for {elapsed:.3f}s"
            # Confirm the worker did pick it up
            assert flush_started.wait(timeout=1.0)
        finally:
            release_flush.set()
            provider.shutdown()

    def test_multiple_sync_turns_do_not_serialize_caller(self):
        """Back-to-back sync_turns must not block on prior turn's completion."""
        provider = _make_provider()

        gate = threading.Event()
        provider._manager._flush_session.side_effect = lambda _s: gate.wait(timeout=3.0)

        try:
            t0 = time.monotonic()
            for _ in range(5):
                provider.sync_turn("u", "a")
            elapsed = time.monotonic() - t0
            # Without fire-and-forget, the old code would serialize on
            # the previous turn's join(timeout=5.0).  5 turns × 5s = 25s
            # worst case.  We assert << 1s.
            assert elapsed < 0.2, f"5 sync_turns took {elapsed:.3f}s"
        finally:
            gate.set()
            provider.shutdown()


class TestLayer3BacklogAndBreaker:
    def test_breaker_open_backlogs_task(self):
        """While the breaker is open, sync_turn tasks must land in the backlog."""
        provider = _make_provider()

        # Trip the breaker manually.
        provider._breaker._state = provider._breaker.STATE_OPEN
        provider._breaker._opened_at = float("inf")  # never recover

        try:
            provider.sync_turn("hello", "world")
            # The task should have landed in the backlog rather than run.
            assert len(provider._backlog) == 1
            assert provider._backlog[0].name == "sync_turn"
        finally:
            provider.shutdown()

    def test_backlog_drains_when_breaker_closes(self):
        """Once the breaker closes, next sync_turn drains the backlog."""
        provider = _make_provider()

        # Trip the breaker and enqueue a backlog.
        provider._breaker._state = provider._breaker.STATE_OPEN
        provider._breaker._opened_at = float("inf")
        for _ in range(3):
            provider.sync_turn("u", "a")
        assert len(provider._backlog) == 3

        # Close the breaker (simulating recovery) and trigger another sync.
        provider._breaker.reset()

        try:
            provider.sync_turn("u", "a")
            # One new task + 3 drained = 4 flushes eventually.
            assert _wait_until(
                lambda: provider._manager._flush_session.call_count >= 4,
                timeout=2.0,
            ), (
                "expected >= 4 flushes after recovery, got "
                f"{provider._manager._flush_session.call_count}"
            )
            assert provider._backlog == []
        finally:
            provider.shutdown()

    def test_backlog_honors_max_size(self):
        """Backlog must not grow unbounded during a long outage."""
        provider = _make_provider()
        provider._BACKLOG_MAX = 5
        provider._breaker._state = provider._breaker.STATE_OPEN
        provider._breaker._opened_at = float("inf")

        try:
            for _ in range(20):
                provider.sync_turn("u", "a")
            assert len(provider._backlog) == 5
        finally:
            provider.shutdown()
