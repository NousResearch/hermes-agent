"""Tests for the ConcurrencyManager."""

import time
from unittest.mock import patch

import pytest

from gateway.daimon.concurrency import ConcurrencyManager


class TestConcurrencyManagerBasic:
    """Basic acquire/release behavior."""

    def test_initial_state(self):
        mgr = ConcurrencyManager(max_active=5, max_threads_per_day=3)
        assert mgr.active_count == 0
        assert mgr.queue_length == 0

    def test_acquire_slot(self):
        mgr = ConcurrencyManager(max_active=5, max_threads_per_day=10)
        acquired, pos = mgr.try_acquire("thread-1", "user-a")
        assert acquired is True
        assert pos == 0
        assert mgr.active_count == 1

    def test_release_slot(self):
        mgr = ConcurrencyManager(max_active=5, max_threads_per_day=10)
        mgr.try_acquire("thread-1", "user-a")
        promoted = mgr.release("thread-1")
        assert promoted is None
        assert mgr.active_count == 0

    def test_acquire_multiple(self):
        mgr = ConcurrencyManager(max_active=3, max_threads_per_day=10)
        for i in range(3):
            acquired, pos = mgr.try_acquire(f"thread-{i}", f"user-{i}")
            assert acquired is True
            assert pos == 0
        assert mgr.active_count == 3


class TestConcurrencyManagerQueue:
    """Queue behavior when max_active is reached."""

    def test_queued_when_full(self):
        mgr = ConcurrencyManager(max_active=2, max_threads_per_day=10)
        mgr.try_acquire("thread-1", "user-a")
        mgr.try_acquire("thread-2", "user-b")

        acquired, pos = mgr.try_acquire("thread-3", "user-c")
        assert acquired is False
        assert pos == 1
        assert mgr.queue_length == 1

    def test_queue_position_increments(self):
        mgr = ConcurrencyManager(max_active=1, max_threads_per_day=10)
        mgr.try_acquire("thread-1", "user-a")

        _, pos1 = mgr.try_acquire("thread-2", "user-b")
        _, pos2 = mgr.try_acquire("thread-3", "user-c")
        assert pos1 == 1
        assert pos2 == 2
        assert mgr.queue_length == 2

    def test_release_promotes_from_queue(self):
        mgr = ConcurrencyManager(max_active=1, max_threads_per_day=10)
        mgr.try_acquire("thread-1", "user-a")
        mgr.try_acquire("thread-2", "user-b")

        promoted = mgr.release("thread-1")
        assert promoted == "thread-2"
        assert mgr.active_count == 1
        assert mgr.queue_length == 0

    def test_release_promotes_fifo_order(self):
        mgr = ConcurrencyManager(max_active=1, max_threads_per_day=10)
        mgr.try_acquire("thread-1", "user-a")
        mgr.try_acquire("thread-2", "user-b")
        mgr.try_acquire("thread-3", "user-c")

        promoted = mgr.release("thread-1")
        assert promoted == "thread-2"

        promoted = mgr.release("thread-2")
        assert promoted == "thread-3"

    def test_release_from_queue_early_termination(self):
        """Releasing a thread that's in the queue (not active) should clean it."""
        mgr = ConcurrencyManager(max_active=1, max_threads_per_day=10)
        mgr.try_acquire("thread-1", "user-a")
        mgr.try_acquire("thread-2", "user-b")
        mgr.try_acquire("thread-3", "user-c")

        # Release thread-2 which is in the queue
        promoted = mgr.release("thread-2")
        assert promoted is None
        assert mgr.queue_length == 1  # Only thread-3 remains

        # Now release thread-1, thread-3 should be promoted
        promoted = mgr.release("thread-1")
        assert promoted == "thread-3"


class TestConcurrencyManagerDailyLimit:
    """Daily limit enforcement."""

    def test_daily_limit_allows_under_limit(self):
        mgr = ConcurrencyManager(max_active=50, max_threads_per_day=3)
        allowed, reason = mgr.check_daily_limit("user-a")
        assert allowed is True
        assert reason == ""

    def test_daily_limit_blocks_at_limit(self):
        mgr = ConcurrencyManager(max_active=50, max_threads_per_day=2)
        mgr.try_acquire("thread-1", "user-a")
        mgr.try_acquire("thread-2", "user-a")

        allowed, reason = mgr.check_daily_limit("user-a")
        assert allowed is False
        assert "Daily limit reached" in reason

    def test_daily_limit_blocks_acquire(self):
        mgr = ConcurrencyManager(max_active=50, max_threads_per_day=2)
        mgr.try_acquire("thread-1", "user-a")
        mgr.try_acquire("thread-2", "user-a")

        acquired, pos = mgr.try_acquire("thread-3", "user-a")
        assert acquired is False
        assert pos == 0  # Not queued, just denied

    def test_daily_limit_per_user(self):
        """Different users have independent limits."""
        mgr = ConcurrencyManager(max_active=50, max_threads_per_day=1)
        mgr.try_acquire("thread-1", "user-a")

        # user-a is at limit
        allowed, _ = mgr.check_daily_limit("user-a")
        assert allowed is False

        # user-b is fine
        allowed, _ = mgr.check_daily_limit("user-b")
        assert allowed is True

    def test_daily_limit_prunes_old_timestamps(self):
        """Timestamps older than 24h should not count."""
        mgr = ConcurrencyManager(max_active=50, max_threads_per_day=2)

        # Manually inject old timestamps
        old_time = time.time() - 90000  # 25 hours ago
        mgr._daily_usage["user-a"] = [old_time, old_time]

        allowed, reason = mgr.check_daily_limit("user-a")
        assert allowed is True
        assert reason == ""

    def test_active_count_and_queue_length_properties(self):
        mgr = ConcurrencyManager(max_active=2, max_threads_per_day=10)
        mgr.try_acquire("t1", "u1")
        assert mgr.active_count == 1
        assert mgr.queue_length == 0

        mgr.try_acquire("t2", "u2")
        assert mgr.active_count == 2

        mgr.try_acquire("t3", "u3")
        assert mgr.active_count == 2
        assert mgr.queue_length == 1
