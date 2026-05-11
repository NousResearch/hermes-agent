"""Tests for gateway.daimon.thread_filter module."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from gateway.daimon.config import DaimonConfig
from gateway.daimon.thread_filter import ThreadOwnershipTracker


def _cfg(admin_users: list[str] | None = None) -> DaimonConfig:
    """Create a DaimonConfig with optional admin users."""
    return DaimonConfig(admin_users=admin_users or [])


class TestThreadOwnershipTracker:
    """Test suite for ThreadOwnershipTracker."""

    def test_creator_allowed(self) -> None:
        """Register thread with creator, creator's messages pass."""
        tracker = ThreadOwnershipTracker()
        tracker.register("thread-1", "user-A")

        cfg = _cfg()
        assert tracker.should_process("user-A", "thread-1", cfg) is True

    def test_admin_always_allowed(self) -> None:
        """Admin can post in anyone's thread."""
        tracker = ThreadOwnershipTracker()
        tracker.register("thread-1", "user-A")

        cfg = _cfg(admin_users=["admin-1"])
        assert tracker.should_process("admin-1", "thread-1", cfg) is True

    def test_other_user_blocked(self) -> None:
        """Non-creator non-admin is rejected."""
        tracker = ThreadOwnershipTracker()
        tracker.register("thread-1", "user-A")

        cfg = _cfg()
        assert tracker.should_process("user-B", "thread-1", cfg) is False

    def test_unknown_thread_allowed(self) -> None:
        """Unregistered threads pass through (backward compat)."""
        tracker = ThreadOwnershipTracker()

        cfg = _cfg()
        assert tracker.should_process("user-X", "unknown-thread", cfg) is True

    def test_unregister_removes_tracking(self) -> None:
        """After unregister, thread is unknown again and allows anyone."""
        tracker = ThreadOwnershipTracker()
        tracker.register("thread-1", "user-A")

        cfg = _cfg()
        # Before unregister: user-B is blocked
        assert tracker.should_process("user-B", "thread-1", cfg) is False

        tracker.unregister("thread-1")
        # After unregister: thread is unknown, so user-B is allowed
        assert tracker.should_process("user-B", "thread-1", cfg) is True

    def test_tracked_count(self) -> None:
        """Verify count property tracks registered threads."""
        tracker = ThreadOwnershipTracker()
        assert tracker.tracked_count == 0

        tracker.register("thread-1", "user-A")
        assert tracker.tracked_count == 1

        tracker.register("thread-2", "user-B")
        assert tracker.tracked_count == 2

        tracker.unregister("thread-1")
        assert tracker.tracked_count == 1

        # Unregistering non-existent thread is a no-op
        tracker.unregister("thread-999")
        assert tracker.tracked_count == 1

    def test_thread_safe(self) -> None:
        """Concurrent register/should_process doesn't crash."""
        tracker = ThreadOwnershipTracker()
        cfg = _cfg(admin_users=["admin-1"])

        num_threads = 50

        def register_and_check(i: int) -> bool:
            thread_id = f"thread-{i}"
            creator_id = f"user-{i}"
            tracker.register(thread_id, creator_id)
            # Creator should always be allowed
            r1 = tracker.should_process(creator_id, thread_id, cfg)
            # Admin should always be allowed
            r2 = tracker.should_process("admin-1", thread_id, cfg)
            # Other user should be blocked
            other = f"user-{i + num_threads}"
            r3 = tracker.should_process(other, thread_id, cfg)
            return r1 and r2 and not r3

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register_and_check, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        assert all(results)
        assert tracker.tracked_count == num_threads

    def test_get_owner(self) -> None:
        """get_owner returns the creator or None."""
        tracker = ThreadOwnershipTracker()
        assert tracker.get_owner("thread-1") is None

        tracker.register("thread-1", "user-A")
        assert tracker.get_owner("thread-1") == "user-A"

    def test_register_overwrites(self) -> None:
        """Registering the same thread again overwrites the owner."""
        tracker = ThreadOwnershipTracker()
        tracker.register("thread-1", "user-A")
        tracker.register("thread-1", "user-B")

        assert tracker.get_owner("thread-1") == "user-B"
        assert tracker.tracked_count == 1
