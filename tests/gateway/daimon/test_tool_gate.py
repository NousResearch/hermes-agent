# tests/gateway/daimon/test_tool_gate.py
"""Tests for session-scoped tool gating."""
from __future__ import annotations

import threading
import time

import pytest

from gateway.daimon.tool_gate import (
    active_session_count,
    check_tool_call,
    get_limiter,
    register_limiter,
    unregister_limiter,
    _session_limiters,
)
from gateway.daimon.tool_limiter import ToolLimiter


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure a clean limiter registry for each test."""
    _session_limiters.clear()
    yield
    _session_limiters.clear()


class TestNoLimiterAllowsAll:
    """When no limiter is registered for a session, all calls are allowed."""

    def test_unregistered_session_returns_none(self):
        result = check_tool_call("unknown-session", "terminal")
        assert result is None

    def test_get_limiter_returns_none_for_unknown(self):
        assert get_limiter("nonexistent") is None


class TestRegisteredLimiterEnforces:
    """A registered limiter blocks calls at the limit."""

    def test_blocks_after_limit(self):
        limiter = ToolLimiter({"terminal": 2, "read_file": -1})
        register_limiter("sess-1", limiter)

        # First two calls should be allowed
        assert check_tool_call("sess-1", "terminal") is None
        assert check_tool_call("sess-1", "terminal") is None

        # Third call should be blocked
        result = check_tool_call("sess-1", "terminal")
        assert result is not None
        assert "terminal" in result
        assert "2/2" in result

    def test_blocks_disabled_tool(self):
        limiter = ToolLimiter({"terminal": -1, "browser": 0})
        register_limiter("sess-2", limiter)

        result = check_tool_call("sess-2", "browser_navigate")
        assert result is not None
        assert "disabled" in result

    def test_blocks_unknown_tool(self):
        limiter = ToolLimiter({"terminal": 5})
        register_limiter("sess-3", limiter)

        result = check_tool_call("sess-3", "dangerous_tool")
        assert result is not None
        assert "not permitted" in result


class TestRecordsOnAllow:
    """After check_tool_call allows a call, the count increases."""

    def test_count_increments(self):
        limiter = ToolLimiter({"terminal": 5})
        register_limiter("sess-rec", limiter)

        assert limiter.remaining("terminal") == 5
        check_tool_call("sess-rec", "terminal")
        assert limiter.remaining("terminal") == 4
        check_tool_call("sess-rec", "terminal")
        assert limiter.remaining("terminal") == 3

    def test_no_record_on_deny(self):
        limiter = ToolLimiter({"terminal": 1})
        register_limiter("sess-deny", limiter)

        # Use up the limit
        check_tool_call("sess-deny", "terminal")
        assert limiter.remaining("terminal") == 0

        # Denied call should NOT increment further
        check_tool_call("sess-deny", "terminal")
        # Count should still be 1 (the limit), not 2
        assert limiter._counts["terminal"] == 1


class TestUnregisterRemoves:
    """After unregistering, all calls for that session are allowed again."""

    def test_unregister_allows_all(self):
        limiter = ToolLimiter({"terminal": 1})
        register_limiter("sess-unreg", limiter)

        # Use the limit
        check_tool_call("sess-unreg", "terminal")
        assert check_tool_call("sess-unreg", "terminal") is not None  # blocked

        # Unregister
        unregister_limiter("sess-unreg")

        # Now all calls should be allowed (no limiter)
        assert check_tool_call("sess-unreg", "terminal") is None

    def test_unregister_nonexistent_is_safe(self):
        # Should not raise
        unregister_limiter("never-existed")


class TestActiveSessionCount:
    """Tracks the number of registered sessions."""

    def test_starts_at_zero(self):
        assert active_session_count() == 0

    def test_increments_on_register(self):
        register_limiter("s1", ToolLimiter({"terminal": -1}))
        assert active_session_count() == 1
        register_limiter("s2", ToolLimiter({"terminal": -1}))
        assert active_session_count() == 2

    def test_decrements_on_unregister(self):
        register_limiter("s1", ToolLimiter({"terminal": -1}))
        register_limiter("s2", ToolLimiter({"terminal": -1}))
        unregister_limiter("s1")
        assert active_session_count() == 1

    def test_re_register_same_id_does_not_double_count(self):
        register_limiter("s1", ToolLimiter({"terminal": -1}))
        register_limiter("s1", ToolLimiter({"terminal": 5}))
        assert active_session_count() == 1


class TestConcurrentSessionsIsolated:
    """Two sessions with different limiters don't interfere."""

    def test_independent_limits(self):
        limiter_a = ToolLimiter({"terminal": 1, "read_file": -1})
        limiter_b = ToolLimiter({"terminal": 10, "read_file": 2})
        register_limiter("sess-a", limiter_a)
        register_limiter("sess-b", limiter_b)

        # Use up session A's terminal limit
        check_tool_call("sess-a", "terminal")
        assert check_tool_call("sess-a", "terminal") is not None  # blocked

        # Session B still has plenty
        assert check_tool_call("sess-b", "terminal") is None
        assert limiter_b.remaining("terminal") == 9

    def test_threaded_concurrent_access(self):
        """Multiple threads registering/checking simultaneously."""
        results = {}
        barrier = threading.Barrier(4)

        def worker(session_id: str, limit: int):
            limiter = ToolLimiter({"terminal": limit})
            register_limiter(session_id, limiter)
            barrier.wait()
            # Each thread makes `limit` calls
            allowed = 0
            for _ in range(limit + 2):  # Try more than the limit
                if check_tool_call(session_id, "terminal") is None:
                    allowed += 1
            results[session_id] = allowed

        threads = [
            threading.Thread(target=worker, args=(f"t-{i}", i + 1))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each session should have allowed exactly its limit
        for i in range(4):
            assert results[f"t-{i}"] == i + 1

    def test_same_session_check_and_record_is_atomic(self):
        """Parallel calls for one limited session must not all pass the same count."""
        limiter = ToolLimiter({"terminal": 1})
        original_check = limiter.check

        def slow_check(tool_name: str) -> bool:
            allowed = original_check(tool_name)
            time.sleep(0.01)
            return allowed

        limiter.check = slow_check
        register_limiter("shared", limiter)

        barrier = threading.Barrier(8)
        results = []
        results_lock = threading.Lock()

        def worker():
            barrier.wait()
            allowed = check_tool_call("shared", "terminal") is None
            with results_lock:
                results.append(allowed)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results.count(True) == 1
        assert limiter._counts["terminal"] == 1
