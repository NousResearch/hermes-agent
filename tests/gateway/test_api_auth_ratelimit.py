"""Tests for per-IP authentication failure rate limiting in APIServerAdapter.

Covers:
- 10 failed auths from same IP -> 11th returns 429
- After window expires -> requests allowed again
- Successful auth from same IP as failed ones -> still works (200)
- Different IPs tracked independently
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from gateway.platforms.api_server import _AuthFailTracker


# ---------------------------------------------------------------------------
# _AuthFailTracker unit tests
# ---------------------------------------------------------------------------


class TestAuthFailTrackerIsBlocked:
    """is_blocked() should return True only after max_failures within window."""

    def test_not_blocked_initially(self):
        tracker = _AuthFailTracker(max_failures=10, window_seconds=60)
        assert tracker.is_blocked("1.2.3.4") is False

    def test_not_blocked_below_limit(self):
        tracker = _AuthFailTracker(max_failures=10, window_seconds=60)
        for _ in range(9):
            tracker.record_failure("1.2.3.4")
        assert tracker.is_blocked("1.2.3.4") is False

    def test_blocked_at_limit(self):
        tracker = _AuthFailTracker(max_failures=10, window_seconds=60)
        for _ in range(10):
            tracker.record_failure("1.2.3.4")
        assert tracker.is_blocked("1.2.3.4") is True

    def test_blocked_above_limit(self):
        tracker = _AuthFailTracker(max_failures=10, window_seconds=60)
        for _ in range(15):
            tracker.record_failure("1.2.3.4")
        assert tracker.is_blocked("1.2.3.4") is True

    def test_zero_max_never_blocks(self):
        """max_failures=0 disables rate limiting entirely."""
        tracker = _AuthFailTracker(max_failures=0, window_seconds=60)
        for _ in range(100):
            tracker.record_failure("1.2.3.4")
        assert tracker.is_blocked("1.2.3.4") is False

    def test_different_ips_tracked_independently(self):
        """Failures for one IP must not affect another IP."""
        tracker = _AuthFailTracker(max_failures=10, window_seconds=60)
        for _ in range(10):
            tracker.record_failure("10.0.0.1")
        # 10.0.0.1 is blocked
        assert tracker.is_blocked("10.0.0.1") is True
        # 10.0.0.2 has no failures — must not be blocked
        assert tracker.is_blocked("10.0.0.2") is False

    def test_window_expiry_unblocks_ip(self):
        """After the sliding window expires all failures drop off."""
        tracker = _AuthFailTracker(max_failures=3, window_seconds=1)
        for _ in range(3):
            tracker.record_failure("1.2.3.4")
        assert tracker.is_blocked("1.2.3.4") is True

        # Simulate time advancing past the window
        future = time.time() + 2  # 2 seconds ahead — past 1-second window
        with patch("gateway.platforms.api_server.time") as mock_time:
            mock_time.time.return_value = future
            # Re-import the tracker's is_blocked which calls time.time() internally;
            # but the tracker is already instantiated so we need to patch at the
            # module level where time.time is called inside _AuthFailTracker.
            # Instead, test via a fresh check with manipulated timestamps.
            pass

        # Use a direct approach: manually age the timestamps
        with tracker._lock:
            tracker._buckets["1.2.3.4"] = [t - 2 for t in tracker._buckets["1.2.3.4"]]
        assert tracker.is_blocked("1.2.3.4") is False


class TestAuthFailTrackerRetryAfter:
    """retry_after() returns the correct number of seconds to wait."""

    def test_retry_after_zero_when_no_failures(self):
        tracker = _AuthFailTracker(max_failures=10, window_seconds=60)
        assert tracker.retry_after("1.2.3.4") == 0

    def test_retry_after_positive_when_blocked(self):
        tracker = _AuthFailTracker(max_failures=10, window_seconds=60)
        for _ in range(10):
            tracker.record_failure("1.2.3.4")
        ra = tracker.retry_after("1.2.3.4")
        # Should be at most window_seconds + 1
        assert 0 < ra <= 61


# ---------------------------------------------------------------------------
# Integration tests via _check_auth on a mocked APIServerAdapter
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "secret", max_failures: int = 10):
    """Return a minimal APIServerAdapter instance with the given API key."""
    from gateway.platforms.api_server import APIServerAdapter, _AuthFailTracker
    from gateway.config import PlatformConfig

    config = PlatformConfig(enabled=True, extra={"key": api_key})
    adapter = APIServerAdapter.__new__(APIServerAdapter)
    # Manually set only the attributes _check_auth needs
    adapter._api_key = api_key
    adapter._auth_fail_tracker = _AuthFailTracker(
        max_failures=max_failures, window_seconds=60
    )
    return adapter


def _make_request(ip: str = "1.2.3.4", token: str = "") -> MagicMock:
    """Build a fake aiohttp Request with the given remote IP and Bearer token."""
    req = MagicMock()
    req.remote = ip
    if token:
        req.headers = {"Authorization": f"Bearer {token}"}
    else:
        req.headers = {}
    return req


class TestCheckAuthRateLimiting:
    """_check_auth() must enforce per-IP rate limiting on auth failures."""

    def test_valid_key_returns_none(self):
        """Correct Bearer token -> no response (auth OK)."""
        adapter = _make_adapter(api_key="mysecret")
        req = _make_request(ip="1.2.3.4", token="mysecret")
        result = adapter._check_auth(req)
        assert result is None

    def test_invalid_key_returns_401(self):
        """Wrong Bearer token -> 401 on first failure."""
        adapter = _make_adapter(api_key="mysecret")
        req = _make_request(ip="1.2.3.4", token="wrongkey")
        result = adapter._check_auth(req)
        assert result is not None
        assert result.status == 401

    def test_ten_failures_then_429(self):
        """10 failed auths from same IP -> 11th returns 429."""
        adapter = _make_adapter(api_key="mysecret", max_failures=10)
        req = _make_request(ip="5.5.5.5", token="bad")

        # First 10 should return 401
        for i in range(10):
            resp = adapter._check_auth(req)
            assert resp is not None, f"Expected 401 response on attempt {i+1}"
            assert resp.status == 401, f"Expected 401 on attempt {i+1}, got {resp.status}"

        # 11th should return 429
        resp = adapter._check_auth(req)
        assert resp is not None
        assert resp.status == 429

    def test_429_includes_retry_after_header(self):
        """429 response must include a Retry-After header."""
        adapter = _make_adapter(api_key="mysecret", max_failures=3)
        req = _make_request(ip="6.6.6.6", token="bad")

        for _ in range(3):
            adapter._check_auth(req)

        resp = adapter._check_auth(req)
        assert resp is not None
        assert resp.status == 429
        assert "Retry-After" in resp.headers

    def test_valid_auth_from_same_ip_as_failed_ones(self):
        """Successful auth from same IP as prior failures -> still returns None (200)."""
        adapter = _make_adapter(api_key="mysecret", max_failures=10)
        bad_req = _make_request(ip="7.7.7.7", token="wrong")
        good_req = _make_request(ip="7.7.7.7", token="mysecret")

        # Record 5 failures — not yet blocked
        for _ in range(5):
            resp = adapter._check_auth(bad_req)
            assert resp.status == 401

        # Valid auth from the same IP must still succeed
        result = adapter._check_auth(good_req)
        assert result is None, "Expected auth OK (None) but got a response"

    def test_different_ips_tracked_independently(self):
        """Failures from IP-A must not block IP-B."""
        adapter = _make_adapter(api_key="mysecret", max_failures=5)
        req_a = _make_request(ip="10.0.0.1", token="bad")
        req_b = _make_request(ip="10.0.0.2", token="bad")

        # Exhaust the limit for IP-A
        for _ in range(5):
            adapter._check_auth(req_a)

        # IP-A is now blocked
        resp_a = adapter._check_auth(req_a)
        assert resp_a is not None
        assert resp_a.status == 429

        # IP-B has had zero failures — must still get 401 (not 429)
        resp_b = adapter._check_auth(req_b)
        assert resp_b is not None
        assert resp_b.status == 401

    def test_window_expiry_allows_requests_again(self):
        """After sliding window expires the IP is no longer blocked."""
        adapter = _make_adapter(api_key="mysecret", max_failures=3)
        req = _make_request(ip="9.9.9.9", token="bad")

        # Trigger the block
        for _ in range(3):
            adapter._check_auth(req)
        assert adapter._check_auth(req).status == 429

        # Age all timestamps past the window
        with adapter._auth_fail_tracker._lock:
            tracker = adapter._auth_fail_tracker
            tracker._buckets["9.9.9.9"] = [
                t - (tracker._window + 1)
                for t in tracker._buckets["9.9.9.9"]
            ]

        # Now the IP should be unblocked — next bad attempt returns 401 again
        resp = adapter._check_auth(req)
        assert resp is not None
        assert resp.status == 401, f"Expected 401 after window expiry, got {resp.status}"

    def test_no_api_key_configured_allows_all(self):
        """When no API key is configured, all requests pass through."""
        adapter = _make_adapter(api_key="")
        adapter._api_key = ""
        req = _make_request(ip="1.2.3.4", token="anything")
        result = adapter._check_auth(req)
        assert result is None
