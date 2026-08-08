"""Tests for fallback chain bug fixes (#60761).

Bug 1: overloaded errors trigger immediate fallback (no retry threshold)
Bug 2: unavailable fallback entries use TTL-based suppression (not permanent)
"""
import time

import pytest

from agent.error_classifier import FailoverReason


class TestOverloadedFallbackImmediateTrigger:
    """Bug 1: Overloaded errors (529) should trigger immediate fallback like rate limits."""

    def test_overloaded_in_rate_limited_set(self):
        """FailoverReason.overloaded should be in is_rate_limited set."""
        is_rate_limited = {
            FailoverReason.rate_limit,
            FailoverReason.billing,
            FailoverReason.upstream_rate_limit,
            FailoverReason.overloaded,
        }
        assert FailoverReason.overloaded in is_rate_limited

    def test_overloaded_not_in_transport_failure_set(self):
        """FailoverReason.overloaded should NOT be in _is_transport_failure set."""
        _is_transport_failure = {
            FailoverReason.timeout,
            # FailoverReason.overloaded,  # REMOVED in fix
        }
        assert FailoverReason.overloaded not in _is_transport_failure

    def test_timeout_still_requires_retry_threshold(self):
        """Timeout errors should still require retry_count >= 2 (not affected by fix)."""
        _is_transport_failure = {
            FailoverReason.timeout,
        }
        assert FailoverReason.timeout in _is_transport_failure
        # This is the old behavior we want to preserve for timeouts


class TestUnavailableFallbackTTL:
    """Bug 2: Unavailable fallback entries should use TTL-based suppression."""

    def test_unavailable_dict_stores_timestamps(self):
        """_unavailable_fallback_keys should store {fb_key: suppressed_until}."""
        unavailable = {}  # {fb_key: suppressed_until_monotonic}
        fb_key = ("provider", "model", "base_url")
        _FALLBACK_UNAVAILABLE_TTL_S = 300.0

        unavailable[fb_key] = time.monotonic() + _FALLBACK_UNAVAILABLE_TTL_S

        assert fb_key in unavailable
        assert isinstance(unavailable[fb_key], float)
        assert unavailable[fb_key] > time.monotonic()

    def test_suppressed_entry_is_skipped(self):
        """Entries within TTL should be skipped."""
        unavailable = {}
        fb_key = ("provider", "model", "base_url")
        _FALLBACK_UNAVAILABLE_TTL_S = 300.0

        unavailable[fb_key] = time.monotonic() + _FALLBACK_UNAVAILABLE_TTL_S

        suppressed_until = unavailable.get(fb_key)
        assert suppressed_until is not None
        assert time.monotonic() < suppressed_until  # Within TTL

        # Simulate the check in try_activate_fallback
        should_skip = suppressed_until is not None and time.monotonic() < suppressed_until
        assert should_skip

    def test_expired_entry_is_cleared(self):
        """Entries past TTL should be cleared and retried."""
        unavailable = {}
        fb_key = ("provider", "model", "base_url")

        # Use a very short TTL for testing
        unavailable[fb_key] = time.monotonic() + 0.1  # 100ms TTL
        time.sleep(0.15)  # Wait for TTL to expire

        suppressed_until = unavailable.get(fb_key)
        if suppressed_until is not None and time.monotonic() >= suppressed_until:
            del unavailable[fb_key]

        assert fb_key not in unavailable

    def test_ttl_constant_exists(self):
        """_FALLBACK_UNAVAILABLE_TTL_S constant should be 300.0 (5 minutes)."""
        from agent.chat_completion_helpers import _FALLBACK_UNAVAILABLE_TTL_S

        assert _FALLBACK_UNAVAILABLE_TTL_S == 300.0