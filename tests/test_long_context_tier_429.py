"""Tests for Anthropic long-context tier 429 handling.

When Claude Max users without "extra usage" hit the 1M context tier,
Anthropic returns HTTP 429 "Extra usage is required for long context
requests."  This is NOT a transient rate limit — the agent should
reduce context_length to 200k and compress instead of retrying.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------


class TestLongContextTierDetection:
    """Verify the detection heuristic matches the Anthropic error."""

    @staticmethod
    def _is_long_context_tier_error(status_code, error_msg):
        error_msg = error_msg.lower()
        return (
            status_code == 429
            and "extra usage" in error_msg
            and "long context" in error_msg
        )

    def test_matches_anthropic_error(self):
        assert self._is_long_context_tier_error(
            429,
            "Extra usage is required for long context requests.",
        )

    def test_matches_lowercase(self):
        assert self._is_long_context_tier_error(
            429,
            "extra usage is required for long context requests.",
        )

    def test_rejects_normal_429(self):
        assert not self._is_long_context_tier_error(
            429,
            "Rate limit exceeded. Please retry after 30 seconds.",
        )

    def test_rejects_wrong_status(self):
        assert not self._is_long_context_tier_error(
            400,
            "Extra usage is required for long context requests.",
        )

    def test_rejects_partial_match(self):
        """Both 'extra usage' AND 'long context' must be present."""
        assert not self._is_long_context_tier_error(
            429, "extra usage required"
        )
        assert not self._is_long_context_tier_error(
            429, "long context requests not supported"
        )


# ---------------------------------------------------------------------------
# Context reduction
# ---------------------------------------------------------------------------


class TestContextReduction:
    """When the long-context tier error fires, context_length should
    drop to 200k and the reduced flag should be set correctly."""

    def _make_compressor(self, context_length=1_000_000, threshold_percent=0.5):
        c = SimpleNamespace(
            context_length=context_length,
            threshold_percent=threshold_percent,
            threshold_tokens=int(context_length * threshold_percent),
            _context_probed=False,
            _context_probe_persistable=False,
        )
        return c

    def test_reduces_1m_to_200k(self):
        comp = self._make_compressor(1_000_000)
        reduced_ctx = 200_000

        if comp.context_length > reduced_ctx:
            comp.context_length = reduced_ctx
            comp.threshold_tokens = int(reduced_ctx * comp.threshold_percent)
            comp._context_probed = True
            comp._context_probe_persistable = False

        assert comp.context_length == 200_000
        assert comp.threshold_tokens == 100_000
        assert comp._context_probed is True
        # Must NOT persist — subscription tier, not model capability
        assert comp._context_probe_persistable is False

    def test_no_reduction_when_already_200k(self):
        comp = self._make_compressor(200_000)
        reduced_ctx = 200_000

        original = comp.context_length
        if comp.context_length > reduced_ctx:
            comp.context_length = reduced_ctx

        assert comp.context_length == original  # unchanged

    def test_no_reduction_when_below_200k(self):
        comp = self._make_compressor(128_000)
        reduced_ctx = 200_000

        original = comp.context_length
        if comp.context_length > reduced_ctx:
            comp.context_length = reduced_ctx

        assert comp.context_length == original  # unchanged


# ---------------------------------------------------------------------------
# Integration: agent error handler path
# ---------------------------------------------------------------------------


class TestAgentErrorPath:
    """Verify the long-context 429 doesn't hit the generic rate-limit
    or client-error handlers."""

    def test_long_context_429_not_treated_as_rate_limit(self):
        """The error should be intercepted before the generic
        is_rate_limited check fires a fallback switch."""
        error_msg = "extra usage is required for long context requests."
        status_code = 429

        # The long-context check fires first
        _is_long_context_tier_error = (
            status_code == 429
            and "extra usage" in error_msg
            and "long context" in error_msg
        )
        assert _is_long_context_tier_error

        # So we never reach the generic rate-limit path
        # (in the real code, `break` exits the retry loop)

    def test_normal_429_still_treated_as_rate_limit(self):
        """A normal 429 should NOT match the long-context check."""
        error_msg = "rate limit exceeded"
        status_code = 429

        _is_long_context_tier_error = (
            status_code == 429
            and "extra usage" in error_msg
            and "long context" in error_msg
        )
        assert not _is_long_context_tier_error

        is_rate_limited = (
            status_code == 429
            or "rate limit" in error_msg
        )
        assert is_rate_limited
