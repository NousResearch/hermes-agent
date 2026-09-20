"""Tests for agent.rate_limit_tracker — header parsing and formatting."""

import time
import pytest
from agent.rate_limit_tracker import (
    RateLimitBucket,
    RateLimitState,
    parse_rate_limit_headers,
    format_rate_limit_display,
    format_rate_limit_compact,
    _fmt_count,
    _fmt_seconds,
    _bar,
)


# ── Sample headers from Nous inference API ──────────────────────────────

NOUS_HEADERS = {
    "x-ratelimit-limit-requests": "800",
    "x-ratelimit-limit-requests-1h": "33600",
    "x-ratelimit-limit-tokens": "8000000",
    "x-ratelimit-limit-tokens-1h": "336000000",
    "x-ratelimit-remaining-requests": "795",
    "x-ratelimit-remaining-requests-1h": "33590",
    "x-ratelimit-remaining-tokens": "7999500",
    "x-ratelimit-remaining-tokens-1h": "335999000",
    "x-ratelimit-reset-requests": "45.5",
    "x-ratelimit-reset-requests-1h": "3500.0",
    "x-ratelimit-reset-tokens": "42.3",
    "x-ratelimit-reset-tokens-1h": "3490.0",
}


class TestParseHeaders:
    def test_basic_parsing(self):
        state = parse_rate_limit_headers(NOUS_HEADERS, provider="nous")
        assert state is not None
        assert state.provider == "nous"
        assert state.has_data

        assert state.requests_min.limit == 800
        assert state.requests_min.remaining == 795
        assert state.requests_min.reset_seconds == 45.5

        assert state.requests_hour.limit == 33600
        assert state.requests_hour.remaining == 33590

        assert state.tokens_min.limit == 8000000
        assert state.tokens_min.remaining == 7999500

        assert state.tokens_hour.limit == 336000000
        assert state.tokens_hour.remaining == 335999000
        assert state.tokens_hour.reset_seconds == 3490.0

    def test_no_headers(self):
        state = parse_rate_limit_headers({})
        assert state is None

    def test_codex_headers_parse_plan_windows_and_credits(self):
        from agent.rate_limit_tracker import parse_codex_headers

        state = parse_codex_headers({
            "X-Codex-Active-Limit": "premium",
            "X-Codex-Plan-Type": "plus",
            "X-Codex-Credits-Balance": "unlimited",
            "X-Codex-Credits-Has-Credits": "true",
            "X-Codex-Credits-Unlimited": "true",
            "X-Codex-Primary-Used-Percent": "40",
            "X-Codex-Primary-Window-Minutes": "300",
            "X-Codex-Primary-Reset-After-Seconds": "6000",
            "X-Codex-Secondary-Used-Percent": "21",
            "X-Codex-Secondary-Window-Minutes": "10080",
            "X-Codex-Secondary-Reset-After-Seconds": "500000",
        })

        assert state is not None
        assert state.plan_type == "plus"
        assert state.active_limit == "premium"
        assert state.credits_balance == "unlimited"
        assert state.credits_has_credits is True
        assert state.credits_unlimited is True
        assert state.primary.used_percent == pytest.approx(40)
        assert state.primary.window_minutes == 300
        assert state.primary.reset_after_seconds == 6000
        assert state.secondary.used_percent == pytest.approx(21)
        assert state.secondary.window_minutes == 10080

    def test_codex_headers_absent(self):
        from agent.rate_limit_tracker import parse_codex_headers

        assert parse_codex_headers({}) is None

    def test_codex_display(self):
        from agent.rate_limit_tracker import format_codex_rate_limit_display, parse_codex_headers

        state = parse_codex_headers({
            "x-codex-plan-type": "plus",
            "x-codex-active-limit": "premium",
            "x-codex-primary-used-percent": "40",
            "x-codex-primary-reset-after-seconds": "6000",
            "x-codex-secondary-used-percent": "21",
            "x-codex-secondary-reset-after-seconds": "500000",
        })
        result = format_codex_rate_limit_display(state)
        assert "Plan: plus (limit: premium)" in result
        assert "5h window" in result and "7d window" in result
        assert "40.0% used" in result and "resets in 1h 40m" in result





class TestBucket:

    def test_usage_pct(self):
        b = RateLimitBucket(limit=100, remaining=20, reset_seconds=30.0, captured_at=time.time())
        assert b.usage_pct == pytest.approx(80.0)


    def test_remaining_seconds_now(self):
        now = time.time()
        b = RateLimitBucket(limit=800, remaining=795, reset_seconds=60.0, captured_at=now - 10)
        # ~50 seconds should remain
        assert 49 <= b.remaining_seconds_now <= 51



class TestFormatting:



    def test_fmt_seconds_short(self):
        assert _fmt_seconds(45) == "45s"
        assert _fmt_seconds(0) == "0s"



    def test_bar(self):
        bar = _bar(50.0, width=10)
        assert bar == "[█████░░░░░]"
        assert _bar(0.0, width=10) == "[░░░░░░░░░░]"
        assert _bar(100.0, width=10) == "[██████████]"




    def test_format_compact(self):
        state = parse_rate_limit_headers(NOUS_HEADERS, provider="nous")
        result = format_rate_limit_compact(state)
        assert "RPM:" in result
        assert "RPH:" in result
        assert "TPM:" in result
        assert "TPH:" in result
        assert "resets" in result



class TestAgentIntegration:
    """Test that AIAgent captures rate limit state correctly."""

    def test_capture_rate_limits_from_headers(self):
        """Simulate the header capture path without a real API call."""
        # Use a mock httpx-like response
        class MockResponse:
            headers = NOUS_HEADERS

        # Import AIAgent minimally

        # Test the parsing directly
        state = parse_rate_limit_headers(MockResponse.headers, provider="nous")
        assert state is not None
        assert state.requests_min.limit == 800
        assert state.tokens_hour.limit == 336000000

    def test_capture_rate_limits_none_response(self):
        """_capture_rate_limits should handle None gracefully."""
        from agent.rate_limit_tracker import parse_rate_limit_headers
        # None should not crash
        result = parse_rate_limit_headers({})
        assert result is None
