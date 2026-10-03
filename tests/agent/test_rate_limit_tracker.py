"""Tests for agent.rate_limit_tracker — header parsing and formatting."""

import time
from datetime import datetime, timedelta, timezone

import pytest
from agent.rate_limit_tracker import (
    RateLimitBucket,
    format_rate_limit_compact,
    format_rate_limit_display,
    parse_rate_limit_headers,
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

class TestBucket:

    def test_usage_pct(self):
        b = RateLimitBucket(limit=100, remaining=20, reset_seconds=30.0, captured_at=time.time())
        assert b.usage_pct == pytest.approx(80.0)

    def test_remaining_seconds_now(self):
        now = time.time()
        b = RateLimitBucket(limit=800, remaining=795, reset_seconds=60.0, captured_at=now - 10)
        # ~50 seconds should remain
        assert 49 <= b.remaining_seconds_now <= 51


# ── Anthropic unified (subscription) headers ────────────────────────────

# The 5-hour window is anchored to the session's first request, so its reset lands on an
# arbitrary instant — deliberately off every clock grid here (odd minutes AND odd seconds).
_5H_OFFSET = 5 * 3600 - 1543
_7D_OFFSET = 7 * 86400 - 907


def _unified_headers(skew: int = 0):
    """Unified headers as Anthropic sends them: ISO-8601 for one window, unix epoch for the
    other, mixed header casing, plus a family member that is not a window. ``skew`` shifts the
    5-hour anchor, standing in for a session that started at a different moment."""
    now = datetime.now(timezone.utc)
    five_h = now + timedelta(seconds=_5H_OFFSET + skew)
    seven_d = now + timedelta(seconds=_7D_OFFSET)
    return {
        "anthropic-ratelimit-unified-status": "allowed_warning",
        "anthropic-ratelimit-unified-5h-limit": "7000",
        "Anthropic-RateLimit-Unified-5h-Remaining": "1234",
        "anthropic-ratelimit-unified-5h-reset": five_h.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "anthropic-ratelimit-unified-5h-status": "allowed",
        "anthropic-ratelimit-unified-7d-limit": "90000",
        "anthropic-ratelimit-unified-7d-remaining": "40000",
        "anthropic-ratelimit-unified-7d-reset": str(int(seven_d.timestamp())),
        "anthropic-ratelimit-unified-fallback-percentage": "0.5",
    }


class TestUnifiedHeaders:
    def test_unified_only_response_is_captured_with_an_absolute_reset(self):
        """A subscription response carries no x-ratelimit-* at all; it must still parse, and the
        reset must survive as the exact instant sent rather than being floored into a bucket."""
        headers = _unified_headers()
        state = parse_rate_limit_headers(headers, provider="anthropic")
        assert state is not None and state.has_data

        five_h = state.unified["5h"]
        assert (five_h.limit, five_h.remaining) == (7000, 1234)
        assert five_h.status == "allowed"
        assert state.unified["overall"].status == "allowed_warning"

        # Stored as the exact instant the header named, not derived from a window width.
        sent = datetime.strptime(
            headers["anthropic-ratelimit-unified-5h-reset"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc).timestamp()
        assert five_h.reset_at == pytest.approx(sent, abs=1)
        assert five_h.remaining_seconds_now == pytest.approx(_5H_OFFSET, abs=5)

        # Unix-epoch resets parse to the same absolute scale as the ISO-8601 ones.
        assert state.unified["7d"].remaining_seconds_now == pytest.approx(_7D_OFFSET, abs=5)
        # ``-fallback-percentage`` has no window/field shape and is not mistaken for one.
        assert "fallback" not in state.unified

        # Parsed state reaches both /usage renderers.
        assert "5h" in format_rate_limit_compact(state)
        assert "allowed_warning" in format_rate_limit_display(state)

    def test_reset_instants_are_not_snapped_to_a_fixed_window_grid(self):
        """The 5-hour window is anchored to a session's first request, so two sessions whose
        resets differ by 137s must stay 137s apart — flooring onto a 4h/5h grid would collapse
        them onto the same instant."""
        skew = 137
        early = parse_rate_limit_headers(_unified_headers(), provider="anthropic")
        late = parse_rate_limit_headers(_unified_headers(skew=skew), provider="anthropic")
        assert late.unified["5h"].reset_at - early.unified["5h"].reset_at == pytest.approx(skew, abs=1)

    def test_classic_and_unified_families_coexist_without_changing_classic_semantics(self):
        """x-ratelimit-* resets stay seconds-from-capture; unified resets stay absolute."""
        classic_only = parse_rate_limit_headers(NOUS_HEADERS, provider="nous")
        assert classic_only.unified == {}
        assert classic_only.requests_min.reset_at == 0.0
        assert classic_only.requests_min.remaining_seconds_now == pytest.approx(45.5, abs=2)

        both = parse_rate_limit_headers({**NOUS_HEADERS, **_unified_headers()}, provider="anthropic")
        assert both.requests_min.limit == classic_only.requests_min.limit
        assert both.requests_min.reset_seconds == classic_only.requests_min.reset_seconds
        assert set(both.unified) == {"overall", "5h", "7d"}
