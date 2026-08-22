"""Tests for agent.rate_limit_tracker — header parsing and formatting."""

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from agent.chat_completion_helpers import (
    _dispatch_nonstreaming_api_request,
    interruptible_api_call,
    interruptible_streaming_api_call,
)
from hermes_cli.config import get_custom_provider_rpm_throttle_threshold
from agent.rate_limit_tracker import (
    RateLimitBucket,
    RateLimitState,
    parse_rate_limit_headers,
    format_rate_limit_display,
    format_rate_limit_compact,
    _fmt_count,
    _fmt_seconds,
    _bar,
    wait_for_low_rpm,
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

    def test_documented_reset_formats_preserve_route_and_remaining(self):
        state = parse_rate_limit_headers(
            {
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "2",
                "x-ratelimit-reset-requests": "1m30s",
            },
            provider="openai",
            base_url="https://api.openai.com/v1/",
        )
        assert state is not None
        assert state.requests_min.reset_seconds == 90.0
        assert state.requests_min.has_remaining is True
        assert state.base_url == "https://api.openai.com/v1"

        reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        anthropic = parse_rate_limit_headers(
            {
                "anthropic-ratelimit-requests-limit": "50",
                "anthropic-ratelimit-requests-remaining": "1",
                "anthropic-ratelimit-requests-reset": reset_at.isoformat(),
            },
            provider="anthropic",
        )
        assert anthropic is not None
        assert anthropic.requests_min.remaining == 1
        assert 28.0 <= anthropic.requests_min.reset_seconds <= 30.0





class TestBucket:

    def test_usage_pct(self):
        b = RateLimitBucket(limit=100, remaining=20, reset_seconds=30.0, captured_at=time.time())
        assert b.usage_pct == pytest.approx(80.0)


    def test_remaining_seconds_now(self):
        now = time.time()
        b = RateLimitBucket(limit=800, remaining=795, reset_seconds=60.0, captured_at=now - 10)
        # ~50 seconds should remain
        assert 49 <= b.remaining_seconds_now <= 51


class TestPreemptivePacing:
    def test_waits_only_for_the_matching_low_rpm_route(self):
        clock = [0.0]
        sleeps = []
        state = SimpleNamespace(
            has_data=True,
            provider="openai",
            base_url="https://api.openai.com/v1",
            requests_min=SimpleNamespace(
                has_remaining=True,
                limit=100,
                remaining=2,
                remaining_seconds_now=0.5,
            ),
        )

        waited = wait_for_low_rpm(
            state,
            provider="openai",
            base_url="https://api.openai.com/v1",
            sleep_fn=lambda seconds: (
                sleeps.append(seconds), clock.__setitem__(0, clock[0] + seconds)
            ),
            monotonic_fn=lambda: clock[0],
        )
        assert waited == 0.5
        assert sleeps == [0.25, 0.25]
        assert wait_for_low_rpm(
            state,
            provider="openai",
            base_url="https://other.example/v1",
            sleep_fn=MagicMock(),
        ) == 0.0
        assert wait_for_low_rpm(
            state,
            provider="openrouter",
            base_url="https://api.openai.com/v1",
            sleep_fn=MagicMock(),
        ) == 0.0

    def test_custom_provider_threshold_and_api_entries_are_wired(self):
        config = {
            "providers": {
                "gateway": {
                    "api": "https://gateway.example/v1",
                    "rpm_throttle_threshold": 4,
                }
            }
        }
        assert get_custom_provider_rpm_throttle_threshold(
            "https://gateway.example/v1/", config=config
        ) == 4

        parsed = object()
        raw_response = SimpleNamespace(headers={}, parse=lambda: parsed)
        raw_create = MagicMock(return_value=raw_response)
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    with_raw_response=SimpleNamespace(create=raw_create),
                    create=MagicMock(),
                )
            )
        )
        agent = SimpleNamespace(
            api_mode="chat_completions",
            provider="openai",
            _capture_rate_limits=MagicMock(),
        )
        assert _dispatch_nonstreaming_api_request(
            agent,
            {"model": "test", "messages": []},
            make_client=lambda _reason: client,
        ) is parsed
        agent._capture_rate_limits.assert_called_once_with(raw_response)

        agent = MagicMock(
            platform="cron", api_mode="chat_completions", provider="openai"
        )
        agent._interrupt_requested = False
        with (
            patch("agent.rate_limit_tracker.wait_for_low_rpm") as throttle,
            patch("agent.chat_completion_helpers.direct_api_call", return_value=parsed),
        ):
            assert interruptible_api_call(agent, {"messages": []}) is parsed
        throttle.assert_called_once()

        agent.platform = "cli"
        agent.api_mode = "codex_responses"
        agent._interruptible_api_call.return_value = parsed
        with patch("agent.rate_limit_tracker.wait_for_low_rpm") as throttle:
            assert interruptible_streaming_api_call(agent, {"input": []}) is parsed
        throttle.assert_called_once()



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
