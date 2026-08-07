"""The reset-window hint must actually drive the retry delay.

Guards the wiring in agent/conversation_loop.py: a gateway 429 that carries no
Retry-After header but states "(reset after Ns)" in its body must produce a wait
of N seconds, not the generic exponential backoff — otherwise the retry lands
inside the still-closed window and the provider gets abandoned for a fallback.
"""

from agent.retry_utils import jittered_backoff, parse_reset_after_seconds


class _FakeResponse:
    """A 429 response the way an aggregator gateway sends it: no Retry-After."""

    def __init__(self):
        self.headers = {"content-type": "application/json"}


class _FakeRateLimitError(Exception):
    def __init__(self, summary):
        super().__init__(summary)
        self.response = _FakeResponse()
        self.summary = summary


def _resolve_wait(error_summary, headers, retry_count=1):
    """Mirror of the conversation_loop precedence: header, then body hint, then backoff."""
    retry_after = None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw:
        try:
            retry_after = min(float(raw), 600)
        except (TypeError, ValueError):
            retry_after = None
    if retry_after is None:
        retry_after = parse_reset_after_seconds(error_summary)
    return retry_after if retry_after else jittered_backoff(
        retry_count, base_delay=2.0, max_delay=60.0
    )


class TestBodyHintDrivesTheWait:
    def test_short_window_is_honored_instead_of_backoff(self):
        err = _FakeRateLimitError(
            "HTTP 429: [antigravity/claude-sonnet-4-6] [429]: "
            '{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}} (reset after 4s)'
        )
        assert _resolve_wait(str(err), err.response.headers) == 4.0

    def test_minute_window_is_honored(self):
        err = _FakeRateLimitError(
            "HTTP 429: [claude/claude-opus-5] [429]: rate_limit_error (reset after 2m 13s)"
        )
        assert _resolve_wait(str(err), err.response.headers) == 133.0

    def test_real_header_still_wins_over_body_hint(self):
        # A provider that does send Retry-After keeps its old behavior.
        summary = "HTTP 429: throttled (reset after 300s)"
        assert _resolve_wait(summary, {"retry-after": "7"}) == 7.0

    def test_without_hint_falls_back_to_jittered_backoff(self):
        wait = _resolve_wait("HTTP 429: rate limited", {}, retry_count=1)
        # jittered_backoff(1, base=2.0) -> 2.0 plus up to 50% jitter.
        assert 2.0 <= wait <= 3.0

    def test_overlong_window_falls_back_to_backoff_not_a_long_sleep(self):
        wait = _resolve_wait("HTTP 429: down (reset after 45m)", {}, retry_count=1)
        assert wait <= 3.0
