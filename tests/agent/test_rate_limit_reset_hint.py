"""A rate-limit retry names the provider's reset window on the status line (#26889).

"Rate limited. Waiting 60s" hides the one fact that decides whether to wait or
switch models: a per-minute throttle and a 13-minute plan window look identical
without it. The hint is derived from the same ``reset_at`` the credential pool
uses, so the relative ``resets_in_seconds`` a Codex/ChatGPT usage-limit body
carries must land there too.
"""

import time
from unittest.mock import MagicMock

import pytest

from agent.agent_runtime_helpers import extract_api_error_context
from agent.turn_recovery import compute_error_backoff, reset_hint


def _codex_429(**fields):
    err = Exception("HTTP 429: The usage limit has been reached")
    err.body = {"error": {"type": "usage_limit_reached", "message": "The usage limit has been reached", **fields}}
    return err


def test_relative_resets_in_seconds_becomes_reset_at_when_no_epoch_is_given():
    ctx = extract_api_error_context(_codex_429(resets_in_seconds=756))
    assert abs(ctx["reset_at"] - (time.time() + 756)) < 5
    # An explicit epoch wins over the relative field (same precedence as the credential pool).
    ctx = extract_api_error_context(_codex_429(resets_at=1_900_000_000, resets_in_seconds=756))
    assert ctx["reset_at"] == 1_900_000_000
    # No signal at all → no hint, not a crash.
    assert reset_hint(Exception("HTTP 429")) == "" and reset_hint(_codex_429(resets_in_seconds=-5)) == ""


def test_rate_limit_retry_status_names_the_reset_window():
    agent = MagicMock()
    from agent.status_output import StatusOutputMixin
    for name in ("_emit_diagnostic_wait", "_buffer_diagnostic_status"):
        setattr(agent, name, getattr(StatusOutputMixin, name).__get__(agent))
    agent._client_log_context.return_value = ""

    compute_error_backoff(
        agent, _codex_429(resets_in_seconds=756), retry_count=1, max_retries=3,
        is_rate_limited=True, is_zai_coding_overload=False,
        base_url="https://example.test/v1", model="test/model",
    )

    text = agent._buffer_status.call_args.args[0]
    assert text.startswith("⏱️ Rate limited. Resets in ~13m. Waiting ")


def test_live_wait_line_names_the_reset_window_too():
    """The buffered line replays only if every retry fails; the line rendered during the wait is
    ``_emit_diagnostic_wait`` → ``_emit_wait_notice``, and it must carry the same hint."""
    agent = MagicMock()
    from agent.status_output import StatusOutputMixin
    for name in ("_emit_diagnostic_wait", "_buffer_diagnostic_status"):
        setattr(agent, name, getattr(StatusOutputMixin, name).__get__(agent))
    agent._client_log_context.return_value = ""

    def run(err):
        compute_error_backoff(
            agent, err, retry_count=1, max_retries=3, is_rate_limited=True, is_zai_coding_overload=False,
            base_url="https://example.test/v1", model="test/model",
        )
        return agent._emit_wait_notice.call_args.args[0]

    live = run(_codex_429(resets_in_seconds=756))
    assert live.startswith("⏳ rate limited — resets in ~13m, retrying in ") and "(attempt 1/3)" in live
    # No reset known → the existing anonymous-wait wording is unchanged.
    assert run(Exception("HTTP 429")).startswith("⏳ waiting on provider — retrying in ")


@pytest.mark.parametrize("headers, expected_seconds", [
    ({"x-ratelimit-reset-requests": "6m0s"}, 360),
    ({"x-ratelimit-reset-tokens": "1.5s"}, 1.5),
    ({"x-ratelimit-reset-requests": "20ms"}, 0.02),
    ({"anthropic-ratelimit-requests-reset": None}, 900),  # filled below with now+900 ISO-8601 'Z'
    ({"retry-after": "30", "x-ratelimit-reset-requests": "6m0s"}, 30),  # Retry-After still wins
])
def test_vendor_reset_headers_feed_reset_at(headers, expected_seconds):
    from datetime import datetime, timedelta, timezone
    if "anthropic-ratelimit-requests-reset" in headers:
        when = datetime.now(timezone.utc) + timedelta(seconds=900)
        headers["anthropic-ratelimit-requests-reset"] = when.strftime("%Y-%m-%dT%H:%M:%SZ")
    err = Exception("HTTP 429: rate limit")
    err.response = MagicMock(headers=headers)
    assert abs(extract_api_error_context(err)["reset_at"] - (time.time() + expected_seconds)) < 5
    # A reset already in the past is ignored rather than reported as "resets in ~0s".
    past = Exception("HTTP 429")
    past.response = MagicMock(headers={"anthropic-ratelimit-tokens-reset": "2020-01-01T00:00:00Z"})
    assert "reset_at" not in extract_api_error_context(past)
