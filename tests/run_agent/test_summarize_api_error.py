"""Tests for AIAgent._summarize_api_error in run_agent.py.

The summarizer walks the exception chain to find the deepest network
marker and surfaces it as a one-liner. The previous behaviour was a
generic "you may be offline" — fine for desktop users, but actively
misleading on a headless Pi where the failure is more often wifi
flapping, ENETUNREACH, or DNS hiccups than the provider being down.

The patch closes the gap between cron-failure-as-incomplete-no-output
and the operator's ability to triage the root cause.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def summarizer():
    """Import AIAgent fresh so the patched _summarize_api_error is exercised."""
    from run_agent import AIAgent

    return AIAgent._summarize_api_error


def test_summarize_returns_root_cause_for_enetunreach(summarizer):
    """ENETUNREACH from wlan0 → surfacing the kernel errno helps triage."""
    # Build the chain: APIConnectionError -> httpcore -> OSError ENETUNREACH.
    # This mirrors what httpx + anthropic SDK produce on the Pi when wifi
    # flaps.
    try:
        try:
            raise ConnectionError("Connection error.")
        except ConnectionError as e:
            raise OSError(101, "Network is unreachable") from e
    except OSError as e:
        bottom = e

    # The summarizer walks the chain via __cause__ — both frames must be
    # reachable, so wrap the OSError in something the SDK would actually
    # raise.
    class FakeAPIConnError(Exception):
        pass

    try:
        raise FakeAPIConnError("Connection error.") from bottom
    except FakeAPIConnError as e:
        top = e

    result = summarizer(top)
    assert "Network unreachable to provider" in result, (
        f"summarizer should announce it's a host-network problem, got: {result!r}"
    )
    assert "[Errno 101]" in result, (
        f"summarizer should include the actual errno, got: {result!r}"
    )
    assert "Network is unreachable" in result


def test_summarize_handles_dns_failure(summarizer):
    """DNS resolution failure (Errno -3) should surface with the real error."""
    try:
        raise ConnectionError("Connection error.")
    except ConnectionError as e:
        try:
            raise OSError(-3, "Temporary failure in name resolution") from e
        except OSError as dns_err:
            wrapped_err = dns_err

    class FakeAPIConnError(Exception):
        pass

    try:
        raise FakeAPIConnError("Connection error.") from wrapped_err
    except FakeAPIConnError as e:
        top = e

    result = summarizer(top)
    assert "Network unreachable to provider" in result
    assert "[Errno -3]" in result


def test_summarize_handles_connection_refused(summarizer):
    """ECONNREFUSED (provider up but port closed) → mention refused."""
    try:
        raise ConnectionError("Connection refused")
    except ConnectionError as e:
        wrapped = e

    class FakeAPIConnError(Exception):
        pass

    try:
        raise FakeAPIConnError("Connection error.") from wrapped
    except FakeAPIConnError as e:
        top = e

    result = summarizer(top)
    assert "Network unreachable to provider" in result
    assert "refused" in result.lower()


def test_summarize_handles_plain_value_error(summarizer):
    """Non-network errors should NOT be rebranded as network errors.

    This guards against the patch accidentally swallowing unrelated
    failures and misleading operators about the cause.
    """
    err = ValueError("Expected ident at line 1 column 2 (line 1)")
    result = summarizer(err)
    assert "Network unreachable" not in result, (
        "ValueError must NOT be tagged as a network problem"
    )
    # The raw message must be preserved so operators see what actually
    # happened. The exact wording depends on upstream's fallback
    # branch (HTTP prefix + raw[:N]), so check for the message text.
    assert "Expected ident at line" in result


def test_summarize_handles_html_error_page(summarizer):
    """Cloudflare 503 page → extract <title>, don't mis-tag as network."""
    err = Exception(
        "<!DOCTYPE html><html><head><title>503 Service Temporarily Unavailable</title></head>"
    )
    result = summarizer(err)
    assert "Network unreachable" not in result
    assert "503 Service Temporarily Unavailable" in result


def test_summarize_truncates_very_long_messages(summarizer):
    """Very long underlying errors must be truncated to fit Telegram notifications."""
    long_msg = "Network is unreachable. " * 50  # ~1100 chars
    try:
        raise ConnectionError("Connection error.")
    except ConnectionError as e:
        try:
            raise OSError(101, long_msg) from e
        except OSError as big_err:
            wrapped = big_err

    class FakeAPIConnError(Exception):
        pass

    try:
        raise FakeAPIConnError("Connection error.") from wrapped
    except FakeAPIConnError as e:
        top = e

    result = summarizer(top)
    # Result should be under 220 chars to leave room for the cron
    # notification header ("Cronjob Response: ... job_id: ..."). The
    # summary template adds ~85 chars of prefix/suffix around the
    # truncated inner message (117 chars + "..." if needed).
    assert len(result) < 220, f"summarized message too long: {len(result)} chars"


def test_summarize_handles_unknown_error_gracefully(summarizer):
    """Unrecognized exceptions fall through to a generic string, not crash."""
    class WeirdError(Exception):
        pass

    err = WeirdError("something exotic happened")
    # Should not raise.
    result = summarizer(err)
    assert isinstance(result, str)
    assert len(result) > 0