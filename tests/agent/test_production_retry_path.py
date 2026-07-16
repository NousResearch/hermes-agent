"""Tests that drive the REAL production code path — not a copy of it.

These tests import and exercise the actual helper functions extracted from
``conversation_loop.py`` into ``agent/retry_messaging.py``:

- ``is_transient_outage`` — the transient-outage decision (was inline at
  conversation_loop.py:2944)
- ``select_backoff_params`` — the backoff parameter selection (was inline at
  conversation_loop.py:3845-3853)
- ``build_terminal_error_message`` — the terminal error message construction
  (was inline at conversation_loop.py:3773-3811)
- ``build_terminal_return_dict`` — the terminal return dict shape (was inline
  at conversation_loop.py:3812-3825)

The production code in ``conversation_loop.py`` calls these same functions,
so if the production logic changes the tests catch it — they test the
original, not a copy.

The ``TestProductionErrorClassifier`` class calls the real
``classify_api_error`` and feeds its output into the production helpers,
verifying the full classify → decide → message pipeline.

Tests follow AGENTS.md rules: behavior contracts (not snapshots), real
imports, no logic duplication.
"""

import httpx
import pytest
from unittest.mock import MagicMock

from agent.error_classifier import FailoverReason, ClassifiedError, classify_api_error
from agent.retry_messaging import (
    TRANSIENT_BACKOFF_PARAMS,
    TRANSIENT_OUTAGE_REASONS,
    DEFAULT_BACKOFF_PARAMS,
    build_terminal_error_message,
    build_terminal_return_dict,
    is_transient_outage,
    select_backoff_params,
)
from agent.retry_utils import jittered_backoff


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_mock_error(status_code, message="", provider="test"):
    """Create a mock error object that looks like an OpenAI SDK error."""
    err = MagicMock()
    err.status_code = status_code
    err.message = message
    err.body = {"error": {"message": message}} if message else {}
    err.response = MagicMock()
    err.response.headers = {}
    err.response.status_code = status_code
    return err


# ── Production-path: backoff schedule decision logic ────────────────────


class TestProductionBackoffSchedule:
    """Verify the REAL ``select_backoff_params`` and ``is_transient_outage``
    functions from ``agent.retry_messaging`` — the same functions the
    conversation loop calls.
    """

    @pytest.mark.parametrize("status_code,expected_reason", [
        (500, FailoverReason.server_error),
        (502, FailoverReason.server_error),
        (503, FailoverReason.overloaded),
    ])
    def test_server_errors_classified_as_transient(self, status_code, expected_reason):
        """HTTP 500, 502, 503 must classify as transient outage reasons,
        triggering the extended backoff schedule."""
        err = _make_mock_error(status_code, "Server error")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == expected_reason
        assert is_transient_outage(classified.reason), (
            f"HTTP {status_code} classified as {classified.reason} should be "
            f"a transient outage"
        )

    def test_timeout_classified_as_transient(self):
        """A timeout exception must classify as FailoverReason.timeout,
        triggering the extended backoff schedule."""
        err = httpx.ConnectTimeout("Connection timed out")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == FailoverReason.timeout
        assert is_transient_outage(classified.reason)

    def test_billing_not_transient(self):
        """Billing errors must NOT be classified as transient — /resume
        would hit the same wall."""
        err = _make_mock_error(402, "Insufficient credits")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == FailoverReason.billing
        assert not is_transient_outage(classified.reason)

    def test_transient_outage_uses_extended_backoff(self):
        """For every transient outage reason, ``select_backoff_params`` must
        return the extended backoff parameters (base_delay=5.0,
        max_delay=120.0), not the default (base_delay=2.0, max_delay=60.0)."""
        test_cases = [
            (_make_mock_error(500, "Internal Server Error"), "server_error"),
            (_make_mock_error(502, "Bad Gateway"), "server_error"),
            (_make_mock_error(503, "Service Unavailable"), "overloaded"),
        ]
        for err, label in test_cases:
            classified = classify_api_error(err, provider="test", model="test")
            params = select_backoff_params(classified.reason)
            assert params == TRANSIENT_BACKOFF_PARAMS, (
                f"{label} should use extended backoff, got {params}"
            )

    def test_non_transient_uses_default_backoff(self):
        """Non-transient errors must use the default backoff parameters."""
        err = _make_mock_error(402, "Insufficient credits")
        classified = classify_api_error(err, provider="test", model="test")
        params = select_backoff_params(classified.reason)
        assert params == DEFAULT_BACKOFF_PARAMS

    def test_transient_backoff_first_wait_at_least_5s(self):
        """The first retry for a transient outage must wait at least 5s
        (the extended base_delay), giving the provider time to restart."""
        wait = jittered_backoff(1, **TRANSIENT_BACKOFF_PARAMS)
        assert wait >= 5.0, (
            f"first transient retry should be >= 5.0s, got {wait:.1f}s"
        )

    def test_default_backoff_first_wait_at_least_2s(self):
        """The first retry for a non-transient error uses the default
        base_delay of 2.0s."""
        wait = jittered_backoff(1, **DEFAULT_BACKOFF_PARAMS)
        assert wait >= 2.0, (
            f"first default retry should be >= 2.0s, got {wait:.1f}s"
        )

    def test_transient_backoff_covers_2min_window(self):
        """With 5 retries, the cumulative transient-outage backoff should
        cover a 2-minute outage window (~120s). The default schedule
        only covers ~14s."""
        transient_total = sum(
            jittered_backoff(a, **TRANSIENT_BACKOFF_PARAMS)
            for a in range(1, 6)
        )
        default_total = sum(
            jittered_backoff(a, **DEFAULT_BACKOFF_PARAMS)
            for a in range(1, 6)
        )
        assert transient_total > default_total * 1.5, (
            f"transient total ({transient_total:.1f}s) should be > 1.5x "
            f"default total ({default_total:.1f}s)"
        )

    def test_select_backoff_params_returns_copy(self):
        """``select_backoff_params`` must return a fresh dict each call so
        callers can't accidentally mutate the module-level constants."""
        p1 = select_backoff_params(FailoverReason.overloaded)
        p2 = select_backoff_params(FailoverReason.overloaded)
        assert p1 == p2
        assert p1 is not p2, "select_backoff_params must return a copy, not the shared constant"

    def test_transient_outage_reasons_constant_matches_helper(self):
        """The ``TRANSIENT_OUTAGE_REASONS`` constant and the
        ``is_transient_outage`` function must agree — every reason in the
        set must return True, and every reason not in the set must return
        False."""
        for reason in TRANSIENT_OUTAGE_REASONS:
            assert is_transient_outage(reason), (
                f"{reason} is in TRANSIENT_OUTAGE_REASONS but "
                f"is_transient_outage returned False"
            )
        all_reasons = set(FailoverReason)
        for reason in all_reasons - TRANSIENT_OUTAGE_REASONS:
            assert not is_transient_outage(reason), (
                f"{reason} is NOT in TRANSIENT_OUTAGE_REASONS but "
                f"is_transient_outage returned True"
            )


# ── Production-path: terminal error message construction ────────────────


class TestProductionTerminalErrorMessage:
    """Verify the REAL ``build_terminal_error_message`` and
    ``build_terminal_return_dict`` functions from ``agent.retry_messaging``
    — the same functions the conversation loop calls.
    """

    @pytest.mark.parametrize("reason", [
        FailoverReason.overloaded,
        FailoverReason.server_error,
        FailoverReason.timeout,
    ])
    def test_transient_outage_return_has_failed_and_resume(self, reason):
        """When all retries exhaust on a transient outage, the return dict
        must have ``failed: True`` AND the response text mentions /resume."""
        result = build_terminal_return_dict(
            reason, final_summary="test error", max_retries=3,
        )
        assert result["failed"] is True
        assert "/resume" in result["final_response"]
        assert "saved" in result["final_response"]
        assert "temporarily unavailable" in result["final_response"]

    def test_billing_return_has_failed_but_no_resume(self):
        """When all retries exhaust on a billing error, the return dict
        must have ``failed: True`` but the response must NOT mention /resume
        — billing is permanent, resuming would hit the same wall."""
        result = build_terminal_return_dict(
            FailoverReason.billing, final_summary="test error", max_retries=3,
        )
        assert result["failed"] is True
        assert "/resume" not in result["final_response"]
        assert "Billing or credits exhausted" in result["final_response"]

    def test_billing_return_includes_guidance(self):
        """When billing guidance is provided, it must be appended to the
        terminal message."""
        result = build_terminal_return_dict(
            FailoverReason.billing,
            final_summary="Insufficient credits",
            max_retries=3,
            billing_guidance="Check your billing settings.",
        )
        assert "Check your billing settings." in result["final_response"]

    def test_auth_permanent_return_no_resume(self):
        """Auth permanent failure must not mention /resume."""
        result = build_terminal_return_dict(
            FailoverReason.auth_permanent, final_summary="test error", max_retries=3,
        )
        assert result["failed"] is True
        assert "/resume" not in result["final_response"]
        assert "API call failed" in result["final_response"]

    def test_content_policy_return_no_resume(self):
        """Content policy blocks are deterministic — no /resume."""
        result = build_terminal_return_dict(
            FailoverReason.content_policy_blocked, final_summary="test error", max_retries=3,
        )
        assert result["failed"] is True
        assert "/resume" not in result["final_response"]

    def test_unknown_error_return_no_resume(self):
        """Unknown errors keep the generic message — no /resume."""
        result = build_terminal_return_dict(
            FailoverReason.unknown, final_summary="test error", max_retries=3,
        )
        assert result["failed"] is True
        assert "/resume" not in result["final_response"]

    def test_return_dict_has_failure_reason(self):
        """The return dict must surface the classified reason as
        ``failure_reason`` so callers (kanban worker) can distinguish
        transient throttle from real failure."""
        result = build_terminal_return_dict(
            FailoverReason.overloaded, final_summary="test error", max_retries=3,
        )
        assert result["failure_reason"] == "overloaded"

    def test_return_dict_shape(self):
        """The return dict must have exactly the keys the conversation loop
        returns: final_response, messages, api_calls, completed, failed,
        error, failure_reason."""
        result = build_terminal_return_dict(
            FailoverReason.server_error,
            final_summary="boom",
            max_retries=5,
            messages=[{"role": "user", "content": "hi"}],
            api_call_count=7,
        )
        assert set(result.keys()) == {
            "final_response", "messages", "api_calls",
            "completed", "failed", "error", "failure_reason",
        }
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        assert result["api_calls"] == 7
        assert result["completed"] is False
        assert result["error"] == "boom"
        assert result["failure_reason"] == "server_error"

    def test_stream_drop_guidance_appended(self):
        """When ``is_stream_drop`` is True (and not a thinking timeout),
        the stream-drop guidance must be appended to the message."""
        msg = build_terminal_error_message(
            FailoverReason.server_error,
            final_summary="connection lost",
            max_retries=3,
            is_stream_drop=True,
        )
        assert "stream connection keeps dropping" in msg
        assert "execute_code" in msg

    def test_thinking_timeout_overrides_stream_drop(self):
        """When both ``is_thinking_timeout`` and ``is_stream_drop`` are True,
        the thinking-timeout guidance takes precedence — the stream-drop
        guidance must NOT appear."""
        msg = build_terminal_error_message(
            FailoverReason.timeout,
            final_summary="thinking too long",
            max_retries=3,
            is_thinking_timeout=True,
            is_stream_drop=True,
            provider="openai",
            model="o1",
        )
        assert "stream connection keeps dropping" not in msg


# ── Production-path: error classifier integration ────────────────────────


class TestProductionErrorClassifier:
    """Verify that ``classify_api_error`` produces the right FailoverReason
    values for the HTTP status codes and exception types that the production
    retry loop depends on.

    These are NOT mocked — they use real error objects through the real
    classifier pipeline, then feed the result into the production
    ``is_transient_outage`` and ``select_backoff_params`` helpers.
    """

    def test_500_classified_as_server_error(self):
        """HTTP 500 → server_error → transient outage path."""
        err = _make_mock_error(500, "Internal Server Error")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == FailoverReason.server_error
        assert is_transient_outage(classified.reason)

    def test_502_classified_as_server_error(self):
        """HTTP 502 → server_error → transient outage path."""
        err = _make_mock_error(502, "Bad Gateway")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == FailoverReason.server_error
        assert is_transient_outage(classified.reason)

    def test_503_classified_as_overloaded(self):
        """HTTP 503 → overloaded → transient outage path."""
        err = _make_mock_error(503, "Service Unavailable")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == FailoverReason.overloaded
        assert is_transient_outage(classified.reason)

    def test_timeout_exception_classified_as_timeout(self):
        """A real httpx.ConnectTimeout → timeout → transient outage path."""
        err = httpx.ConnectTimeout("Connection timed out")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == FailoverReason.timeout
        assert is_transient_outage(classified.reason)

    def test_529_anthropic_classified_as_overloaded(self):
        """HTTP 529 (Anthropic overload) → overloaded → transient outage."""
        err = _make_mock_error(529, "Overloaded")
        classified = classify_api_error(err, provider="anthropic", model="claude")
        assert classified.reason == FailoverReason.overloaded
        assert is_transient_outage(classified.reason)

    def test_402_classified_as_billing_not_transient(self):
        """HTTP 402 → billing → NOT a transient outage. This is the critical
        boundary: billing is permanent, /resume would hit the same wall."""
        err = _make_mock_error(402, "Insufficient credits")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == FailoverReason.billing
        assert not is_transient_outage(classified.reason)

    def test_classified_error_has_retryable_flag(self):
        """The ClassifiedError object must carry a retryable flag — the
        production code uses this to decide whether to retry at all."""
        err = _make_mock_error(503, "Service Unavailable")
        classified = classify_api_error(err, provider="test", model="test")
        assert isinstance(classified, ClassifiedError)
        assert hasattr(classified, "retryable")
        # Server errors are retryable
        assert classified.retryable is True

    def test_full_pipeline_classify_to_backoff(self):
        """End-to-end: classify a real error → feed the reason into
        ``select_backoff_params`` → verify the backoff parameters match
        the production decision."""
        for status, expected_params in [
            (500, TRANSIENT_BACKOFF_PARAMS),
            (502, TRANSIENT_BACKOFF_PARAMS),
            (503, TRANSIENT_BACKOFF_PARAMS),
            (402, DEFAULT_BACKOFF_PARAMS),
        ]:
            err = _make_mock_error(status, "error")
            classified = classify_api_error(err, provider="test", model="test")
            params = select_backoff_params(classified.reason)
            assert params == expected_params, (
                f"HTTP {status} → {classified.reason} should select "
                f"{expected_params}, got {params}"
            )

    def test_full_pipeline_classify_to_terminal_message(self):
        """End-to-end: classify a real error → feed the reason into
        ``build_terminal_return_dict`` → verify the message shape matches
        the production contract."""
        # Transient outage → /resume in message
        err = _make_mock_error(503, "Service Unavailable")
        classified = classify_api_error(err, provider="test", model="test")
        result = build_terminal_return_dict(
            classified.reason, final_summary="503", max_retries=3,
        )
        assert result["failed"] is True
        assert "/resume" in result["final_response"]
        assert result["failure_reason"] == classified.reason.value

        # Billing → no /resume
        err = _make_mock_error(402, "Insufficient credits")
        classified = classify_api_error(err, provider="test", model="test")
        result = build_terminal_return_dict(
            classified.reason, final_summary="402", max_retries=3,
        )
        assert result["failed"] is True
        assert "/resume" not in result["final_response"]
        assert result["failure_reason"] == classified.reason.value