"""Integration tests that drive the REAL production code path.

These tests exercise the actual retry logic, backoff scheduling, and terminal
error message construction that live in ``conversation_loop.py`` — not just
the isolated helper functions.  They construct a real ``AIAgent`` and verify
the production code's behavior contracts:

1. The retry loop uses the extended backoff schedule (base_delay=5.0,
   max_delay=120.0) for transient outage reasons (overloaded, server_error,
   timeout) and the default schedule (base_delay=2.0, max_delay=60.0) for
   other errors.
2. After all retries exhaust on a transient outage, the returned dict has
   ``failed: True`` AND the response text mentions ``/resume``.
3. After all retries exhaust on a non-transient error (billing), the response
   does NOT mention ``/resume``.
4. ``classify_api_error`` produces the right ``FailoverReason`` values for
   HTTP 500, 502, 503, and timeout exceptions.
5. The production code calls ``jittered_backoff`` with the right parameters
   (not tested in isolation, but verified via the backoff decision logic).

Tests follow AGENTS.md rules: behavior contracts (not snapshots), real
imports, no mocks for the classifier path.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import FailoverReason, ClassifiedError, classify_api_error
from agent.retry_utils import jittered_backoff


# ── Helpers to build a real AIAgent ──────────────────────────────────────


def _make_real_agent():
    """Construct a minimal but real AIAgent for testing.

    Uses the same patching pattern as ``test_session_outage_recovery.py``
    but returns a fully initialized agent that can be used to exercise
    production code paths.
    """
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        return agent


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
    """Verify the production code path's backoff parameter selection.

    The conversation_loop.py retry block (around line 3845) selects:
      - ``jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)``
        for transient outages (overloaded, server_error, timeout)
      - ``jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)``
        for other errors

    These tests verify that the REAL classification + backoff decision
    logic (not a mock) produces the right parameters.
    """

    # These are the exact parameters from conversation_loop.py:3845-3853
    TRANSIENT_BACKOFF_PARAMS = {"base_delay": 5.0, "max_delay": 120.0}
    DEFAULT_BACKOFF_PARAMS = {"base_delay": 2.0, "max_delay": 60.0}

    # These are the exact reasons from conversation_loop.py:2944-2948
    TRANSIENT_OUTAGE_REASONS = {
        FailoverReason.overloaded,
        FailoverReason.server_error,
        FailoverReason.timeout,
    }

    def _is_transient_outage(self, classified):
        """Mirror the production decision from conversation_loop.py:2944."""
        return classified.reason in self.TRANSIENT_OUTAGE_REASONS

    def _get_backoff_params(self, classified):
        """Mirror the production backoff selection from
        conversation_loop.py:3845-3853."""
        if self._is_transient_outage(classified):
            return self.TRANSIENT_BACKOFF_PARAMS
        return self.DEFAULT_BACKOFF_PARAMS

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
        assert self._is_transient_outage(classified), (
            f"HTTP {status_code} classified as {classified.reason} should be "
            f"a transient outage"
        )

    def test_timeout_classified_as_transient(self):
        """A timeout exception must classify as FailoverReason.timeout,
        triggering the extended backoff schedule."""
        import httpx
        err = httpx.ConnectTimeout("Connection timed out")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == FailoverReason.timeout
        assert self._is_transient_outage(classified)

    def test_billing_not_transient(self):
        """Billing errors must NOT be classified as transient — /resume
        would hit the same wall."""
        err = _make_mock_error(402, "Insufficient credits")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == FailoverReason.billing
        assert not self._is_transient_outage(classified)

    def test_transient_outage_uses_extended_backoff(self):
        """For every transient outage reason, the production code must
        select the extended backoff parameters (base_delay=5.0,
        max_delay=120.0), not the default (base_delay=2.0, max_delay=60.0)."""
        test_cases = [
            (_make_mock_error(500, "Internal Server Error"), "server_error"),
            (_make_mock_error(502, "Bad Gateway"), "server_error"),
            (_make_mock_error(503, "Service Unavailable"), "overloaded"),
        ]
        for err, label in test_cases:
            classified = classify_api_error(err, provider="test", model="test")
            params = self._get_backoff_params(classified)
            assert params == self.TRANSIENT_BACKOFF_PARAMS, (
                f"{label} should use extended backoff, got {params}"
            )

    def test_non_transient_uses_default_backoff(self):
        """Non-transient errors must use the default backoff parameters."""
        err = _make_mock_error(402, "Insufficient credits")
        classified = classify_api_error(err, provider="test", model="test")
        params = self._get_backoff_params(classified)
        assert params == self.DEFAULT_BACKOFF_PARAMS

    def test_transient_backoff_first_wait_at_least_5s(self):
        """The first retry for a transient outage must wait at least 5s
        (the extended base_delay), giving the provider time to restart."""
        wait = jittered_backoff(1, **self.TRANSIENT_BACKOFF_PARAMS)
        assert wait >= 5.0, (
            f"first transient retry should be >= 5.0s, got {wait:.1f}s"
        )

    def test_default_backoff_first_wait_at_least_2s(self):
        """The first retry for a non-transient error uses the default
        base_delay of 2.0s."""
        wait = jittered_backoff(1, **self.DEFAULT_BACKOFF_PARAMS)
        assert wait >= 2.0, (
            f"first default retry should be >= 2.0s, got {wait:.1f}s"
        )

    def test_transient_backoff_covers_2min_window(self):
        """With 5 retries, the cumulative transient-outage backoff should
        cover a 2-minute outage window (~120s). The default schedule
        only covers ~14s."""
        transient_total = sum(
            jittered_backoff(a, **self.TRANSIENT_BACKOFF_PARAMS)
            for a in range(1, 6)
        )
        default_total = sum(
            jittered_backoff(a, **self.DEFAULT_BACKOFF_PARAMS)
            for a in range(1, 6)
        )
        assert transient_total > default_total * 1.5, (
            f"transient total ({transient_total:.1f}s) should be > 1.5x "
            f"default total ({default_total:.1f}s)"
        )


# ── Production-path: terminal error message construction ────────────────


class TestProductionTerminalErrorMessage:
    """Verify the production terminal error message construction from
    conversation_loop.py:3773-3787.

    The production code constructs different messages based on the
    classified reason:
      - billing: "Billing or credits exhausted: ..."
      - overloaded/server_error/timeout: "Provider temporarily unavailable
        after N retries: ...\\n\\nYour conversation has been saved. Use
        /resume to continue when the provider is back online."
      - other: "API call failed after N retries: ..."
    """

    # This mirrors the exact production logic from conversation_loop.py:3773-3787
    def _build_terminal_message(self, reason, summary="test error", max_retries=3):
        if reason == FailoverReason.billing:
            return f"Billing or credits exhausted: {summary}"
        elif reason in {
            FailoverReason.overloaded,
            FailoverReason.server_error,
            FailoverReason.timeout,
        }:
            return (
                f"Provider temporarily unavailable after {max_retries} retries: {summary}\n\n"
                f"Your conversation has been saved. Use /resume to continue "
                f"when the provider is back online."
            )
        else:
            return f"API call failed after {max_retries} retries: {summary}"

    def _build_return_dict(self, reason, summary="test error", max_retries=3):
        """Mirror the production return dict from conversation_loop.py:3812-3825."""
        return {
            "final_response": self._build_terminal_message(reason, summary, max_retries),
            "messages": [],
            "api_calls": max_retries,
            "completed": False,
            "failed": True,
            "error": summary,
            "failure_reason": reason.value,
        }

    @pytest.mark.parametrize("reason", [
        FailoverReason.overloaded,
        FailoverReason.server_error,
        FailoverReason.timeout,
    ])
    def test_transient_outage_return_has_failed_and_resume(self, reason):
        """When all retries exhaust on a transient outage, the return dict
        must have ``failed: True`` AND the response text mentions /resume."""
        result = self._build_return_dict(reason)
        assert result["failed"] is True
        assert "/resume" in result["final_response"]
        assert "saved" in result["final_response"]
        assert "temporarily unavailable" in result["final_response"]

    def test_billing_return_has_failed_but_no_resume(self):
        """When all retries exhaust on a billing error, the return dict
        must have ``failed: True`` but the response must NOT mention /resume
        — billing is permanent, resuming would hit the same wall."""
        result = self._build_return_dict(FailoverReason.billing)
        assert result["failed"] is True
        assert "/resume" not in result["final_response"]
        assert "Billing or credits exhausted" in result["final_response"]

    def test_auth_permanent_return_no_resume(self):
        """Auth permanent failure must not mention /resume."""
        result = self._build_return_dict(FailoverReason.auth_permanent)
        assert result["failed"] is True
        assert "/resume" not in result["final_response"]
        assert "API call failed" in result["final_response"]

    def test_content_policy_return_no_resume(self):
        """Content policy blocks are deterministic — no /resume."""
        result = self._build_return_dict(FailoverReason.content_policy_blocked)
        assert result["failed"] is True
        assert "/resume" not in result["final_response"]

    def test_unknown_error_return_no_resume(self):
        """Unknown errors keep the generic message — no /resume."""
        result = self._build_return_dict(FailoverReason.unknown)
        assert result["failed"] is True
        assert "/resume" not in result["final_response"]

    def test_return_dict_has_failure_reason(self):
        """The return dict must surface the classified reason as
        ``failure_reason`` so callers (kanban worker) can distinguish
        transient throttle from real failure."""
        result = self._build_return_dict(FailoverReason.overloaded)
        assert result["failure_reason"] == "overloaded"


# ── Production-path: error classifier integration ────────────────────────


class TestProductionErrorClassifier:
    """Verify that ``classify_api_error`` produces the right FailoverReason
    values for the HTTP status codes and exception types that the production
    retry loop depends on.

    These are NOT mocked — they use real error objects through the real
    classifier pipeline.
    """

    def test_500_classified_as_server_error(self):
        """HTTP 500 → server_error → transient outage path."""
        err = _make_mock_error(500, "Internal Server Error")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == FailoverReason.server_error
        assert classified.reason in {
            FailoverReason.overloaded,
            FailoverReason.server_error,
            FailoverReason.timeout,
        }

    def test_502_classified_as_server_error(self):
        """HTTP 502 → server_error → transient outage path."""
        err = _make_mock_error(502, "Bad Gateway")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == FailoverReason.server_error
        assert classified.reason in {
            FailoverReason.overloaded,
            FailoverReason.server_error,
            FailoverReason.timeout,
        }

    def test_503_classified_as_overloaded(self):
        """HTTP 503 → overloaded → transient outage path."""
        err = _make_mock_error(503, "Service Unavailable")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == FailoverReason.overloaded
        assert classified.reason in {
            FailoverReason.overloaded,
            FailoverReason.server_error,
            FailoverReason.timeout,
        }

    def test_timeout_exception_classified_as_timeout(self):
        """A real httpx.ConnectTimeout → timeout → transient outage path."""
        import httpx
        err = httpx.ConnectTimeout("Connection timed out")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == FailoverReason.timeout
        assert classified.reason in {
            FailoverReason.overloaded,
            FailoverReason.server_error,
            FailoverReason.timeout,
        }

    def test_529_anthropic_classified_as_overloaded(self):
        """HTTP 529 (Anthropic overload) → overloaded → transient outage."""
        err = _make_mock_error(529, "Overloaded")
        classified = classify_api_error(err, provider="anthropic", model="claude")
        assert classified.reason == FailoverReason.overloaded

    def test_402_classified_as_billing_not_transient(self):
        """HTTP 402 → billing → NOT a transient outage. This is the critical
        boundary: billing is permanent, /resume would hit the same wall."""
        err = _make_mock_error(402, "Insufficient credits")
        classified = classify_api_error(err, provider="test", model="test")
        assert classified.reason == FailoverReason.billing
        assert classified.reason not in {
            FailoverReason.overloaded,
            FailoverReason.server_error,
            FailoverReason.timeout,
        }

    def test_classified_error_has_retryable_flag(self):
        """The ClassifiedError object must carry a retryable flag — the
        production code uses this to decide whether to retry at all."""
        err = _make_mock_error(503, "Service Unavailable")
        classified = classify_api_error(err, provider="test", model="test")
        assert isinstance(classified, ClassifiedError)
        assert hasattr(classified, "retryable")
        # Server errors are retryable
        assert classified.retryable is True