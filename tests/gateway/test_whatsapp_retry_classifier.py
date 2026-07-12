"""Tests for the pure WhatsApp delivery retry classifier.

Retry policy (delivery-reliability plan):
- Retryable: connection refused before request acceptance, explicit 429/502/503/504.
- Ambiguous (never retried): timeouts — the request may have been accepted.
- Non-retryable: 400/401/403/404, any other HTTP status, unknown exceptions.
Categories must be sanitized — no exception text (PII) may leak into them.
"""

import asyncio
import errno

import pytest

from plugins.platforms.whatsapp.delivery_reliability import (
    AMBIGUOUS,
    NON_RETRYABLE,
    RETRYABLE,
    classify_delivery_failure,
)


# ---------------------------------------------------------------------------
# HTTP status classes
# ---------------------------------------------------------------------------

class TestHttpStatusClassification:
    @pytest.mark.parametrize("status", [429, 502, 503, 504])
    def test_explicit_retryable_statuses(self, status):
        result = classify_delivery_failure(status=status)
        assert result.decision == RETRYABLE
        assert result.category == f"http_{status}"
        assert result.status == status

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_permanent_client_errors(self, status):
        result = classify_delivery_failure(status=status)
        assert result.decision == NON_RETRYABLE
        assert result.category == f"http_{status}"

    @pytest.mark.parametrize("status", [409, 410, 422, 500, 501])
    def test_unlisted_statuses_are_never_retried(self, status):
        result = classify_delivery_failure(status=status)
        assert result.decision == NON_RETRYABLE


# ---------------------------------------------------------------------------
# Connection refusal (before request acceptance) — safe to retry
# ---------------------------------------------------------------------------

class TestConnectionRefused:
    def test_connection_refused_error(self):
        result = classify_delivery_failure(exception=ConnectionRefusedError())
        assert result.decision == RETRYABLE
        assert result.category == "connection_refused"

    def test_oserror_with_econnrefused_errno(self):
        exc = OSError(errno.ECONNREFUSED, "connection refused")
        result = classify_delivery_failure(exception=exc)
        assert result.decision == RETRYABLE
        assert result.category == "connection_refused"

    def test_aiohttp_style_wrapper_with_os_error_attr(self):
        """aiohttp.ClientConnectorError wraps the OSError in ``.os_error``."""
        class FakeConnectorError(Exception):
            os_error = ConnectionRefusedError()

        result = classify_delivery_failure(exception=FakeConnectorError())
        assert result.decision == RETRYABLE
        assert result.category == "connection_refused"


# ---------------------------------------------------------------------------
# Timeout — ambiguous: the request may have been accepted; never retry
# ---------------------------------------------------------------------------

class TestTimeout:
    @pytest.mark.parametrize("exc", [asyncio.TimeoutError(), TimeoutError()])
    def test_timeout_is_ambiguous(self, exc):
        result = classify_delivery_failure(exception=exc)
        assert result.decision == AMBIGUOUS
        assert result.category == "timeout"


# ---------------------------------------------------------------------------
# Unknown exceptions — never retried, category sanitized
# ---------------------------------------------------------------------------

class TestUnknownException:
    def test_unknown_exception_is_non_retryable(self):
        result = classify_delivery_failure(exception=ValueError("boom"))
        assert result.decision == NON_RETRYABLE
        assert result.category == "unknown_exception"

    def test_category_never_contains_exception_text(self):
        """PII gate: message bodies/phones in exception text must not leak."""
        exc = RuntimeError("failed to send 'hello' to +5511999998888@s.whatsapp.net")
        result = classify_delivery_failure(exception=exc)
        assert "5511999998888" not in result.category
        assert "hello" not in result.category

    def test_connection_reset_is_not_retried(self):
        """Reset after the request may have been accepted — never retry."""
        result = classify_delivery_failure(exception=ConnectionResetError())
        assert result.decision == NON_RETRYABLE


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------

class TestInputContract:
    def test_requires_status_or_exception(self):
        with pytest.raises(ValueError):
            classify_delivery_failure()

    def test_exception_takes_precedence_when_both_given(self):
        result = classify_delivery_failure(status=503, exception=TimeoutError())
        assert result.decision == AMBIGUOUS

    def test_success_status_is_rejected(self):
        """2xx is not a failure — classifying it is a caller bug."""
        with pytest.raises(ValueError):
            classify_delivery_failure(status=200)
