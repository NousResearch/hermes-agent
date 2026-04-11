"""Tests for iteration budget refund on transient network errors."""
from unittest.mock import MagicMock, patch

import pytest


class TestBudgetRefundOnTransientError:
    """Iteration budget should be refunded when API calls fail due to transient network errors."""

    def test_timeout_error_classified_as_refundable(self):
        """Connection timeout errors should have should_refund_budget=True."""
        from agent.error_classifier import classify_api_error, FailoverReason

        error = TimeoutError("Connection timed out")
        classified = classify_api_error(error)

        assert classified.reason == FailoverReason.timeout
        assert classified.retryable is True
        assert classified.should_refund_budget is True

    def test_connection_error_classified_as_refundable(self):
        """Connection errors should have should_refund_budget=True."""
        from agent.error_classifier import classify_api_error, FailoverReason

        error = ConnectionError("Connection refused")
        classified = classify_api_error(error)

        assert classified.reason == FailoverReason.timeout
        assert classified.should_refund_budget is True

    def test_os_error_classified_as_refundable(self):
        """OS errors (network unreachable) should have should_refund_budget=True."""
        from agent.error_classifier import classify_api_error, FailoverReason

        error = OSError("Network is unreachable")
        classified = classify_api_error(error)

        assert classified.reason == FailoverReason.timeout
        assert classified.should_refund_budget is True

    def test_context_overflow_not_refundable(self):
        """Context overflow errors should NOT have should_refund_budget=True (they used API resources)."""
        from agent.error_classifier import classify_api_error, FailoverReason

        error = Exception("context length exceeded")
        error.status_code = 400
        classified = classify_api_error(error)

        assert classified.reason == FailoverReason.context_overflow
        assert classified.should_refund_budget is False

    def test_rate_limit_not_refundable(self):
        """Rate limit errors should NOT have should_refund_budget=True (request reached the server)."""
        from agent.error_classifier import classify_api_error, FailoverReason

        error = Exception("rate limit exceeded")
        error.status_code = 429
        classified = classify_api_error(error)

        assert classified.reason == FailoverReason.rate_limit
        assert classified.should_refund_budget is False
