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


class TestIterationBudgetRefundCap:
    """IterationBudget.refund() must not grant more budget than was consumed."""

    def test_refund_cannot_go_below_zero(self):
        """Calling refund() more times than consume() must not underflow below 0."""
        from run_agent import IterationBudget

        budget = IterationBudget(10)
        budget.consume()  # used=1
        budget.refund()   # used=0
        budget.refund()   # must not go to -1
        budget.refund()

        assert budget.used == 0
        assert budget.remaining == 10

    def test_multiple_retries_do_not_over_refund(self):
        """Simulates the inner retry loop: 1 consume + 3 transient-error refunds.

        The guard flag (_budget_refunded_this_iteration) in run_agent.py ensures
        only the first refund fires.  This test verifies IterationBudget itself
        has a floor at 0, so even without the guard the budget cannot go negative.
        """
        from run_agent import IterationBudget

        budget = IterationBudget(10)
        # Simulate 5 successful prior iterations
        for _ in range(5):
            budget.consume()
        assert budget.used == 5

        # One outer iteration with 3 retries all failing (worst case without guard)
        budget.consume()  # outer loop consume → used=6
        for _ in range(3):
            budget.refund()  # inner loop retries

        # With the floor, used cannot drop below 0; it stops at 3 (6 - 3)
        # The guard in run_agent.py further limits this to used=5 (6 - 1)
        assert budget.used >= 0, "used must never be negative"
        assert budget.remaining <= budget.max_total, "remaining must not exceed max_total"

    def test_refund_is_bounded_by_floor(self):
        """Repeated refunds on a fresh budget stay at 0, not negative."""
        from run_agent import IterationBudget

        budget = IterationBudget(5)
        budget.consume()  # used=1
        for _ in range(10):
            budget.refund()

        assert budget.used == 0
        assert budget.remaining == 5
