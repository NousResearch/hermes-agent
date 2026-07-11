"""Regression coverage for transport-error fallback gating in the conversation loop."""

from unittest.mock import MagicMock

from agent.conversation_loop import _should_attempt_eager_error_fallback
from agent.error_classifier import FailoverReason


def _recoverable_rate_limit_pool():
    """Pool shape where rate-limit recovery should suppress rate-limit fallback."""
    pool = MagicMock()
    pool.has_available.return_value = True
    pool.entries.return_value = [MagicMock(), MagicMock()]
    return pool


def test_timeout_transport_failure_waits_one_retry_before_fallback():
    assert (
        _should_attempt_eager_error_fallback(
            FailoverReason.timeout,
            retry_count=1,
            credential_pool=_recoverable_rate_limit_pool(),
        )
        is False
    )


def test_timeout_transport_failure_fallback_after_retry_threshold_even_if_pool_can_recover_rate_limit():
    assert (
        _should_attempt_eager_error_fallback(
            FailoverReason.timeout,
            retry_count=2,
            credential_pool=_recoverable_rate_limit_pool(),
        )
        is True
    )


def test_overloaded_transport_failure_fallback_after_retry_threshold_even_if_pool_can_recover_rate_limit():
    assert (
        _should_attempt_eager_error_fallback(
            FailoverReason.overloaded,
            retry_count=2,
            credential_pool=_recoverable_rate_limit_pool(),
        )
        is True
    )


def test_rate_limit_fallback_still_respects_recoverable_credential_pool():
    assert (
        _should_attempt_eager_error_fallback(
            FailoverReason.rate_limit,
            retry_count=1,
            credential_pool=_recoverable_rate_limit_pool(),
        )
        is False
    )
