"""Regression tests for eager fallback on plain server_error (500/502).

_is_eager_fallback_transport_reason() decides whether an error reason
qualifies for the main retry loop's eager-fallback gate (one retry, then
fail over to fallback_providers) rather than only reaching a configured
fallback chain via the separate max-retry-exhaustion attempt later in the
same loop. server_error (500/502) joining timeout/overloaded here means
that failover happens sooner — after 2 retries — instead of waiting out
the full retry budget on a primary that's already failing.
"""
import inspect

from agent import conversation_loop
from agent.conversation_loop import _is_eager_fallback_transport_reason
from agent.error_classifier import FailoverReason


def test_server_error_is_eager_fallback_eligible():
    assert _is_eager_fallback_transport_reason(FailoverReason.server_error) is True


def test_overloaded_is_eager_fallback_eligible():
    assert _is_eager_fallback_transport_reason(FailoverReason.overloaded) is True


def test_timeout_is_eager_fallback_eligible():
    assert _is_eager_fallback_transport_reason(FailoverReason.timeout) is True


def test_rate_limit_is_not_transport_eligible():
    # rate_limit takes the separate, immediate is_rate_limited path in the
    # loop's _should_fallback gate - it must not double-count here.
    assert _is_eager_fallback_transport_reason(FailoverReason.rate_limit) is False


def test_billing_is_not_transport_eligible():
    assert _is_eager_fallback_transport_reason(FailoverReason.billing) is False


def test_unknown_is_not_transport_eligible():
    assert _is_eager_fallback_transport_reason(FailoverReason.unknown) is False


def test_should_fallback_activates_at_retry_count_two():
    """Reproduces the loop's _should_fallback expression directly: eager
    fallback for a transport/server-error reason requires retry_count >= 2,
    matching the existing overloaded/timeout threshold - not retry 0 or 1."""
    is_rate_limited = False  # server_error never sets this
    for retry_count in (0, 1):
        _is_transport_failure = _is_eager_fallback_transport_reason(FailoverReason.server_error)
        should_fallback = is_rate_limited or (_is_transport_failure and retry_count >= 2)
        assert should_fallback is False, f"retry_count={retry_count} must not eager-fallback yet"

    _is_transport_failure = _is_eager_fallback_transport_reason(FailoverReason.server_error)
    should_fallback = is_rate_limited or (_is_transport_failure and 2 >= 2)
    assert should_fallback is True


def test_run_conversation_uses_extracted_reason_helper():
    """The loop's eager-fallback gate must actually call the extracted,
    unit-tested helper rather than an inline literal set that could drift
    out of sync with it (e.g. a future edit adding a reason to one but not
    the other)."""
    source = inspect.getsource(conversation_loop.run_conversation)

    assert "_is_transport_failure = _is_eager_fallback_transport_reason(classified.reason)" in source
    assert "retry_count >= 2" in source
