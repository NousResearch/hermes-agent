"""Regression guard for the proxy 403-throttle retry-delay wiring.

Background (desktop.log abort):
A local model router proxy (provider ``custom:9router-proxy`` at
localhost:20128) signalled a SOFT THROTTLE as ``HTTP 403 (reset after 2m)``
with the delay embedded in the body — NOT a standard HTTP Retry-After header
and NOT a real credential rejection. The error classifier now maps that to a
retryable ``rate_limit`` carrying ``retry_after_seconds`` (see CHANGE 1 in
``agent/error_classifier.py`` + tests/agent/test_error_classifier.py).

This file guards CHANGE 2: ``agent/conversation_loop.py`` must prefer the
classifier's body-parsed ``retry_after_seconds`` when no Retry-After *header*
is present, so the backoff waits the advertised window instead of a too-short
exponential delay (3 attempts × max 60s never spans a 120s window).

We mirror the exact ``_retry_after`` resolution snippet from the loop, in the
same spirit as tests/run_agent/test_31273_402_not_retried.py (which mirrors the
``is_client_error`` predicate). If you change one, change both.
"""
from __future__ import annotations

from agent.error_classifier import ClassifiedError, FailoverReason


def _resolve_retry_after(*, header_value, classified):
    """Exact shape of conversation_loop.py's _retry_after resolution.

    is_rate_limited is assumed True here (the only branch that reads it).
    Returns the resolved wait override in seconds, or None to use backoff.
    """
    _retry_after = None
    # Header path (pre-existing behavior).
    if header_value is not None:
        try:
            _retry_after = min(float(header_value), 120)
        except (TypeError, ValueError):
            pass
    # Body-parsed path (CHANGE 2).
    if _retry_after is None:
        _classified_ra = getattr(classified, "retry_after_seconds", None)
        if _classified_ra:
            try:
                _retry_after = min(float(_classified_ra), 120)
            except (TypeError, ValueError):
                pass
    return _retry_after


class TestRetryAfterResolution:
    def _throttle_403(self, seconds):
        return ClassifiedError(
            reason=FailoverReason.rate_limit,
            status_code=403,
            retryable=True,
            retry_after_seconds=seconds,
        )

    def test_body_delay_used_when_no_header(self):
        """The desktop.log case: no Retry-After header, body says 2m."""
        classified = self._throttle_403(120.0)
        assert _resolve_retry_after(header_value=None, classified=classified) == 120.0

    def test_header_takes_precedence_over_body(self):
        """A real Retry-After header still wins (pre-existing behavior)."""
        classified = self._throttle_403(120.0)
        assert _resolve_retry_after(header_value="7", classified=classified) == 7.0

    def test_body_delay_capped_at_120(self):
        classified = self._throttle_403(600.0)
        assert _resolve_retry_after(header_value=None, classified=classified) == 120.0

    def test_no_header_no_body_falls_through_to_backoff(self):
        """No header and no classified delay → None → loop uses jittered backoff."""
        classified = ClassifiedError(
            reason=FailoverReason.rate_limit, status_code=429, retryable=True
        )
        assert _resolve_retry_after(header_value=None, classified=classified) is None

    def test_garbage_header_falls_back_to_body(self):
        classified = self._throttle_403(30.0)
        assert _resolve_retry_after(header_value="not-a-number", classified=classified) == 30.0


class TestThrottle403StaysRetryable:
    """The classified throttle 403 must NOT hit the client-error abort path.

    Mirrors the is_client_error predicate from conversation_loop.py — a
    rate_limit reason is explicitly excluded, so the loop retries instead of
    aborting (which is what happened before CHANGE 1).
    """

    def _is_client_error(self, classified):
        return (
            not classified.retryable
            and not classified.should_compress
            and classified.reason not in {
                FailoverReason.rate_limit,
                FailoverReason.overloaded,
                FailoverReason.context_overflow,
                FailoverReason.payload_too_large,
                FailoverReason.long_context_tier,
                FailoverReason.thinking_signature,
            }
        )

    def test_throttle_403_does_not_abort(self):
        classified = ClassifiedError(
            reason=FailoverReason.rate_limit,
            status_code=403,
            retryable=True,
            retry_after_seconds=120.0,
        )
        assert not self._is_client_error(classified)

    def test_real_auth_403_still_aborts(self):
        classified = ClassifiedError(
            reason=FailoverReason.auth,
            status_code=403,
            retryable=False,
        )
        assert self._is_client_error(classified)
