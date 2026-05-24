"""Regression guard for #31273: HTTP 402 (Payment Required) must abort
immediately rather than burn ``agent.api_max_retries`` more 402 charges
against an already-depleted credit balance.

The conversation loop's is_client_error predicate has an exclusion set of
``FailoverReason`` values that bypass the non-retryable abort path. Before
this fix, ``FailoverReason.billing`` was in that exclusion set, which meant
a single-credential OpenRouter pool with no fallback chain configured would
fall through to ``while retry_count < max_retries`` and retry the same 402
three times by default — exactly the runaway-token-spend behavior reported
in #31273.

This test pins the invariant: billing must NOT be in
``_NON_CLIENT_ERROR_REASONS``, so the retry loop classifies a 402 billing
error as a client error and aborts after the usual pool-rotation and
eager-fallback recovery paths run upstream.
"""
from __future__ import annotations

from agent.conversation_loop import _NON_CLIENT_ERROR_REASONS
from agent.error_classifier import (
    ClassifiedError,
    FailoverReason,
    classify_api_error,
)


class _APIError(Exception):
    """Minimal exception that mimics an SDK HTTP error with a status code."""

    def __init__(self, message: str, status_code: int, body: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


def _is_client_error(classified: ClassifiedError) -> bool:
    """Mirror of conversation_loop.py's is_client_error predicate.

    Kept in lock-step with the source. Local-validation branch is not
    relevant for an SDK-raised HTTP 402 (which is an Exception, not a
    ValueError/TypeError), so this mirror only tracks the classifier
    branch — which is the branch the fix touches.
    """
    return (
        not classified.retryable
        and not classified.should_compress
        and classified.reason not in _NON_CLIENT_ERROR_REASONS
    )


class TestBillingNotInExclusionSet:
    """The set itself must not list billing."""

    def test_billing_not_in_non_client_error_reasons(self):
        assert FailoverReason.billing not in _NON_CLIENT_ERROR_REASONS

    def test_retryable_reasons_remain_in_exclusion_set(self):
        # Belt-and-suspenders: rate_limit and overloaded are retryable=True
        # and their own special-case paths must run before is_client_error.
        # Keep them in the set so a future refactor that flips retryable to
        # False on either (unlikely but possible) doesn't accidentally abort.
        assert FailoverReason.rate_limit in _NON_CLIENT_ERROR_REASONS
        assert FailoverReason.overloaded in _NON_CLIENT_ERROR_REASONS

    def test_compression_reasons_remain_in_exclusion_set(self):
        # context_overflow and payload_too_large set should_compress=True;
        # the compression path must run instead of the abort path.
        assert FailoverReason.context_overflow in _NON_CLIENT_ERROR_REASONS
        assert FailoverReason.payload_too_large in _NON_CLIENT_ERROR_REASONS


class TestPlain402AbortsImmediately:
    """End-to-end through the classifier + predicate: a 402 with a billing
    body classifies as billing and falls into the client-error abort path."""

    def test_402_payment_required_classifies_as_billing(self):
        err = _APIError("Payment Required", status_code=402)
        classified = classify_api_error(err, provider="openrouter")
        assert classified.reason == FailoverReason.billing
        assert classified.retryable is False

    def test_402_insufficient_credits_is_client_error(self):
        # Real OpenRouter 402 body — depleted balance.
        err = _APIError(
            "Insufficient credits",
            status_code=402,
            body={"error": {"message": "Insufficient credits. Top up at openrouter.ai/credits"}},
        )
        classified = classify_api_error(err, provider="openrouter")
        assert classified.reason == FailoverReason.billing
        assert _is_client_error(classified), (
            "402 billing must classify as a client error so the retry loop "
            "aborts instead of burning api_max_retries more 402 charges."
        )

    def test_402_transient_usage_limit_is_not_client_error(self):
        # 402 with "try again" signal is rate_limit (retryable), not billing.
        # rate_limit is in the exclusion set, so is_client_error must stay False.
        err = _APIError(
            "Usage limit exceeded, try again in 5 minutes",
            status_code=402,
        )
        classified = classify_api_error(err, provider="openrouter")
        assert classified.reason == FailoverReason.rate_limit
        assert not _is_client_error(classified)

    def test_400_billing_reason_is_also_client_error(self):
        # Some providers (Anthropic "out of extra usage") return HTTP 400
        # but the classifier maps it to billing. Same invariant must hold:
        # if pool rotation + eager fallback both fail, abort, don't retry.
        synthetic = ClassifiedError(
            reason=FailoverReason.billing,
            status_code=400,
            retryable=False,
        )
        assert _is_client_error(synthetic)
