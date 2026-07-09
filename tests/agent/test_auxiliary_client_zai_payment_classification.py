"""Regression tests for Z.AI GLM Coding Plan payment-error misclassification.

Bug
---
``agent.auxiliary_client._is_payment_error()`` over-matches. The keyword
block treats ``"resource exhausted"`` (line in the existing implementation)
and ``"reached your session usage limit"`` (intended for Nous / OpenCode Go
weekly subscription caps) as payment errors.

Z.AI's Coding Plan per-key rolling quotas surface as either:

    1113  "Insufficient balance or no resource package"
    1308  "Usage limit reached for 5 hour"

The first matches ``"no resource"``/``"insufficient balance"`` substrings
that the keyword block flags as payment. The second matches
``"reached your session usage limit"`` even though it is a per-key
rolling window, not wallet billing exhaustion.

When these Z.AI errors are misclassified, ``credential_pool.recover_provider_pool``
takes the wrong branch: instead of marking the single exhausted key as
`STATUS_EXHAUSTED` and rotating to the next key in the round-robin pool,
it triggers a global provider fallback that takes the whole Z.AI pool
offline. Curl tests against the surviving keys return HTTP 200, confirming
they are healthy; the bug is purely in the error-classification layer.

Fix
---
Add an explicit Z.AI exemption block in ``_is_payment_error`` that returns
``False`` early when the error originates from ``api.z.ai`` /
``z.ai/api/coding`` and contains any of the Z.AI Coding Plan signature
patterns. Plain Z.AI billing errors without these signatures (real wallet
depletion) still flow through the normal payment-error path.

These tests pin the fix so a future refactor of ``_is_payment_error``
cannot silently regress Z.AI multi-key pool rotation.

See: Hermes Agent v0.18.x, ``agent/auxiliary_client.py`` line ~2816.
"""

import pytest

from agent.auxiliary_client import _is_payment_error


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _exc(message: str, status_code: int | None = 429) -> Exception:
    """Build an exception with both ``status_code`` and a Z.AI-shaped body.

    Z.AI surfaces quota errors as HTTP 429 with a code field in the body
    (e.g. ``"code": 1113``), so this mirrors the real on-the-wire shape.
    """
    e = Exception(message)
    if status_code is not None:
        e.status_code = status_code
    return e


# --------------------------------------------------------------------------- #
# 1. Exact Z.AI messages from real Coding Plan responses                      #
# --------------------------------------------------------------------------- #


class TestZaiCodingPlanQuotasAreNotPaymentErrors:
    """The four signature Z.AI Coding Plan responses must NOT be payment errors."""

    def test_1113_no_resource_package_is_not_payment(self):
        """1113 "Insufficient balance or no resource package" — pool rotation."""
        exc = _exc(
            "Error code: 1113 - Insufficient balance or no resource package. "
            "Purchase resource package at https://z.ai/subscribe",
            status_code=429,
        )
        assert _is_payment_error(exc) is False

    def test_1308_usage_limit_reached_5_hour_is_not_payment(self):
        """1308 "Usage limit reached for 5 hour" — pool rotation."""
        exc = _exc(
            "Error code: 1308 - Usage limit reached for 5 hour, please try again later.",
            status_code=429,
        )
        assert _is_payment_error(exc) is False

    def test_1113_with_full_url_in_message(self):
        """Same as 1113 but with ``api.z.ai`` URL fully spelled out (curl shape)."""
        exc = _exc(
            "POST https://api.z.ai/api/coding/paas/v4/chat/completions -> 429: "
            '{"error":{"code":"1113","message":"Insufficient balance or no resource package"}}',
            status_code=429,
        )
        assert _is_payment_error(exc) is False

    def test_1308_with_full_url_in_message(self):
        """Same as 1308 but with the ``/api/coding/paas/v4`` endpoint visible."""
        exc = _exc(
            "POST https://api.z.ai/api/coding/paas/v4/chat/completions -> 429: "
            '{"error":{"code":"1308","message":"Usage limit reached for 5 hour"}}',
            status_code=429,
        )
        assert _is_payment_error(exc) is False

    def test_usage_limit_reached_phrase_without_code_field(self):
        """The phrase alone (no code number) should also exempt."""
        exc = _exc(
            "Z.AI Coding Plan: usage limit reached for 5 hour.",
            status_code=429,
        )
        assert _is_payment_error(exc) is False


# --------------------------------------------------------------------------- #
# 2. Negative tests: same shapes from OTHER providers stay payment errors    #
# --------------------------------------------------------------------------- #


class TestOtherProvidersUnaffected:
    """The fix must NOT swallow real payment/quota errors from other providers."""

    def test_vertex_ai_resource_exhausted_still_payment(self):
        """PR #26803 — Vertex AI ``resource exhausted`` is real billing."""
        exc = _exc(
            "RESOURCE_EXHAUSTED: quota exceeded for project my-proj",
            status_code=429,
        )
        assert _is_payment_error(exc) is True

    def test_bedrock_daily_token_limit_still_payment(self):
        """PR #26803 — Bedrock daily token limit is real billing."""
        exc = _exc(
            "Too many tokens per day: 1000000 used, 1000000 limit",
            status_code=429,
        )
        assert _is_payment_error(exc) is True

    def test_openrouter_credits_still_payment(self):
        """OpenRouter credit-exhaustion is still a payment error."""
        exc = _exc(
            "insufficient credits remaining for this request",
            status_code=429,
        )
        assert _is_payment_error(exc) is True

    def test_402_with_zai_body_still_payment(self):
        """If Z.AI ever returns a real bare HTTP 402 it is still a payment error —
        the exemption only fires for the message-driven codes 1113 / 1308."""
        exc = _exc(
            "Payment required for this API key",
            status_code=402,
        )
        assert _is_payment_error(exc) is True

    def test_daily_quota_non_zai_still_payment(self):
        """Generic 'daily quota' phrasing from another provider stays a payment err."""
        exc = _exc(
            "Daily quota of 500 requests reached. api.openai.com",
            status_code=429,
        )
        assert _is_payment_error(exc) is True


# --------------------------------------------------------------------------- #
# 3. Exact reproduction of the bug as reported in the upstream bug report     #
# --------------------------------------------------------------------------- #


class TestBugReportReproduction:
    """These are the verbatim shapes from the upstream bug report. They must pass."""

    def test_1113_keyword_substring_resource_exhausted_no_longer_matches(self):
        """Bug: 'no resource' substring matched 'resource exhausted' (Vertex keyword).

        After the fix, the explicit Z.AI 1113 check short-circuits to False
        before the keyword block runs.
        """
        exc = _exc(
            '{"code":1113,"message":"Insufficient balance or no resource package"}',
            status_code=429,
        )
        assert _is_payment_error(exc) is False

    def test_1308_keyword_substring_session_usage_limit_no_longer_matches(self):
        """Bug: 'reached' substring matched 'reached your session usage limit' (Nous kw).

        After the fix, the explicit Z.AI 1308 check short-circuits to False
        before the keyword block runs.
        """
        exc = _exc(
            '{"code":1308,"message":"Usage limit reached for 5 hour"}',
            status_code=429,
        )
        assert _is_payment_error(exc) is False


# --------------------------------------------------------------------------- #
# 4. Smoke: the pool rotation layer sees the exemption                        #
# --------------------------------------------------------------------------- #


class TestExemptionFeedsPoolRotationPath:
    """The whole point: a 1308 from Z.AI must NOT trigger a global provider
    fallback. We assert the predicate value, not the rotation behavior (the
    latter is integration-tested elsewhere — see Issue #53654 for the
    related Gemini pool-rotation PR), because the predicate is the
    deciding gate; if it returns False the pool rotation path is taken.
    """

    def test_one_zai_exhausted_key_only_that_key_paid_error_block(self):
        """The exemption must be keyed on the message shape, not the provider name
        string — so that *any* Z.AI Coding Plan error that mentions 1113/1308
        gets the rotation-friendly classification, regardless of which key
        in the pool served it.
        """
        exc_1113_key_a = _exc(
            "api.z.ai/api/coding/paas/v4: error code 1113 (key A)",
            status_code=429,
        )
        exc_1308_key_b = _exc(
            "api.z.ai/api/coding/paas/v4: error code 1308 (key B)",
            status_code=429,
        )
        assert _is_payment_error(exc_1113_key_a) is False
        assert _is_payment_error(exc_1308_key_b) is False

    def test_zai_unknown_quota_message_falls_through_to_keyword_block(self):
        """If a Z.AI error message contains none of our exemption patterns but
        happens to look like a real billing error (e.g. 'daily limit' on the
        metered endpoint, not the Coding Plan endpoint), we still trust the
        generic keyword block. We assert this so the fix doesn't
        over-generalize and disable real Z.AI metered-endpoint
        payment errors.
        """
        exc = _exc(
            "api.z.ai/api/paas/v4: you have exceeded your daily limit",
            status_code=429,
        )
        # Should still match via the existing 'daily limit' keyword.
        assert _is_payment_error(exc) is True
