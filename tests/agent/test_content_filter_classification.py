"""Regression tests: the z-ai content-filter 403 must classify as
``content_policy_blocked`` (non-retryable, safety-refusal guidance), not
``auth`` (which misroutes recovery into key/re-auth advice).

Incident 2026-09-12: "Request blocked by content filter: [SECRET:…]" 403s
were classified as auth-fallback because ``_CONTENT_POLICY_BLOCKED_PATTERNS``
matched only the underscore variant ``content_filter``; the space variant is
deliberately excluded (echoed-config false positives), so the full sentence
is matched instead.
"""

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class _FakeAPIError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        self.body = body or {}


def _classify(msg, body_msg=None):
    return classify_api_error(
        error=_FakeAPIError(msg, 403, {"error": {"message": body_msg or msg, "code": 403}}),
        provider="openrouter",
        model="z-ai/glm-5.3",
        approx_tokens=5000,
        context_length=32000,
        num_messages=20,
    )


def test_zai_content_filter_403_is_content_policy_blocked():
    v = _classify("Error code: 403 - Request blocked by content filter: [SECRET:google-oauth-client-secret]")
    assert v.reason == FailoverReason.content_policy_blocked
    assert v.retryable is False


def test_bare_content_filter_phrase_is_not_blocked():
    # Polarity: echoed config mentioning "content filter" must stay auth-classified
    # (the space variant of the bare token is deliberately unmatched).
    v = _classify("Error code: 403 - config uses content filter mode")
    assert v.reason != FailoverReason.content_policy_blocked


def test_billing_403_still_billing():
    v = _classify("Error code: 403 - Key limit exceeded")
    assert v.reason == FailoverReason.billing
