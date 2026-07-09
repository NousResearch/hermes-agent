"""
Regression tests for issue #60846 - 429 rate-limit error envelopes that
contain "Provider authentication failed" prefix must be classified as
rate-limit, not auth.

The bug: when openai-codex (or any OAuth-pool credential) returns 429
for rate-limit, the gateway wraps the error envelope into text like
"Provider authentication failed: Rate limit exceeded (HTTP 429)".
The current regex dispatch in `_gateway_provider_error_reply` checks
the auth pattern first, so the 429 path is unreachable for these
real-world envelopes. Users get told to re-auth when the real problem
is rate-limiting.

Fix: reorder the dispatch so rate-limit classification runs first.
The 401/invalid-key cases still match the auth pattern and are reached
after the rate-limit return.

These tests exercise the public surface (_sanitize_gateway_final_response)
because the helper itself is private and not exported, but the public
function is the entry point that user-facing chat surfaces hit.
"""

from __future__ import annotations

import pytest

from gateway.config import Platform
from gateway.run import _sanitize_gateway_final_response


RATE_LIMIT_ENVELOPES_UNDER_AUTH_PREFIX = [
    "Provider authentication failed: Rate limit exceeded (HTTP 429)",
    "Provider authentication failed: You exceeded your quota (HTTP 429)",
    "Provider authentication failed: Usage limit reached (HTTP 429)",
    "Error code: 429 - Provider authentication failed: quota exhausted",
]

TRUE_AUTH_ENVELOPES = [
    "Provider authentication failed: invalid api key (HTTP 401)",
    "Provider authentication failed: incorrect api key (HTTP 401)",
    "Provider authentication failed: 401 Unauthorized",
]


@pytest.mark.parametrize("raw", RATE_LIMIT_ENVELOPES_UNDER_AUTH_PREFIX)
def test_429_under_auth_prefix_classifies_as_rate_limit(raw):
    """The bug: 429 envelopes prefixed with 'Provider authentication
    failed' currently classify as auth-failed. After the fix they should
    classify as rate-limit.
    """
    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, raw)
    assert "rate" in sanitized.lower() and "limit" in sanitized.lower(), (
        f"429 envelope misclassified as auth; expected rate-limit message; "
        f"got {sanitized!r} for input {raw!r}. Issue #60846: 429 envelopes "
        f"prefixed with 'Provider authentication failed' are misclassified "
        f"because the auth regex matches first."
    )
    assert "authentication" not in sanitized.lower() or "rate" in sanitized.lower(), (
        f"got the auth-failed message instead of the rate-limit message: "
        f"{sanitized!r}"
    )


@pytest.mark.parametrize("raw", TRUE_AUTH_ENVELOPES)
def test_true_auth_failures_still_classify_as_auth(raw):
    """Regression guard: real auth failures must still surface the
    auth-failed message after the rate-limit check is moved to first
    position. If the fix accidentally breaks auth detection, this fails.
    """
    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, raw)
    assert "authentication" in sanitized.lower(), (
        f"true auth failure not classified as auth; got {sanitized!r} "
        f"for input {raw!r}. The reorder fix should still surface this "
        f"as 'Provider authentication failed' (or equivalent auth message)."
    )


def test_reorder_does_not_cause_classification_drift_on_other_categories():
    """Regression guard: provider-policy envelopes (e.g. safety violations)
    should still classify as policy-rejected, not as rate-limit.
    """
    raw = "Request was rejected: safety policy violation"
    sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, raw)
    # The provider-policy branch is preserved. We don't pin the exact
    # message text (it may include warnings, slugs, etc.); we just check
    # it does NOT contain the rate-limit emoji/marker.
    assert "⏱" not in sanitized, (
        f"provider-policy envelope classified as rate-limit; got {sanitized!r}"
    )
    # And it's not the auth-failed message either.
    assert "authentication" not in sanitized.lower() or "policy" in sanitized.lower(), (
        f"provider-policy envelope misclassified; got {sanitized!r}"
    )
