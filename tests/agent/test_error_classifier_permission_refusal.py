"""A 403 that calls itself a permission refusal is not a bill, whatever its ``code`` says.

Relays reuse a billing-shaped code for their default 403: OmniRoute 3.8.49 answers a combo
allow-list denial with ``type: permission_error`` and ``code: insufficient_quota``, which is that
release's default 403 code and not a claim about a balance. The serialized body reaches the
message scan, so the word alone classified the refusal as ``billing`` — which benched credentials
that were healthy (``should_rotate_credential``) and told the user to top up an account with no
billing relationship to the gateway (#146649). The structured ``type`` is the reliable half of
that body, and the sibling case already honours it: a 403 whose body self-identifies as a server
error is read as one.

"""

import json

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class _APIError(Exception):
    """Simulates an OpenAI SDK status error carrying an already-parsed JSON body."""

    def __init__(self, status_code, body):
        super().__init__(f"Error code: {status_code} - {json.dumps(body)}")
        self.status_code = status_code
        self.body = body


def _combo_denial(type_name, code, message='Combo "deepseek-flash-alpha" is not allowed for this API key'):
    return {"error": {"message": message, "type": type_name, "code": code}}


def test_permission_refusal_wearing_a_billing_code_is_policy_blocked():
    result = classify_api_error(
        _APIError(403, _combo_denial("permission_error", "insufficient_quota")),
        provider="custom",
        model="deepseek-flash-alpha",
        base_url="http://localhost:20128/v1",
    )

    assert result.reason == FailoverReason.provider_policy_blocked
    # A permission refusal says nothing about the credential, so the pool must keep serving it.
    assert result.should_rotate_credential is False
    assert result.should_fallback is True
    assert result.retryable is False
    assert result.billing_unverified is False


def test_permission_denied_type_is_policy_blocked_too():
    """OmniRoute ≥3.8.52 moved the code but kept ``type: permission_error``; same verdict."""
    result = classify_api_error(
        _APIError(403, _combo_denial("permission_denied", "permission_denied")),
        provider="custom",
    )

    assert result.reason == FailoverReason.provider_policy_blocked
    assert result.should_rotate_credential is False


@pytest.mark.parametrize("body, reason", [
    # A real quota wall keeps reading as billing — including when the relay mislabels its type.
    (
        _combo_denial(
            "insufficient_quota",
            "insufficient_quota",
            "You exceeded your current quota, please check your plan and billing details",
        ),
        FailoverReason.billing,
    ),
    (
        _combo_denial("permission_error", "budget_exceeded", "Budget limit exceeded (monthly limit)"),
        FailoverReason.billing,
    ),
    # Without a structured permission claim the 403 keeps its previous reading.
    (
        _combo_denial("invalid_request_error", "forbidden", "Forbidden"),
        FailoverReason.auth,
    ),
])
def test_403_bodies_that_are_not_bare_permission_refusals_keep_their_verdict(body, reason):
    result = classify_api_error(_APIError(403, body), provider="custom")

    assert result.reason == reason


def test_provider_scoped_spending_limit_code_outranks_the_type_label():
    """xAI's Grok credit exhaustion arrives as 403 with its own code; a permission ``type`` on it
    does not make it an access-control block (#146649 stays scoped to relays with no such code)."""
    result = classify_api_error(
        _APIError(403, _combo_denial("permission_error", "personal-team-blocked:spending-limit",
                                     "Access denied for this account")),
        provider="xai-oauth",
    )

    assert result.reason == FailoverReason.billing
    assert result.should_rotate_credential is True
