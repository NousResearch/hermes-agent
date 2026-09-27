"""Structured governance refusals must not invalidate a working credential (#125058)."""
import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_failure_copy import nonretryable_copy


DENIED = {
    "type": "wardryx_denied",
    "reason": 'estimated cost $0.02 exceeds policy "deny-beta" hard ceiling; no approval can authorize this',
    "retryable": False,
}
APPROVAL = {
    "type": "approval_pending",
    "approval_token_required": True,
    "detail": "resubmit this request with header x-fuse-approval-token after approval",
    "reason": "estimated cost $0.02 exceeds policy threshold; human approval required",
    "retryable": False,
}


def classify(body, status=403):
    response = httpx.Response(status, request=httpx.Request("POST", "https://policy.example/v1/chat/completions"))
    error = openai.APIStatusError("Forbidden", response=response, body=body)
    return classify_api_error(error, provider="custom", model="test", base_url="https://policy.example/v1")


@pytest.mark.parametrize("error", [DENIED, APPROVAL])
@pytest.mark.parametrize("wrapped", [True, False])
def test_governance_refusal_preserves_key_and_explains_policy(error, wrapped):
    result = classify({"error": error} if wrapped else error)
    assert result.reason == FailoverReason.provider_policy_blocked
    assert not result.is_auth
    assert not result.should_rotate_credential
    assert not result.retryable
    # Do not route around an explicit governance refusal with a different provider.
    assert not result.should_fallback
    assert result.message == error["reason"]
    copy = nonretryable_copy(result, provider="custom", model="test", summary="Forbidden")
    assert error["reason"] in copy
    assert "rejected your API key" not in copy
    assert "switch models" not in copy


@pytest.mark.parametrize("body", [
    {}, {"error": {}}, {"error": "forbidden"},
    {"error": {"type": "unknown_error", "reason": "unexpected failure", "retryable": False}},
    {"error": {"type": "wardryx_denied", "reason": "policy denied", "retryable": True}},
    {"error": {"type": "wardryx_denied", "reason": {"policy": "denied"}, "retryable": False}},
    *[{"error": {**DENIED, "type": code}} for code in
      ("authentication_error", "invalid_api_key", "permission_denied", "unauthorized")],
])
def test_ambiguous_or_auth_errors_keep_existing_auth_fallback(body):
    result = classify(body)
    assert result.reason == FailoverReason.auth
    assert result.should_fallback


def test_401_remains_auth_even_with_policy_envelope():
    assert classify({"error": DENIED}, status=401).is_auth


@pytest.mark.parametrize("message, expected", [
    ("insufficient credits", FailoverReason.billing),
    ("Enable JavaScript and cookies to continue", FailoverReason.upstream_blocked),
])
def test_existing_decisive_403_handlers_keep_precedence(message, expected):
    assert classify({"error": {**DENIED, "message": message}}).reason == expected
