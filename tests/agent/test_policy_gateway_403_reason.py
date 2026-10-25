"""A structured non-auth 403 body must not be reported as a rejected API key (#125058).

A governance/policy gateway in front of a custom endpoint answers 403 with a
machine-readable error ``type``/``code`` plus its own ``reason`` (policy ceiling,
pending human approval). ``_status_403`` only special-cased billing wording and
WAF/CDN markers, so these fell through to the auth verdict and the user was told
their key was rejected when it was valid. These tests pin:

  1. Both bodies from the report classify as ``provider_policy_blocked`` — not
     auth, no credential rotation — and the gateway's reason becomes the message.
  2. Auth-typed codes (``permission_error`` etc.), untyped/plain 403s, and
     structured billing codes keep today's classification.
  3. The one-line summary reads ``error.reason``/``error.detail`` so the
     "Provider said:" line carries the policy explanation.
"""
import pytest

from agent.api_error_summary import ApiErrorSummaryMixin
from agent.error_classifier import FailoverReason, classify_api_error


class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError (status_code + parsed body)."""

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


DENIED = {
    "type": "wardryx_denied",
    "reason": 'estimated cost $0.02 exceeds policy "deny-beta" hard ceiling; no approval can authorize this',
    "retryable": False,
    "run_id": "run-1",
    "policy_version": "v3",
}
APPROVAL_PENDING = {
    "type": "cost_approval_pending",
    "approval_id": "ap_1",
    "approval_token_required": True,
    "detail": "resubmit this request with header x-fuse-approval-token after approval",
    "reason": "estimated cost $0.02 exceeds policy threshold; human approval required",
    "retryable": False,
}


def _classify(body, provider="custom"):
    err = MockAPIError(f"Error code: 403 - {body}", status_code=403, body=body)
    return classify_api_error(err, provider=provider, model="test-model")


@pytest.mark.parametrize("body", [
    {"error": DENIED},
    {"error": APPROVAL_PENDING},
], ids=["wrapped-denied", "wrapped-approval"])
def test_structured_non_auth_403_is_policy_blocked_not_bad_key(body):
    result = _classify(body)
    assert result.reason is FailoverReason.provider_policy_blocked
    assert result.is_auth is False
    # The key never failed: no refresh, no rotation — the verdict must not bench it.
    assert result.should_rotate_credential is False
    assert result.retryable is False
    # The gateway's own explanation becomes the surfaced message.
    assert DENIED["reason"][:60] in result.message or APPROVAL_PENDING["reason"][:60] in result.message


def test_authorization_substring_in_policy_code_is_not_auth_classification():
    body = {"error": {"type": "policy_authorization_required", "reason": "policy needs a named approver"}}
    assert _classify(body).reason is FailoverReason.provider_policy_blocked


@pytest.mark.parametrize("body", [
    # Real permission refusals word themselves with auth-shaped types.
    {"error": {"type": "permission_error", "message": "you lack permission for this model"}},

    {"error": {"code": "invalid_api_key", "message": "Incorrect API key provided"}},
    {"error": {"type": "unauthenticated_waf", "message": "user identity expired"}},
    # No structured code: today's wording-based classification must stand.
    {"message": "You tried to access something that you don't have permissions for."},
])
def test_auth_typed_or_untyped_403_keeps_auth_classification(body):
    assert _classify(body).reason is FailoverReason.auth


def test_typed_403_without_explanatory_text_stays_auth():
    """A code with nothing to surface must not flip the verdict — the auth
    guidance is still the most useful default for a bare typed 403."""
    assert _classify({"error": {"type": "wardryx_denied"}}).reason is FailoverReason.auth


def test_transient_and_waf_403_handlers_keep_precedence():
    assert _classify({"error": {"type": "upstream_unavailable", "code": "upstream_unavailable",
                                "message": "retry later"}}).reason is FailoverReason.overloaded
    waf = MockAPIError("Error code: 403 - Your request was blocked.", status_code=403)
    assert classify_api_error(waf, provider="custom").reason is FailoverReason.upstream_blocked
    # A billing-worded typed 403 is account exhaustion, not policy: the decisive
    # billing branch keeps precedence ("exceeded your current quota" is in _BILLING_PATTERNS).
    assert _classify({"error": {"type": "quota_check", "code": "insufficient_quota",
                                "message": "You exceeded your current quota"}}).reason is FailoverReason.billing


def test_summary_surfaces_reason_and_detail_without_message():
    """``_summarize_api_error`` read only ``message``; the reported bodies carry
    ``reason``/``detail`` — without this fix the "Provider said:" line was empty."""
    for body in ({"error": DENIED}, {"error": APPROVAL_PENDING}):
        err = MockAPIError("Error code: 403 - policy", status_code=403, body=body)
        summary = ApiErrorSummaryMixin._summarize_api_error(err)
        assert summary.startswith("HTTP 403:")
        assert "estimated cost $0.02" in summary


def test_summary_redacts_sensitive_reason_text():
    err = MockAPIError(
        "Error code: 403 - policy",
        status_code=403,
        body={"error": {"type": "wardryx_denied", "reason": "key sk-test-secret and http://10.0.0.4/internal"}},
    )
    summary = ApiErrorSummaryMixin._summarize_api_error(err)
    assert "sk-test-secret" not in summary
