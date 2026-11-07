"""Google reuses HTTP 429 + ``RESOURCE_EXHAUSTED`` for an account spending cap and a per-minute throttle."""
import time
from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class MockAPIError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.response = SimpleNamespace(headers={})


@pytest.mark.parametrize("reason", ["QUOTA_EXHAUSTED", "INSUFFICIENT_G1_CREDITS_BALANCE", None])
def test_google_429_billing_reason_is_terminal_billing(reason):
    """Google reuses 429 + RESOURCE_EXHAUSTED for an account ceiling; only
    ``google.rpc.ErrorInfo.reason`` says so. The native route's message
    ("Gemini HTTP 429 (RESOURCE_EXHAUSTED): ...") is itself a rate-limit
    phrase, so without the structured read every billing 429 burned the
    retry budget as a throttle (ported from can1357/oh-my-pi#13092). The cap also
    arrives with EMPTY details (``reason=None``): the "spending cap" prose decides then."""
    message = "Your project has exceeded its monthly spending cap."
    details = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason}] if reason else []
    body = {"error": {"code": 429, "message": message, "status": "RESOURCE_EXHAUSTED", "details": details}}
    result = classify_api_error(
        MockAPIError(f"Gemini HTTP 429 (RESOURCE_EXHAUSTED): {message}", status_code=429, body=body),
        provider="gemini",
    )
    assert result.reason == FailoverReason.billing
    assert result.retryable is False
    assert result.should_rotate_credential is True

def test_google_429_throttle_reason_stays_rate_limit_with_retry_info():
    """The other direction: a per-minute throttle whose prose ("Quota exceeded
    ... per_minute") reads as a quota wall with no transient signal must not
    bench a healthy key as billing; RetryInfo.retryDelay feeds ``reset_at``."""
    body = {"error": {"code": 429,
                      "message": "Quota exceeded for aiplatform.googleapis.com/generate_content_requests_per_minute.",
                      "status": "RESOURCE_EXHAUSTED",
                      "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "RATE_LIMIT_EXCEEDED"},
                                  {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "27s"}]}}
    result = classify_api_error(MockAPIError(body["error"]["message"], status_code=429, body=body), provider="gemini")
    assert result.reason == FailoverReason.rate_limit
    assert result.retryable is not False
    assert 20 < result.error_context["reset_at"] - time.time() <= 27


@pytest.mark.parametrize(("quota_id", "value", "error_info", "unwrapped", "expected"), [
    ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "5", False, False, FailoverReason.rate_limit),
    ("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "100", False, False, FailoverReason.billing),
    ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "0", False, False, FailoverReason.billing),
    ("GenerateRequestsPerDayPerProjectPerModel", "1000", True, False, FailoverReason.billing),
    # OpenAI-compatible route: the SDK's ``.body`` is the already-unwrapped ``error`` object.
    ("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "100", False, True, FailoverReason.billing),
])
def test_google_429_quota_failure_window_decides(quota_id, value, error_info, unwrapped, expected):
    """AI Studio's free-tier 429s carry only ``QuotaFailure`` + ``RetryInfo`` (no ErrorInfo), and
    always a short retryDelay. A daily window or a zero quota (no entitlement to the model) replays
    until tomorrow or forever, so it is billing even beside ErrorInfo ``RATE_LIMIT_EXCEEDED``;
    a per-minute one stays a throttle with RetryInfo as ``reset_at`` (ported from can1357/oh-my-pi#17127)."""
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": quota_id, "quotaValue": value,
                                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests"}]},
               {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "33s"}]
    if error_info:
        details.insert(0, {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "RATE_LIMIT_EXCEEDED"})
    message = f"You exceeded your current quota, please check your plan and billing details. limit: {value}. Please retry in 33.6s."
    body = {"error": {"code": 429, "message": message, "status": "RESOURCE_EXHAUSTED", "details": details}}
    result = classify_api_error(
        MockAPIError(f"Gemini HTTP 429 (RESOURCE_EXHAUSTED): {message}", status_code=429,
                     body=body["error"] if unwrapped else body), provider="gemini",
    )
    assert result.reason == expected
    if expected == FailoverReason.rate_limit:
        assert 25 < result.error_context["reset_at"] - time.time() <= 33
