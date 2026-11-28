"""Anthropic Console spend caps are billing walls, not malformed requests (#80553).

The account cap and the workspace cap both arrive as HTTP 400 ``invalid_request_error``;
classified as ``format_error`` the key never rotated and fallback never engaged.
"""
import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class _APIError(Exception):
    def __init__(self, message, status_code, body):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@pytest.mark.parametrize("scope", ["", "workspace "])
def test_400_anthropic_console_spend_cap_is_billing(scope):
    msg = f"You have reached your specified {scope}API usage limits. You will regain access on 2026-09-01"
    error = _APIError(msg, 400, {"error": {"type": "invalid_request_error", "message": msg}})
    result = classify_api_error(error, provider="anthropic", model="claude-opus-5")
    assert result.reason == FailoverReason.billing
    assert result.retryable is False
    assert result.should_fallback is True and result.should_rotate_credential is True
