"""Anthropic subscription extra-usage *entitlement* 400 classifies as billing, not format_error.

Native Anthropic answers a Claude Pro/Max subscription that must enable extra usage with HTTP 400
"Third-party apps now draw from your extra usage. To continue, enable extra usage in your account
settings." As ``format_error`` the user saw an opaque failure and the credential pool never rotated.
The rolling-cap / long-context tier gate ("extra usage" + "long context") must stay
``long_context_tier``, and the #82154 "out of extra usage" body keeps its unverified marker.
"""

import pytest

from agent.error_classifier import FailoverReason, classify_api_error

_THIRD_PARTY_GATE = (
    "Third-party apps now draw from your extra usage. To continue, "
    "enable extra usage in your account settings."
)


class _Resp:
    def __init__(self, body, status):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class _APIError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
        self.status_code = status
        self.body = body
        self.response = _Resp(body, status) if status is not None else None


def _classify(message, status=400):
    return classify_api_error(
        _APIError(message, status), provider="anthropic", model="claude-opus-4-8",
        approx_tokens=4000, context_length=200000, num_messages=10,
    )


@pytest.mark.parametrize("status", [400, None])
@pytest.mark.parametrize("message", [
    _THIRD_PARTY_GATE,
    "Third-party apps now draw from extra usage.",
    "This request would exceed your extra usage allowance.",
    "Please enable extra usage in your account settings to continue.",
])
def test_extra_usage_entitlement_gate_is_billing(message, status):
    result = _classify(message, status)
    assert result.reason == FailoverReason.billing
    assert result.retryable is False
    assert result.should_rotate_credential is True
    assert result.should_fallback is True
    assert result.billing_unverified is False


def test_long_context_tier_gate_is_not_entitlement():
    result = _classify("Extra usage is required for long context requests over 200k tokens", status=429)
    assert result.reason == FailoverReason.long_context_tier
    assert result.should_compress is True


def test_entitlement_phrase_naming_long_context_is_not_billing():
    """The entitlement rule declines; the message falls through to the rules it hit before."""
    result = _classify("Enable extra usage for long context requests.", status=400)
    assert result.reason == FailoverReason.format_error


def test_out_of_extra_usage_keeps_unverified_billing_marker():
    result = _classify("You're out of extra usage. Add more at claude.ai/settings/usage and keep going.")
    assert result.reason == FailoverReason.billing
    assert result.billing_unverified is True


def test_unrelated_400_stays_format_error():
    assert _classify("messages.0.content: field required").reason == FailoverReason.format_error
