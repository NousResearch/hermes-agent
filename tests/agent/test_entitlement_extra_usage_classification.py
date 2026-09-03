"""Anthropic subscription *entitlement* 400 ("extra usage" / "third-party apps
now draw from your extra usage") must classify as BILLING, not format_error.

Repro (forensics in this repo's own incident history): when a Claude
subscription must enable extra usage, native Anthropic returns HTTP 400 with a
body like "Third-party apps now draw from your extra usage. To continue, enable
extra usage in your account settings." The generic 400 path classified this as
``format_error`` — so the user got an opaque "format error" instead of the
actionable billing/entitlement guidance, and no credential rotation was
triggered (``should_rotate_credential`` stayed False).

These are behavior-contract tests: the entitlement signal => billing + rotate +
fallback, while the *rolling-cap* 429 ("extra usage" + "long context") must stay
``long_context_tier`` (retryable, NOT billing). Guards both the fix and the
no-collision invariant.
"""

from agent.error_classifier import FailoverReason, classify_api_error


class _FakeResp:
    def __init__(self, body, status=400):
        self.status_code = status
        self._b = body

    def json(self):
        return self._b


class _FakeErr(Exception):
    def __init__(self, msg, body, status=400):
        super().__init__(msg)
        self.status_code = status
        self.body = body
        self.response = _FakeResp(body, status)


def _classify(msg, status=400):
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": msg}}
    return classify_api_error(
        _FakeErr(msg, body, status),
        provider="anthropic",
        model="claude-opus-4-8",
        approx_tokens=4000,
        context_length=200000,
        num_messages=10,
    )


class TestEntitlementExtraUsageIsBilling:
    def test_third_party_draw_from_extra_usage_is_billing(self):
        r = _classify(
            "Third-party apps now draw from your extra usage. To continue, "
            "enable extra usage in your account settings."
        )
        assert r.reason == FailoverReason.billing
        assert r.should_fallback is True
        assert r.should_rotate_credential is True

    def test_exceed_extra_usage_allowance_is_billing(self):
        r = _classify("This request would exceed your extra usage allowance.")
        assert r.reason == FailoverReason.billing
        assert r.should_rotate_credential is True

    def test_enable_extra_usage_phrasing_is_billing(self):
        r = _classify("Please enable extra usage in your account settings to continue.")
        assert r.reason == FailoverReason.billing


class TestNoCollisionWithLongContextTier:
    """The 429 rolling-cap / long-context tier gate must remain untouched —
    it is retryable and NOT billing."""

    def test_429_extra_usage_long_context_stays_tier_gate(self):
        r = _classify(
            "Request requires extra usage for the long context window.",
            status=429,
        )
        assert r.reason == FailoverReason.long_context_tier
        assert r.reason != FailoverReason.billing

    def test_real_billing_credit_balance_still_billing(self):
        r = _classify("Your credit balance is too low to access the Anthropic API.")
        assert r.reason == FailoverReason.billing
