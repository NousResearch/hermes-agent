"""Regression tests for the xAI / X Premium+ usage fetcher.

The Grok subscription's weekly pool is served by the Grok CLI's own chat proxy:
``GET https://cli-chat-proxy.grok.com/v1/billing?format=credits`` with the CLI
identity header ``x-xai-token-auth: xai-grok-cli``.

Two things this pins, both of which cost a wrong answer before:

1. **``format=credits`` is mandatory.** Without the query the endpoint answers the
   METERED-API billing shape (``monthlyLimit``/``used``/``onDemandCap``), which is
   all zeros for a subscription account and says nothing about the weekly pool.
2. ``creditUsagePercent`` is the PERCENT USED (not remaining), and the reset is
   ``currentPeriod.end``.

Live payload captured 2026-11-03:

    {"config": {
       "currentPeriod": {"type": "USAGE_PERIOD_TYPE_WEEKLY",
                         "start": "2026-10-31T23:42:23.488235+00:00",
                         "end": "2026-11-07T23:42:23.488235+00:00"},
       "creditUsagePercent": 11.0,
       "onDemandCap": {"val": 0}, "onDemandUsed": {"val": 0},
       "productUsage": [{"product": "GrokChat", "usagePercent": 10.0},
                        {"product": "GrokBuild", "usagePercent": 1.0},
                        {"product": "GrokImagine"}],
       "isUnifiedBillingUser": true, "prepaidBalance": {"val": 0},
       "billingPeriodStart": "2026-10-31T23:42:23.488235+00:00",
       "billingPeriodEnd": "2026-11-07T23:42:23.488235+00:00"}}
"""

from agent import account_usage

_CREDITS = {
    "config": {
        "currentPeriod": {"type": "USAGE_PERIOD_TYPE_WEEKLY",
                          "start": "2026-10-31T23:42:23.488235+00:00",
                          "end": "2026-11-07T23:42:23.488235+00:00"},
        "creditUsagePercent": 11.0,
        "onDemandCap": {"val": 0}, "onDemandUsed": {"val": 0},
        "productUsage": [
            {"product": "GrokChat", "usagePercent": 10.0},
            {"product": "GrokBuild", "usagePercent": 1.0},
            {"product": "GrokImagine"},
        ],
        "isUnifiedBillingUser": True, "prepaidBalance": {"val": 0},
        "billingPeriodStart": "2026-10-31T23:42:23.488235+00:00",
        "billingPeriodEnd": "2026-11-07T23:42:23.488235+00:00",
    }
}
_SETTINGS = {"subscription_tier_display": "X Premium+"}


def _fetch_with(credits, settings=None):
    original = account_usage._get_json
    calls = []

    def _fake(url, headers=None, timeout=None):
        calls.append(url)
        if "settings" in url:
            return settings if settings is not None else {}
        return credits

    account_usage._get_json = _fake
    try:
        return account_usage._fetch_xai_account_usage(api_key="test-token"), calls
    finally:
        account_usage._get_json = original


def test_xai_requests_the_credits_format():
    """The metered-API shape (no format=credits) is all zeros for a subscription."""
    _, calls = _fetch_with(_CREDITS)
    assert any("format=credits" in c for c in calls), f"credits format required, got {calls}"
    assert all("cli-chat-proxy.grok.com" in c for c in calls), calls


def test_xai_weekly_window_uses_credit_usage_percent():
    snap, _ = _fetch_with(_CREDITS)
    assert snap is not None
    assert len(snap.windows) == 1
    window = snap.windows[0]
    assert window.label == "Weekly"
    assert window.used_percent == 11.0, "creditUsagePercent is the percent USED"
    assert window.reset_at is not None
    assert window.reset_at.timestamp() == 1794094943.488235


def test_xai_product_breakdown_is_in_details():
    snap, _ = _fetch_with(_CREDITS)
    assert snap is not None
    joined = " | ".join(snap.details)
    assert "GrokChat" in joined and "10%" in joined, joined
    assert "GrokBuild" in joined and "1%" in joined, joined
    # a product with no percent is not reported as 0%
    assert "GrokImagine" not in joined, joined


def test_xai_plan_comes_from_settings():
    snap, _ = _fetch_with(_CREDITS, settings=_SETTINGS)
    assert snap is not None
    assert snap.plan == "X Premium+"


def test_xai_missing_credit_percent_yields_no_fabricated_window():
    credits = {"config": {"currentPeriod": _CREDITS["config"]["currentPeriod"]}}
    snap, _ = _fetch_with(credits)
    assert snap is not None
    assert not snap.windows, "no usage value must not become a 0% window"


def test_xai_fetcher_is_registered():
    assert account_usage._USAGE_FETCHERS.get("xai-oauth") is account_usage._fetch_xai_account_usage
