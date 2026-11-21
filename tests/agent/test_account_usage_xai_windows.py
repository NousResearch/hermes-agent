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


def test_xai_weekly_window_stays_account_scoped():
    """Negative control for the MiniMax scope change: the xAI weekly pool IS the account.

    MiniMax's per-family allowances are model-scoped; the xAI weekly credit pool is a
    single account-wide allowance and must stay account-scoped, or an exhausted pool
    would fail to bench the credential.
    """
    snap, _ = _fetch_with(_CREDITS)
    assert snap is not None
    assert snap.windows[0].scope == "account"


def _fetch_read_only_with(monkey_calls):
    """Resolve the credential through a stub so the refresh flag is observable."""
    from hermes_cli import auth_xai

    original_resolver = auth_xai.resolve_xai_oauth_runtime_credentials
    original_get_json = account_usage._get_json

    def _resolver(**kwargs):
        monkey_calls.append(kwargs)
        return {"api_key": "peeked-token"}

    def _fake_get_json(url, headers=None, timeout=None):
        if "settings" in url:
            return _SETTINGS
        return _CREDITS

    auth_xai.resolve_xai_oauth_runtime_credentials = _resolver
    account_usage._get_json = _fake_get_json
    try:
        return account_usage._fetch_xai_account_usage_read_only()
    finally:
        auth_xai.resolve_xai_oauth_runtime_credentials = original_resolver
        account_usage._get_json = original_get_json


def test_xai_read_only_fetch_never_refreshes_the_credential():
    """A usage READ must not take the auth-store lock or rotate the xAI refresh token.

    ``resolve_xai_oauth_runtime_credentials`` defaults to ``refresh_if_expiring=True``,
    so the live fetch may repair the credential it owns -- but the picker's per-credential
    read only wants to LOOK at it. This path must resolve with refreshing off.
    """
    calls = []
    snap = _fetch_read_only_with(calls)
    assert calls, "the read-only path must resolve the credential"
    assert calls[0].get("refresh_if_expiring") is False, \
        f"a usage read must not refresh the credential, got {calls[0]}"
    assert snap is not None
    assert snap.windows[0].used_percent == 11.0


def test_xai_live_fetch_may_refresh_the_credential():
    """Negative control: the live path keeps its repair path for the credential it owns."""
    from hermes_cli import auth_xai

    original_resolver = auth_xai.resolve_xai_oauth_runtime_credentials
    original_get_json = account_usage._get_json
    calls = []

    auth_xai.resolve_xai_oauth_runtime_credentials = lambda **kw: (calls.append(kw), {"api_key": "live-token"})[1]
    account_usage._get_json = lambda url, headers=None, timeout=None: (_SETTINGS if "settings" in url else _CREDITS)
    try:
        snap = account_usage._fetch_xai_account_usage()
    finally:
        auth_xai.resolve_xai_oauth_runtime_credentials = original_resolver
        account_usage._get_json = original_get_json
    assert calls and calls[0].get("refresh_if_expiring") is True, calls
    assert snap is not None


def test_xai_read_only_fetcher_is_registered_for_the_picker():
    """The picker's read-only path must have its own xAI fetcher, not the refreshing one."""
    assert account_usage._READ_ONLY_USAGE_FETCHERS.get("xai-oauth") is \
        account_usage._fetch_xai_account_usage_read_only
    assert account_usage._READ_ONLY_USAGE_FETCHERS.get("xai-oauth") is not \
        account_usage._USAGE_FETCHERS.get("xai-oauth")
