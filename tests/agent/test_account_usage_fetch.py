import concurrent.futures
import contextvars
import threading
import time
from datetime import datetime, timezone, UTC

import pytest

from agent.account_usage import (
    AccountUsageSnapshot,
    AccountUsageWindow,
    _fetch_portal_account,
    fetch_account_usage,
    render_account_usage_lines,
)
from agent.billing_usage import fetch_nous_account as _billing_fetch_nous_account
from providers.base import ProviderProfile


class _UsageProfile(ProviderProfile):
    def __init__(self, snapshot=None, error=None, name="plugin-usage"):
        super().__init__(name=name)
        self.snapshot = snapshot
        self.error = error
        self.calls = 0

    def fetch_account_usage(self, *, base_url=None, api_key=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.snapshot


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _Client:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        return _Response(self._payload)


class _RoutingClient:
    def __init__(self, payloads):
        self._payloads = payloads

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        return _Response(self._payloads[url])


def test_fetch_account_usage_codex(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_codex_runtime_credentials",
        lambda refresh_if_expiring=True: {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "access-token",
        },
    )
    monkeypatch.setattr(
        "agent.account_usage._read_codex_tokens",
        lambda: {"tokens": {"account_id": "acct_123"}},
    )
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=15.0: _Client(
            {
                "plan_type": "pro",
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 15,
                        "reset_at": 1_900_000_000,
                        "limit_window_seconds": 18000,
                    },
                    "secondary_window": {
                        "used_percent": 40,
                        "reset_at": 1_900_500_000,
                        "limit_window_seconds": 604800,
                    },
                },
                "credits": {"has_credits": True, "balance": 12.5},
            }
        ),
    )

    snapshot = fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert snapshot.plan == "Pro"
    assert len(snapshot.windows) == 2
    assert snapshot.windows[0].label == "Session"
    assert snapshot.windows[0].used_percent == 15.0
    assert snapshot.windows[0].reset_at == datetime.fromtimestamp(1_900_000_000, tz=UTC)
    assert "Credits balance: $12.50" in snapshot.details


def _register_profile(monkeypatch, profile):
    import providers

    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    providers.register_provider(profile)


def test_fetch_account_usage_reaches_registered_plugin_profile_and_fails_open(monkeypatch):
    """A profile registered through the public registry (as a plugin does) feeds /usage; a profile
    without the hook, or one whose hook raises, is indistinguishable from today's empty block."""
    snapshot = AccountUsageSnapshot(
        provider="plugin-usage", source="plugin", fetched_at=datetime.now(UTC),
        details=("Credit: 10/100",),
    )
    profile = _UsageProfile(snapshot)
    _register_profile(monkeypatch, profile)
    _register_profile(monkeypatch, ProviderProfile(name="plugin-silent"))
    _register_profile(monkeypatch, _UsageProfile(error=RuntimeError("nope"), name="plugin-broken"))

    assert fetch_account_usage("plugin-usage", base_url="https://plugin.test", api_key="key") is snapshot
    assert profile.calls == 1
    assert fetch_account_usage("plugin-silent") is None
    assert fetch_account_usage("plugin-broken") is None


def test_fetch_account_usage_prefers_builtin_fetcher_over_profile(monkeypatch):
    builtin = AccountUsageSnapshot(
        provider="openrouter", source="builtin", fetched_at=datetime.now(UTC),
    )
    profile = _UsageProfile(
        AccountUsageSnapshot(provider="openrouter", source="plugin", fetched_at=datetime.now(UTC)),
        name="openrouter",
    )
    monkeypatch.setattr("agent.account_usage._USAGE_FETCHERS", {"openrouter": lambda base_url, api_key: builtin})
    _register_profile(monkeypatch, profile)

    assert fetch_account_usage("openrouter") is builtin
    assert profile.calls == 0


def test_fetch_account_usage_openrouter_uses_limit_remaining_and_ignores_deprecated_rate_limit(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda requested, explicit_base_url=None, explicit_api_key=None: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-test",
        },
    )
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=10.0: _RoutingClient(
            {
                "https://openrouter.ai/api/v1/credits": {
                    "data": {"total_credits": 300.0, "total_usage": 10.92}
                },
                "https://openrouter.ai/api/v1/key": {
                    "data": {
                        "limit": 100.0,
                        "limit_remaining": 70.0,
                        "limit_reset": "monthly",
                        "usage": 12.5,
                        "usage_daily": 0.5,
                        "usage_weekly": 2.0,
                        "usage_monthly": 8.0,
                        "rate_limit": {"requests": -1, "interval": "10s"},
                    }
                },
            }
        ),
    )

    snapshot = fetch_account_usage("openrouter")

    assert snapshot is not None
    assert snapshot.windows == (
        AccountUsageWindow(
            label="API key quota",
            used_percent=30.0,
            detail="$70.00 of $100.00 remaining • resets monthly",
        ),
    )
    assert "Credits balance: $289.08" in snapshot.details
    assert "API key usage: $12.50 total • $0.50 today • $2.00 this week • $8.00 this month" in snapshot.details
    assert all("-1 requests / 10s" not in line for line in render_account_usage_lines(snapshot))


def test_fetch_account_usage_openrouter_omits_quota_window_when_key_has_no_limit(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda requested, explicit_base_url=None, explicit_api_key=None: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-test",
        },
    )
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=10.0: _RoutingClient(
            {
                "https://openrouter.ai/api/v1/credits": {
                    "data": {"total_credits": 100.0, "total_usage": 25.5}
                },
                "https://openrouter.ai/api/v1/key": {
                    "data": {
                        "limit": None,
                        "limit_remaining": None,
                        "usage": 25.5,
                        "usage_daily": 1.25,
                        "usage_weekly": 4.5,
                        "usage_monthly": 18.0,
                    }
                },
            }
        ),
    )

    snapshot = fetch_account_usage("openrouter")

    assert snapshot is not None
    assert snapshot.windows == ()
    assert "Credits balance: $74.50" in snapshot.details
    assert "API key usage: $25.50 total • $1.25 today • $4.50 this week • $18.00 this month" in snapshot.details


def test_plugin_usage_hook_is_bounded_and_fails_open(monkeypatch):
    """A plugin hook that overruns the shared deadline yields None within deadline+1 s on every surface
    (gateway/TUI call ``fetch_account_usage`` with no bound of their own); built-in fetchers are untouched."""
    import threading
    import time

    from agent import account_usage

    started = threading.Event()

    class _Hang(ProviderProfile):
        def fetch_account_usage(self, *, base_url=None, api_key=None):
            started.set()
            time.sleep(5)
            return AccountUsageSnapshot(provider=self.name, source="late", fetched_at=datetime.now(UTC))

    _register_profile(monkeypatch, _Hang(name="plugin-hang"))
    monkeypatch.setattr(account_usage, "PLUGIN_USAGE_HOOK_DEADLINE_S", 0.3)
    builtin_calls = []
    monkeypatch.setattr(account_usage, "_USAGE_FETCHERS",
                        {"openrouter": lambda base_url, api_key: builtin_calls.append(1)})

    t0 = time.monotonic()
    assert account_usage.fetch_account_usage("plugin-hang") is None
    assert time.monotonic() - t0 < 1.3 and started.is_set()
    account_usage.fetch_account_usage("openrouter")
    assert builtin_calls == [1]


def test_plugin_usage_hook_failure_never_reaches_threading_excepthook(monkeypatch):
    """A raising hook fails open in the caller — it must not die on a worker thread, where
    ``threading.excepthook`` prints a traceback into every ``/usage`` surface."""
    import threading

    from agent import account_usage

    class _Boom(ProviderProfile):
        def fetch_account_usage(self, *, base_url=None, api_key=None):
            raise RuntimeError("boom from plugin")

    _register_profile(monkeypatch, _Boom(name="plugin-boom"))
    monkeypatch.setattr(account_usage, "_USAGE_FETCHERS", {})
    escaped = []
    monkeypatch.setattr(threading, "excepthook", lambda args: escaped.append(args.exc_value))

    assert account_usage.fetch_account_usage("plugin-boom") is None
    for t in threading.enumerate():
        if t is not threading.current_thread() and "account-usage" in t.name:
            t.join(2)
    assert escaped == []


@pytest.mark.parametrize("fetch", [_fetch_portal_account, _billing_fetch_nous_account])
def test_fetch_portal_account_is_wall_clock_bounded(monkeypatch, fetch):
    """A portal that accepts the connection but never answers must release the
    caller at ``timeout``, not when the wedged worker finishes on its own
    (``Executor.__exit__`` used to join it via ``shutdown(wait=True)``) — on the
    /usage path and the /billing path alike (#115982)."""
    release = threading.Event()

    def hanging_portal_fetch(*, force_fresh):
        release.wait(timeout=30)
        return object()

    monkeypatch.setattr(
        "hermes_cli.nous_account.get_nous_portal_account_info", hanging_portal_fetch
    )
    started = time.monotonic()
    try:
        with pytest.raises(concurrent.futures.TimeoutError):
            fetch(timeout=0.5)
    finally:
        release.set()
    assert time.monotonic() - started < 10


def test_fetch_portal_account_returns_value_and_keeps_caller_context(monkeypatch):
    marker = contextvars.ContextVar("portal_fetch_test_marker", default="unset")
    sentinel = object()
    seen = {}

    def probing_portal_fetch(*, force_fresh):
        seen["force_fresh"] = force_fresh
        seen["marker"] = marker.get()
        return sentinel

    monkeypatch.setattr(
        "hermes_cli.nous_account.get_nous_portal_account_info", probing_portal_fetch
    )
    token = marker.set("profile-scope")
    try:
        assert _fetch_portal_account(timeout=5) is sentinel
    finally:
        marker.reset(token)
    assert seen == {"force_fresh": True, "marker": "profile-scope"}


_XAI_FAKE_TOKEN = "fake-xai-oauth-token-123"
_XAI_BILLING = "https://cli-chat-proxy.grok.com/v1/billing?format=credits"
_XAI_SETTINGS = "https://cli-chat-proxy.grok.com/v1/settings"
_XAI_BILLING_PAYLOAD = {
    "config": {
        "currentPeriod": {
            "type": "USAGE_PERIOD_TYPE_WEEKLY",
            "start": "2026-09-27T03:07:35+00:00",
            "end": "2026-10-04T03:07:35+00:00",
        },
        "creditUsagePercent": 14.0,
        "onDemandCap": {"val": 0},
        "onDemandUsed": {"val": 0},
        "productUsage": [
            {"product": "GrokBuild", "usagePercent": 10.0},
            {"product": "GrokChat", "usagePercent": 4.0},
        ],
        "prepaidBalance": {"val": 0},
    }
}


class _XaiClient:
    """Records every request; ``statuses`` overrides the HTTP status per URL."""

    def __init__(self, payloads, statuses=None, calls=None):
        self._payloads, self._statuses, self.calls = payloads, statuses or {}, calls if calls is not None else []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        import httpx

        self.calls.append((url, headers))
        status = self._statuses.get(url, 200)
        request = httpx.Request("GET", url)
        if status >= 400:
            return httpx.Response(status, request=request, json={"error": "secret body"})
        return httpx.Response(status, request=request, json=self._payloads.get(url, {}))


class _XaiCalls(list):
    """HTTP calls, plus the resolver kwargs and pool peeks the fetcher made."""

    def __init__(self):
        super().__init__()
        self.resolve_kwargs, self.peeks = [], []


class _PeekOnlyPool:
    """Exposes only ``peek``: a select/refresh/persist call would raise AttributeError."""

    def __init__(self, entry, peeks):
        self._entry, self._peeks = entry, peeks

    def peek(self):
        self._peeks.append(True)
        return self._entry


def _patch_xai(monkeypatch, *, token=_XAI_FAKE_TOKEN, statuses=None, payloads=None, pool_entry=None,
               resolve_error=None):
    from hermes_cli.auth import AuthError

    calls = _XaiCalls()

    def _resolve(**kw):
        calls.resolve_kwargs.append(kw)
        if resolve_error is not None:
            raise resolve_error
        if token is None:
            raise AuthError("not signed in", provider="xai-oauth", code="missing")
        return {"provider": "xai-oauth", "api_key": token}

    monkeypatch.setattr("hermes_cli.auth_xai.resolve_xai_oauth_runtime_credentials", _resolve)
    monkeypatch.setattr("agent.credential_pool.load_pool",
                        lambda provider: _PeekOnlyPool(pool_entry if provider == "xai-oauth" else None, calls.peeks))
    payloads = payloads if payloads is not None else {
        _XAI_BILLING: _XAI_BILLING_PAYLOAD, _XAI_SETTINGS: {"subscription_tier_display": "SuperGrok"},
    }
    monkeypatch.setattr("agent.account_usage.httpx.Client",
                        lambda timeout=None: _XaiClient(payloads, statuses, calls))
    return calls


def test_xai_oauth_usage_renders_weekly_pool_without_leaking_token(monkeypatch):
    calls = _patch_xai(monkeypatch)

    snapshot = fetch_account_usage("xai-oauth")
    text = "\n".join(render_account_usage_lines(snapshot))

    assert snapshot is not None and snapshot.available
    assert snapshot.plan == "SuperGrok"
    assert snapshot.windows[0].label == "Weekly pool"
    assert snapshot.windows[0].reset_at == datetime(2026, 10, 4, 3, 7, 35, tzinfo=timezone.utc)
    assert "Weekly pool: 86% remaining (14% used) • resets " in text
    for line in ("Grok Build: 10% used", "Grok Chat: 4% used", "Pay-as-you-go: off", "Prepaid balance: $0"):
        assert line in snapshot.details
    assert any(line.startswith("Window: ") for line in snapshot.details)
    assert _XAI_FAKE_TOKEN not in text
    assert {url for url, _ in calls} == {_XAI_BILLING, _XAI_SETTINGS}
    assert all(h["Authorization"] == f"Bearer {_XAI_FAKE_TOKEN}" for _, h in calls)


def test_xai_oauth_nonzero_money_is_raw_units_and_unknown_products_kept(monkeypatch):
    config = dict(_XAI_BILLING_PAYLOAD["config"], onDemandCap={"val": "2500"}, prepaidBalance={"val": 700},
                  productUsage=[{"product": "GrokImagineVideo", "usagePercent": 1.5}])
    _patch_xai(monkeypatch, payloads={_XAI_BILLING: {"config": config}}, statuses={_XAI_SETTINGS: 500})

    snapshot = fetch_account_usage("xai-oauth")

    assert snapshot is not None and snapshot.available and snapshot.plan is None
    assert "Grok Imagine Video: 1.5% used" in snapshot.details
    assert "Pay-as-you-go: on (cap 2500 raw units, used 0 raw units)" in snapshot.details
    assert "Prepaid balance: 700 raw units" in snapshot.details
    assert not any("$" in line for line in snapshot.details)


@pytest.mark.parametrize("status", [401, 403, 500])
def test_xai_oauth_billing_failure_is_unavailable_not_raised(monkeypatch, status):
    _patch_xai(monkeypatch, statuses={_XAI_BILLING: status})

    snapshot = fetch_account_usage("xai-oauth")
    text = "\n".join(render_account_usage_lines(snapshot))

    assert snapshot is not None and not snapshot.available and snapshot.windows == ()
    assert "hermes auth add xai-oauth" in snapshot.unavailable_reason
    assert _XAI_FAKE_TOKEN not in text and "secret body" not in text


def test_xai_oauth_without_token_returns_none(monkeypatch):
    calls = _patch_xai(monkeypatch, token=None)

    assert fetch_account_usage("xai-oauth") is None
    assert calls == []


def test_xai_oauth_usage_never_refreshes_the_singleton(monkeypatch):
    calls = _patch_xai(monkeypatch)

    assert fetch_account_usage("xai-oauth").available
    assert calls.resolve_kwargs and all(kw.get("refresh_if_expiring") is False for kw in calls.resolve_kwargs)
    assert calls.peeks == []  # live singleton token wins; the pool is not consulted


def test_xai_oauth_pool_only_login_uses_peeked_access_token(monkeypatch):
    from types import SimpleNamespace

    pooled = "pooled-xai-access-token"
    calls = _patch_xai(monkeypatch, token=None, pool_entry=SimpleNamespace(access_token=pooled))

    snapshot = fetch_account_usage("xai-oauth")

    assert snapshot is not None and snapshot.available
    assert calls.peeks == [True]
    assert _XAI_BILLING in {url for url, _ in calls}
    assert all(h["Authorization"] == f"Bearer {pooled}" for _, h in calls)
    assert pooled not in "\n".join(render_account_usage_lines(snapshot))


@pytest.mark.parametrize("entry", [None, "empty"])
def test_xai_oauth_no_singleton_and_no_pool_token_returns_none(monkeypatch, entry):
    from types import SimpleNamespace

    pool_entry = SimpleNamespace(access_token="") if entry == "empty" else None
    calls = _patch_xai(monkeypatch, token=None, pool_entry=pool_entry)

    assert fetch_account_usage("xai-oauth") is None
    assert calls == [] and calls.peeks == [True]


def test_xai_oauth_unexpected_resolver_error_fails_open_without_peeking(monkeypatch):
    from types import SimpleNamespace

    calls = _patch_xai(monkeypatch, resolve_error=TimeoutError("auth lock"),
                       pool_entry=SimpleNamespace(access_token="other-account-token"))

    assert fetch_account_usage("xai-oauth") is None
    assert calls == [] and calls.peeks == []


@pytest.mark.parametrize("provider", ["openai-codex", "xai"])
def test_grok_billing_is_only_queried_for_xai_oauth(monkeypatch, provider):
    calls = _patch_xai(monkeypatch)
    monkeypatch.setattr(
        "agent.account_usage.resolve_codex_runtime_credentials",
        lambda **_kw: {"provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex",
                       "api_key": "codex-token"},
    )
    monkeypatch.setattr("agent.account_usage._read_codex_tokens", lambda: {"tokens": {}})

    fetch_account_usage(provider)

    assert not any("cli-chat-proxy.grok.com" in url for url, _ in calls)
