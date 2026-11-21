import concurrent.futures
import contextvars
import io
import json
import threading
import time
from datetime import datetime, timezone, UTC
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest

import agent.account_usage_nous as nous_usage
from hermes_cli import nous_account

from agent.account_usage import (
    AccountUsageSnapshot,
    AccountUsageWindow,
    fetch_account_usage,
    render_account_usage_lines,
)
from agent.account_usage_nous import _fetch_portal_account
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
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)  # else the fixture profile outlives the test
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


@pytest.fixture
def stored_nous_portal(tmp_path, monkeypatch):
    """Real stored-token/auth normalization, with only the HTTP transport replaced."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    state = {"access_token": "stored-access", "refresh_token": "single-use-refresh",
             "expires_at": "2100-01-01T00:00:00Z", "portal_base_url": "https://portal.example.invalid"}
    auth_path = home / "auth.json"
    auth_path.write_text(json.dumps({"version": 1, "providers": {"nous": state}}))
    payload = {
        "user": {"privy_did": "did:privy:member", "email": "member@example.invalid"},
        "organisation": {"id": "org-one"},
        "subscription": {"plan": "Pro"},
        "paid_service_access": {"allowed": True, "purchased_credits_remaining": 12.5},
        "usage_window": {"keep_going_on_credits": True, "gauges": [
            {"id": "day", "scope": "all", "used_bp": 2500, "resets_at_ms": 1_800_000_000_000},
            {"id": "top_tier", "scope": "top_tier", "used_bp": 5000},
        ]},
    }
    calls = []

    def urlopen(request, timeout):
        calls.append((request.get_method(), request.full_url, request.get_header("Authorization"), timeout))
        return io.BytesIO(json.dumps(payload).encode())

    def forbidden(*args, **kwargs):
        pytest.fail("A read-only quota fetch must never resolve/refresh Nous credentials or write auth")

    monkeypatch.setattr(nous_account.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(nous_account, "get_nous_portal_account_info", forbidden)
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", forbidden)
    monkeypatch.setattr("hermes_cli.auth._save_auth_store", forbidden)
    token = set_hermes_home_override(home)
    try:
        yield state, payload, auth_path, calls
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("outcome", ["success", "signed_out", "expired", "401", "503", "timeout", "error_body", "explicit_key"])
def test_nous_read_only_usage_never_repairs_credentials(stored_nous_portal, monkeypatch, outcome):
    state, payload, auth_path, calls = stored_nous_portal
    release = threading.Event()
    if outcome == "signed_out":
        state.clear()
    elif outcome == "expired":
        state["expires_at"] = "2000-01-01T00:00:00Z"
    auth_path.write_text(json.dumps({"version": 1, "providers": {"nous": state}}))
    before = auth_path.read_bytes()
    if outcome in {"401", "503", "timeout"}:
        def unavailable(request, timeout):
            calls.append(request.full_url)
            if outcome == "timeout":
                release.wait(30)
                return io.BytesIO(b"{}")
            raise HTTPError(request.full_url, int(outcome), "unavailable", Message(), None)

        monkeypatch.setattr(nous_account.urllib.request, "urlopen", unavailable)
    if outcome == "timeout":
        original = nous_usage._fetch_portal_account
        monkeypatch.setattr(nous_usage, "_fetch_portal_account",
                            lambda timeout, **kwargs: original(timeout=0.05, **kwargs))
    if outcome == "error_body":
        payload["error"] = "portal unavailable"
    started = time.monotonic()
    try:
        snapshot = fetch_account_usage("nous", read_only=True,
                                       api_key="other-account-key" if outcome == "explicit_key" else None)
    finally:
        release.set()
    assert time.monotonic() - started < 3
    assert auth_path.read_bytes() == before
    if outcome == "success":
        assert snapshot is not None
        assert snapshot.provider == "nous" and snapshot.source == "portal-account"
        assert snapshot.plan == "Pro" and snapshot.usage_window
        assert snapshot.windows == (
            AccountUsageWindow("Today", 25.0, datetime.fromtimestamp(1_800_000_000, tz=UTC)),
            AccountUsageWindow("Large models (monthly)", 50.0),
        )
        assert "Top-up credits: $12.50" in snapshot.details
        assert "Keep going on credits: on. Uses your credits when included usage runs out." in snapshot.details
        assert snapshot.raw is None
        assert calls == [("GET", "https://portal.example.invalid/api/oauth/account", "Bearer stored-access", 8)]
    else:
        assert snapshot is None
        assert bool(calls) == (outcome in {"401", "503", "timeout", "error_body"})


@pytest.mark.parametrize("section,key,value,same_identity", [
    ("state", "access_token", "replacement-access", True),
    ("user", "privy_did", "did:privy:another-member", False),
    ("organisation", "id", "org-two", False),
    ("state", "portal_base_url", "https://other-portal.example.invalid", False),
    ("user", "privy_did", None, None),
    ("organisation", "id", None, None),
], ids=["token", "user", "org", "portal", "missing_user", "missing_org"])
def test_nous_read_only_identity_tracks_the_portal_principal_not_the_token(
    stored_nous_portal, section, key, value, same_identity,
):
    state, payload, auth_path, calls = stored_nous_portal
    original = fetch_account_usage("nous", read_only=True)
    assert original is not None and original.identity is not None
    state["access_token"] = "replacement-access"
    target = {"state": state, "user": payload["user"], "organisation": payload["organisation"]}[section]
    if value is None:
        target.pop(key)
    else:
        target[key] = value
    auth_path.write_text(json.dumps({"version": 1, "providers": {"nous": state}}))
    updated = fetch_account_usage("nous", read_only=True)
    assert updated is not None
    assert len(calls) == 2
    if same_identity is None:
        assert updated.identity is None  # no guess from email, token or profile home
    else:
        assert updated.identity is not None
        assert (updated.identity == original.identity) is same_identity
    for snapshot in (original, updated):
        assert snapshot.raw is None
        for sensitive in ("stored-access", "replacement-access", "single-use-refresh",
                          "member@example.invalid", "did:privy:member"):
            assert sensitive not in repr(snapshot)
