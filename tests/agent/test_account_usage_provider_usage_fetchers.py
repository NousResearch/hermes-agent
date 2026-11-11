"""Core-owned provider usage fetchers and non-secret account identities.

Synthetic payloads only: no account is contacted and no credential leaves the
process.  The assertions pin the fields the Provider Usage plugin pane needs on
``snapshot.raw`` (it parses them itself; core does not reshape the UI) plus the
footer correlation contract:

* ``identity_id`` is the account namespace shared with the usage cache and the
  delegation roster (``_identity_id_for``: Codex principal, existing one-way
  token fingerprint, or ``entry:<pool id>``) — never the token itself;
* ``credential_id`` is the selected pool row's existing stable id;
* ``account_id`` is a provider-exposed stable account id (Codex principal);
* ``profile_scope`` is the canonical non-path home scope.

A missing identity stays ``None`` (unknown), and an explicit per-identity read
never falls back to another pooled account.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime

import httpx
import pytest

from agent import account_usage

SECRET = "«redacted:sk-…»"
OTHER_SECRET = "«redacted:sk-other-…»"

DEEPSEEK_BALANCE = {
    "is_available": True,
    "balance_infos": [
        {"currency": "usd", "total_balance": "12.34",
         "granted_balance": "2.00", "topped_up_balance": "10.34"},
        {"currency": "cny", "total_balance": "88.00"},
    ],
}

OLLAMA_USAGE = {
    "limits": {
        "session": {"usage": 0.4},
        "weekly": {"usage": 0.55},
        "monthly": {"usage": 1.0, "models": {"gpt-oss:120b": {"request_count": 12}}},
    },
    "activity": {"cost": "0.42", "period": {"type": "monthly"}},
    "totals": {"request_count": 1200, "usage_usd": 3.5},
}

OLLAMA_BALANCE = {
    "included": {"allowance_usd": 20.0, "balance_usd": 5.0,
                 "period": {"until": "2026-12-01T00:00:00Z"}},
    "purchased": {"balance_usd": 2.5},
}

ZEN_MODELS = {"data": [{"id": "synthetic-model"}]}

GO_USAGE = {"usage": {
    "rolling": {"percent": 55.0, "resetsAt": "2026-11-11T20:00:00Z"},
    "weekly": {"percent": 20.0, "resetsAt": "2026-11-17T00:00:00Z"},
}}


class _Response:
    def __init__(self, payload, *, status_code=200, url="https://example.test/"):
        self._payload = payload
        self.status_code = status_code
        self._url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", self._url)
            raise httpx.HTTPStatusError(
                "synthetic", request=request,
                response=httpx.Response(self.status_code, request=request))

    def json(self):
        return self._payload


class _Client:
    def __init__(self, routes, calls):
        self._routes = routes
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        self._calls.append((url, dict(headers or {})))
        for suffix, payload in self._routes:
            if url.endswith(suffix):
                if callable(payload):
                    return payload(url)
                if isinstance(payload, _Response):
                    return payload
                return _Response(payload, url=url)
        return _Response({}, status_code=404, url=url)


def _http(monkeypatch, routes):
    calls = []
    monkeypatch.setattr(account_usage.httpx, "Client", lambda timeout=None: _Client(routes, calls))
    return calls


class _Entry:
    def __init__(self, ident, token, *, label="Account", status="ok",
                 base_url="https://pool.example/v1", request_count=3,
                 auth_type="api_key", source="manual"):
        self.id = ident
        self.label = label
        self.auth_type = auth_type
        self.source = source
        self.access_token = token
        self.runtime_api_key = token
        self.runtime_base_url = base_url
        self.last_status = status
        self.request_count = request_count
        self.last_status_at = 1700000000.0


class _Pool:
    def __init__(self, entries):
        self._entries = list(entries)
        self.select_calls = 0

    def entries(self):
        return list(self._entries)

    def select(self, **kwargs):
        self.select_calls += 1
        raise AssertionError("usage reads must not select/rotate a pooled credential")


def _runtime(monkeypatch, *, provider, base_url, api_key):
    calls = []

    def fake(requested=None, explicit_api_key=None, explicit_base_url=None, **kwargs):
        calls.append({"requested": requested, "explicit_api_key": explicit_api_key,
                      "explicit_base_url": explicit_base_url, **kwargs})
        return {"provider": provider, "base_url": base_url, "api_key": api_key}

    monkeypatch.setattr(account_usage, "resolve_runtime_provider", fake)
    return calls


def _jwt(account_id: str, sub: str) -> str:
    def part(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return (f"{part({'alg': 'none'})}."
            f"{part({'sub': sub, 'https://api.openai.com/auth': {'chatgpt_account_id': account_id}})}.sig")


# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------

def test_deepseek_raw_keeps_every_balance_and_currency(monkeypatch):
    calls = _runtime(monkeypatch, provider="deepseek",
                     base_url="https://api.deepseek.com/v1", api_key=SECRET)
    http = _http(monkeypatch, [("/user/balance", DEEPSEEK_BALANCE)])

    snapshot = account_usage.fetch_account_usage("deepseek", read_only=True)

    assert snapshot is not None
    assert snapshot.provider == "deepseek"
    assert snapshot.source == "user_balance_api"
    assert snapshot.title == "DeepSeek balance"
    assert snapshot.windows == ()
    assert snapshot.details == ("USD balance: 12.34", "CNY balance: 88.00")
    assert snapshot.available is True
    assert snapshot.raw == {"balance": DEEPSEEK_BALANCE}
    assert snapshot.identity_id is None, "env/config keys are unknown, never guessed"
    assert snapshot.account_id is None
    assert snapshot.profile_scope.startswith("home:")
    assert http == [("https://api.deepseek.com/user/balance",
                     {"Authorization": f"Bearer {SECRET}", "Accept": "application/json"})]
    assert calls == [{"requested": "deepseek", "explicit_api_key": None,
                      "explicit_base_url": None}]


def test_deepseek_accepts_legacy_flat_balance_payload(monkeypatch):
    payload = {"is_available": True, "currency": "usd", "total_balance": "4.50"}
    _runtime(monkeypatch, provider="deepseek",
             base_url="https://api.deepseek.com/v1", api_key=SECRET)
    _http(monkeypatch, [("/user/balance", payload)])

    snapshot = account_usage.fetch_account_usage("deepseek", read_only=True)

    assert snapshot is not None
    assert snapshot.details == ("USD balance: 4.50",)
    assert snapshot.raw == {"balance": payload}


def test_deepseek_unavailable_and_missing_auth(monkeypatch):
    _runtime(monkeypatch, provider="deepseek",
             base_url="https://api.deepseek.com/v1", api_key=SECRET)
    _http(monkeypatch, [("/user/balance",
                         {"is_available": False, "balance_infos": []})])

    snapshot = account_usage.fetch_account_usage("deepseek")
    assert snapshot is not None
    assert snapshot.details == ("Account available: no",)
    assert snapshot.unavailable_reason == "DeepSeek returned no balance entries."
    assert snapshot.available is False
    assert snapshot.raw == {"balance": {"is_available": False, "balance_infos": []}}

    _runtime(monkeypatch, provider="deepseek",
             base_url="https://api.deepseek.com/v1", api_key="")
    http = _http(monkeypatch, [("/user/balance", DEEPSEEK_BALANCE)])
    assert account_usage.fetch_account_usage("deepseek") is None
    assert http == []


def test_deepseek_raw_is_sanitized_but_usage_fields_survive(monkeypatch):
    payload = {
        "is_available": True,
        "balance_infos": [{"currency": "usd", "total_balance": "1.00"}],
        "api_key": SECRET,
        "nested": {"access_token": SECRET, "input_tokens": 5},
    }
    _runtime(monkeypatch, provider="deepseek",
             base_url="https://api.deepseek.com/v1", api_key=SECRET)
    _http(monkeypatch, [("/user/balance", payload)])

    snapshot = account_usage.fetch_account_usage("deepseek")
    assert snapshot is not None
    assert SECRET not in json.dumps(snapshot.raw)
    assert "api_key" not in snapshot.raw["balance"]
    assert "access_token" not in snapshot.raw["balance"]["nested"]
    assert snapshot.raw["balance"]["nested"]["input_tokens"] == 5


# ---------------------------------------------------------------------------
# Ollama Cloud
# ---------------------------------------------------------------------------

def test_ollama_windows_details_and_raw_keep_activity(monkeypatch):
    _runtime(monkeypatch, provider="ollama-cloud",
             base_url="https://ollama.com/v1", api_key=SECRET)
    http = _http(monkeypatch, [("/api/usage", OLLAMA_USAGE),
                               ("/api/balance", OLLAMA_BALANCE)])

    snapshot = account_usage.fetch_account_usage("ollama-cloud")

    assert snapshot is not None
    assert [window.label for window in snapshot.windows] == [
        "Monthly included", "5-hour", "Weekly", "Monthly"]
    assert [window.used_percent for window in snapshot.windows] == [75.0, 40.0, 55.0, 100.0]
    credit = snapshot.windows[0]
    assert credit.detail == "$15.00 spent · $5.00 left of $20"
    assert credit.reset_at == datetime(2026, 12, 1, tzinfo=UTC)
    assert snapshot.details == ("gpt-oss:120b: 12 requests", "Spend in the monthly: $0.42")
    assert snapshot.raw == {"usage": OLLAMA_USAGE, "balance": OLLAMA_BALANCE}
    assert [url for url, _ in http] == [
        "https://ollama.com/api/usage", "https://ollama.com/api/balance"]


def test_ollama_without_balance_keeps_usage_windows_and_reset_note(monkeypatch):
    _runtime(monkeypatch, provider="ollama-cloud",
             base_url="https://ollama.com/v1", api_key=SECRET)
    _http(monkeypatch, [("/api/usage", OLLAMA_USAGE), ("/api/balance", {})])

    snapshot = account_usage.fetch_account_usage("ollama-cloud")
    assert snapshot is not None
    assert [window.label for window in snapshot.windows] == ["5-hour", "Weekly", "Monthly"]
    assert snapshot.details[-1].startswith("Reset times are not published")
    assert snapshot.raw["balance"] == {}


def test_ollama_legacy_included_remaining_percent_windows(monkeypatch):
    balance = {"included": {
        "session": {"remaining_percent": 70.0, "resets_at": "2026-11-11T20:00:00Z"},
        "weekly": {"remaining_percent": 25.0},
    }}
    _runtime(monkeypatch, provider="ollama-cloud",
             base_url="https://ollama.com/v1", api_key=SECRET)
    _http(monkeypatch, [("/api/usage", {"limits": {}}), ("/api/balance", balance)])

    snapshot = account_usage.fetch_account_usage("ollama-cloud")

    assert snapshot is not None
    assert [(window.label, window.used_percent) for window in snapshot.windows] == [
        ("5-hour", 30.0), ("Weekly", 75.0)]
    assert snapshot.windows[0].reset_at == datetime(2026, 11, 11, 20, tzinfo=UTC)
    assert not any("Reset times are not published" in line for line in snapshot.details)
    assert (snapshot.raw or {}).get("balance") == balance


# ---------------------------------------------------------------------------
# OpenCode Zen / Go fallback
# ---------------------------------------------------------------------------

def test_opencode_zen_authenticated_models_route_keeps_raw(monkeypatch):
    _runtime(monkeypatch, provider="opencode-zen",
             base_url="https://opencode.ai/zen/v1", api_key=SECRET)
    http = _http(monkeypatch, [("/models", ZEN_MODELS)])

    snapshot = account_usage.fetch_account_usage("opencode-zen", read_only=True)

    assert snapshot is not None
    assert snapshot.source == "models_api"
    assert snapshot.plan == "Zen API credits"
    assert snapshot.details == (
        "Zen API access is authenticated. Credit balance and spend are not exposed through OpenCode API keys.",)
    assert snapshot.raw == {"models": ZEN_MODELS}
    assert snapshot.windows == ()
    assert http[0][0] == "https://opencode.ai/zen/v1/models"


def test_opencode_go_falls_back_to_core_when_profile_hook_is_empty(monkeypatch):
    import providers

    monkeypatch.setattr(providers, "get_provider_profile", lambda slug: None)
    _runtime(monkeypatch, provider="opencode-go",
             base_url="https://opencode.ai/zen/go", api_key=SECRET)
    http = _http(monkeypatch, [("/zen/go/v1/usage", GO_USAGE)])

    snapshot = account_usage.fetch_account_usage("opencode-go", read_only=True)

    assert snapshot is not None
    assert snapshot.provider == "opencode-go"
    assert snapshot.source == "go_usage_api"
    assert [(window.label, window.used_percent) for window in snapshot.windows] == [
        ("Rolling window", 55.0), ("Weekly", 20.0)]
    assert snapshot.windows[0].reset_at == datetime(2026, 11, 11, 20, tzinfo=UTC)
    assert snapshot.raw == GO_USAGE
    assert http[0][0] == "https://opencode.ai/zen/go/v1/usage"


def test_opencode_go_fallback_missing_auth_is_none(monkeypatch):
    import providers

    monkeypatch.setattr(providers, "get_provider_profile", lambda slug: None)
    _runtime(monkeypatch, provider="opencode-go",
             base_url="https://opencode.ai/zen/go", api_key="")
    http = _http(monkeypatch, [("/zen/go/v1/usage", GO_USAGE)])

    assert account_usage.fetch_account_usage("opencode-go") is None
    assert http == []


# ---------------------------------------------------------------------------
# Nous / Anthropic
# ---------------------------------------------------------------------------

def _nous_account():
    from hermes_cli.nous_account import (
        NousPaidServiceAccessInfo, NousPortalAccountInfo, NousPortalSubscriptionInfo,
    )

    return NousPortalAccountInfo(
        logged_in=True, source="account_api", fresh=True,
        portal_base_url="https://portal.example.test",
        subscription=NousPortalSubscriptionInfo(
            plan="Nous Pro", monthly_credits=100.0, credits_remaining=40.0,
            rollover_credits=5.0, current_period_end="2026-12-01"),
        paid_service_access=True,
        paid_service_access_info=NousPaidServiceAccessInfo(
            subscription_credits_remaining=35.0, purchased_credits_remaining=5.0,
            total_usable_credits=40.0),
    )


def test_nous_fetch_and_read_only_refresh_semantics(monkeypatch):
    calls = []

    def fake(force_fresh=False):
        calls.append(force_fresh)
        return _nous_account()

    monkeypatch.setattr("hermes_cli.nous_account.get_nous_portal_account_info", fake)

    # Core's provider dispatch deliberately leaves Nous to nous_credits_lines (no duplicate block);
    # the pane uses the named read-only entry point, which never forces a refresh.
    assert account_usage.fetch_account_usage("nous") is None
    normal = account_usage.fetch_nous_account_usage()
    read_only = account_usage.fetch_account_usage("nous", read_only=True)

    assert calls == [True, False]
    assert normal is not None and read_only is not None
    assert normal.plan == "Nous Pro"
    assert normal.windows[0].label == "Subscription"
    assert normal.windows[0].used_percent == 60.0
    assert any("Top up:" in line for line in normal.details)
    assert normal.profile_scope.startswith("home:")
    assert read_only.profile_scope.startswith("home:")


def test_nous_pool_identity_is_not_stamped_on_account_global_credits(monkeypatch):
    pool = _Pool([_Entry("nous-cred", SECRET)])

    def fake(force_fresh=False):
        return _nous_account()

    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)
    monkeypatch.setattr("hermes_cli.nous_account.get_nous_portal_account_info", fake)

    snapshot = account_usage.fetch_account_usage(
        "nous", identity_id="nous-cred", read_only=True)

    assert snapshot is not None
    assert snapshot.identity_id is None
    assert snapshot.account_id is None
    assert (snapshot.profile_scope or "").startswith("home:")


def test_anthropic_read_only_route_uses_the_explicit_token(monkeypatch):
    payload = {
        "five_hour": {"utilization": 0.25, "resets_at": "2026-11-11T12:00:00Z"},
        "seven_day": {"utilization": 0.5},
        "seven_day_opus": {"utilization": 1.0},
        "seven_day_sonnet": {"utilization": 0.05},
    }
    seen = {}

    def fake_get_json(url, headers, *, timeout):
        seen["url"], seen["headers"], seen["timeout"] = url, headers, timeout
        return payload

    monkeypatch.setattr(account_usage, "_get_json", fake_get_json)
    monkeypatch.setattr(account_usage, "resolve_anthropic_token",
                        lambda **kw: (_ for _ in ()).throw(AssertionError(
                            "an explicit token must not consult the ambient singleton")))

    assert "anthropic" in account_usage._READ_ONLY_USAGE_FETCHERS
    snapshot = account_usage.fetch_account_usage(
        "anthropic", api_key="sk-ant-oat-synthetic", read_only=True)

    assert snapshot is not None
    assert seen["url"] == "https://api.anthropic.com/api/oauth/usage"
    assert seen["headers"]["Authorization"] == "Bearer sk-ant-oat-synthetic"
    assert {window.label: window.scope for window in snapshot.windows} == {
        "Current session": "account", "Current week": "account",
        "Opus week": "model", "Sonnet week": "model"}
    assert snapshot.raw == payload
    assert snapshot.profile_scope.startswith("home:")


def test_anthropic_read_only_resolution_never_refreshes(monkeypatch):
    calls = []

    def fake_resolve(**kwargs):
        calls.append(kwargs)
        return "sk-ant-oat01-synthetic"

    monkeypatch.setattr(account_usage, "resolve_anthropic_token", fake_resolve)
    monkeypatch.setattr(account_usage, "_get_json", lambda url, headers, *, timeout: {})

    assert account_usage.fetch_account_usage("anthropic", read_only=True) is not None
    assert calls == [{"refresh": False}]


def test_anthropic_read_only_401_is_reported_without_a_retry(monkeypatch):
    calls = []

    def fake_get_json(url, headers, *, timeout):
        calls.append(url)
        request = httpx.Request("GET", url)
        raise httpx.HTTPStatusError(
            "synthetic", request=request,
            response=httpx.Response(401, request=request))

    monkeypatch.setattr(account_usage, "_get_json", fake_get_json)

    assert account_usage.fetch_account_usage(
        "anthropic", api_key="sk-ant-oat-synthetic", read_only=True) is None
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Inventory / presence / identity correspondence
# ---------------------------------------------------------------------------

def test_inventory_uses_account_namespace_and_never_token_value(monkeypatch):
    entries = [
        _Entry("cred-a", SECRET, label="Work key", request_count=7,
               base_url="https://user:pass@gateway.example/v1?api_key=" + SECRET + "#frag"),
        _Entry("cred-b", OTHER_SECRET, label=f"Key {OTHER_SECRET}", status="exhausted"),
        _Entry("", SECRET, label="No stable id"),
    ]
    pool = _Pool(entries)
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)

    rows = account_usage.list_account_usage_identities("openrouter")

    from agent.account_usage_cache import _identity_id_for
    assert [row["identity_id"] for row in rows] == [
        _identity_id_for("openrouter", entries[0]), _identity_id_for("openrouter", entries[1])]
    assert [row["credential_id"] for row in rows] == ["cred-a", "cred-b"]
    assert [row["label"] for row in rows] == ["Work key", "Key redacted"]
    assert [row["base_url"] for row in rows] == [
        "https://gateway.example/v1", "https://pool.example/v1"]
    assert [row["available"] for row in rows] == [True, False]
    assert all(row["profile_scope"].startswith("home:") for row in rows)
    text = json.dumps(rows)
    assert SECRET not in text and OTHER_SECRET not in text
    assert "user:pass" not in text
    assert "access_token" not in text and "refresh_token" not in text
    assert all(row["identity_id"].startswith("fp:") for row in rows)


def test_credential_summary_preserves_pool_only_env_oauth_and_config(monkeypatch):
    pool = _Pool([_Entry("cred-a", SECRET), _Entry("cred-b", OTHER_SECRET)])
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)
    pool_summary = account_usage.provider_credential_summary("openrouter")
    assert pool_summary == {
        "provider": "openrouter", "configured": True, "account_count": 2,
        "source": "pool", "profile_scope": pool_summary["profile_scope"]}

    empty = _Pool([])
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: empty)

    statuses = {
        "deepseek": {"configured": True, "logged_in": True, "key_source": "DEEPSEEK_API_KEY"},
        "nous": {"logged_in": True, "source": "auth.json"},
        "custom-local": {"configured": True},
        "missing": {"configured": False},
    }
    monkeypatch.setattr("hermes_cli.auth.get_auth_status",
                        lambda provider: statuses.get(provider, {}))
    assert account_usage.provider_credential_summary("deepseek")["source"] == "env"
    assert account_usage.provider_credential_summary("nous")["source"] == "oauth"
    assert account_usage.provider_credential_summary("custom-local")["source"] == "config"
    missing = account_usage.provider_credential_summary("missing")
    assert missing["configured"] is False and missing["source"] == "none"
    assert "DEEPSEEK_API_KEY" not in json.dumps(missing)


def test_codex_inventory_exposes_provider_account_id_not_a_secret(monkeypatch):
    token = _jwt("acct-inventory", "user-inventory")
    pool = _Pool([_Entry("codex-cred", token)])
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)

    rows = account_usage.list_account_usage_identities("openai-codex")

    assert [row["identity_id"] for row in rows] == ["codex:acct-inventory:user-inventory"]
    assert [row["credential_id"] for row in rows] == ["codex-cred"]
    assert [row["account_id"] for row in rows] == ["codex:acct-inventory:user-inventory"]
    assert token not in json.dumps(rows)


def test_identity_fetch_resolves_exactly_one_pool_entry_and_stamps_it(monkeypatch):
    pool = _Pool([
        _Entry("cred-a", SECRET, base_url="https://a.example/v1"),
        _Entry("cred-b", OTHER_SECRET, base_url="https://b.example/v1"),
    ])
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)
    monkeypatch.setattr(account_usage, "resolve_runtime_provider",
                        lambda **kw: (_ for _ in ()).throw(AssertionError(
                            "an explicit pool identity must not re-resolve credentials")))
    http = _http(monkeypatch, [("/user/balance", DEEPSEEK_BALANCE)])

    snapshot = account_usage.fetch_account_usage(
        "deepseek", identity_id="cred-b", read_only=True)

    from agent.account_usage_cache import _identity_id_for
    assert snapshot is not None
    assert snapshot.identity_id == _identity_id_for("deepseek", pool.entries()[1])
    assert (snapshot.identity_id or "").startswith("fp:")
    assert snapshot.account_id is None
    assert snapshot.profile_scope.startswith("home:")
    assert http[0][0] == "https://b.example/user/balance"
    assert http[0][1]["Authorization"] == f"Bearer {OTHER_SECRET}"
    assert pool.select_calls == 0

    before = len(http)
    assert account_usage.fetch_account_usage(
        "deepseek", identity_id="cred-missing", read_only=True) is None
    assert len(http) == before


def test_footer_identity_matches_delegation_provider_account_id(monkeypatch):
    from agent.account_usage_cache import _identity_id_for

    entry = _Entry("cred-a", SECRET, base_url="https://a.example/v1")
    pool = _Pool([entry])
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)
    _runtime(monkeypatch, provider="deepseek",
             base_url="https://api.deepseek.com/v1", api_key=SECRET)
    _http(monkeypatch, [("/user/balance", DEEPSEEK_BALANCE)])

    row = account_usage.list_account_usage_identities("deepseek")[0]
    assert row["identity_id"] == _identity_id_for("deepseek", entry)
    # The delegation roster fingerprints a fixed child key the same way, even
    # when the child has no pool object; the footer sees the identical value.
    from types import SimpleNamespace
    fixed_child_key = SimpleNamespace(runtime_api_key=SECRET, id="")
    assert _identity_id_for("deepseek", fixed_child_key) == row["identity_id"]
    snapshot = account_usage.fetch_account_usage(
        "deepseek", identity_id=row["identity_id"], read_only=True)
    assert snapshot is not None
    assert snapshot.identity_id == row["identity_id"]
    assert snapshot.profile_scope == row["profile_scope"]


def test_codex_account_id_and_legacy_fp_lookup_are_correlated(monkeypatch):
    from agent.account_usage_cache import _identity_id_for

    token = _jwt("acct-9", "user-9")
    entry = _Entry("cred-codex", token, base_url="https://chatgpt.example/backend-api/codex")
    pool = _Pool([entry])
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)

    seen = []

    def fake_fetcher(base_url=None, api_key=None):
        seen.append((base_url, api_key))
        return account_usage.AccountUsageSnapshot(
            provider="openai-codex", source="test", fetched_at=datetime.now(UTC),
            windows=(account_usage.AccountUsageWindow("Synthetic", 1.0),))

    monkeypatch.setitem(account_usage._USAGE_FETCHERS, "openai-codex", fake_fetcher)
    monkeypatch.setitem(account_usage._READ_ONLY_USAGE_FETCHERS, "openai-codex", fake_fetcher)

    snapshot = account_usage.fetch_account_usage(
        "openai-codex", identity_id="cred-codex", read_only=True)
    assert snapshot is not None
    assert snapshot.identity_id == "codex:acct-9:user-9"
    assert snapshot.account_id == "codex:acct-9:user-9"
    assert seen == [("https://chatgpt.example/backend-api/codex", token)]

    legacy = _identity_id_for("openai-codex", entry)
    assert legacy.startswith("fp:") or legacy.startswith("codex:")
    legacy_snapshot = account_usage.fetch_account_usage(
        "openai-codex", identity_id=legacy, read_only=True)
    assert legacy_snapshot is not None and legacy_snapshot.identity_id == "codex:acct-9:user-9"


def test_profile_scope_is_canonical_and_differs_per_profile(monkeypatch, tmp_path):
    first = tmp_path / "profile-a"
    second = tmp_path / "profile-b"
    first.mkdir()
    second.mkdir()
    pool = _Pool([_Entry("cred-a", SECRET)])
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)
    monkeypatch.setattr(account_usage, "resolve_runtime_provider",
                        lambda **kw: {"api_key": "", "base_url": ""})

    monkeypatch.setenv("HERMES_HOME", str(first))
    scope_a = account_usage._profile_scope()
    rows_a = account_usage.list_account_usage_identities("openrouter")
    snapshot_a = account_usage.fetch_account_usage("deepseek", read_only=True)
    monkeypatch.setenv("HERMES_HOME", str(second))
    scope_b = account_usage._profile_scope()
    rows_b = account_usage.list_account_usage_identities("openrouter")
    snapshot_b = account_usage.fetch_account_usage("deepseek", read_only=True)

    assert rows_a[0]["profile_scope"] == scope_a
    assert rows_b[0]["profile_scope"] == scope_b
    assert scope_a != scope_b
    assert str(first) not in json.dumps(rows_a)
    assert str(second) not in json.dumps(rows_b)
    # No credential configured in either synthetic home: snapshots are unknown,
    # never guessed from another profile's pool.
    assert snapshot_a is None and snapshot_b is None


def test_provider_error_is_fail_open_and_never_echoed(monkeypatch):
    _runtime(monkeypatch, provider="deepseek",
             base_url="https://api.deepseek.com/v1", api_key=SECRET)
    http = _http(monkeypatch, [("/user/balance", _Response({}, status_code=401,
                                                           url="https://api.deepseek.com/user/balance"))])

    assert account_usage.fetch_account_usage("deepseek", read_only=True) is None
    assert len(http) == 1, "a read-only read reports once; it never retries with another credential"


def test_broken_pool_store_fails_closed(monkeypatch):
    def broken(provider):
        raise RuntimeError(f"store unavailable: {SECRET}")

    monkeypatch.setattr("agent.credential_pool.load_pool", broken)
    monkeypatch.setattr("hermes_cli.auth.get_auth_status", lambda provider: {})
    assert account_usage.list_account_usage_identities("openrouter") == []
    summary = account_usage.provider_credential_summary("openrouter")
    assert summary["configured"] is False and summary["source"] == "none"
    assert SECRET not in json.dumps(summary)


def test_read_only_variants_cover_every_builtin_fetcher():
    assert set(account_usage._USAGE_FETCHERS) <= set(account_usage._READ_ONLY_USAGE_FETCHERS)
    assert "opencode-go" in account_usage._USAGE_FALLBACK_FETCHERS
    assert account_usage._USAGE_FALLBACK_FETCHERS["opencode-go"] is account_usage._fetch_opencode_go_account_usage
