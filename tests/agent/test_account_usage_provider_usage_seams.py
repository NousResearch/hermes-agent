"""Additive account-usage seams for Provider Usage's credential migration.

Covers the two pieces the plugin cannot implement without reading a credential
store itself:

* OpenRouter snapshots carry the exact decoded bodies on ``raw`` (free-model
  daily quota, per-period key spend, BYOK) and have a read-only fetcher.
* Pool accounts are listed as non-secret identities and a per-identity
  ``fetch_account_usage`` resolves the pool row inside core.

No real credential, network call, or auth store is used.
"""

from datetime import UTC, datetime

import pytest

from agent import account_usage


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _RoutingClient:
    def __init__(self, routes):
        self._routes = routes
        self.seen = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        self.seen.append((url, dict(headers or {})))
        return _Response(self._routes[url])


def _entry(ident: str, token: str, *, status: str = "ok", label: str = "Account",
           base_url: str = "https://pool.example/v1", request_count: int = 3):
    class Entry:
        pass

    entry = Entry()
    entry.id = ident
    entry.label = label
    entry.auth_type = "api_key"
    entry.source = "manual"
    entry.access_token = token
    entry.runtime_api_key = token
    entry.runtime_base_url = base_url
    entry.last_status = status
    entry.request_count = request_count
    entry.last_status_at = 1700000000.0
    return entry


class _Pool:
    def __init__(self, entries):
        self._entries = list(entries)

    def entries(self):
        return list(self._entries)


def test_openrouter_raw_keeps_credits_key_and_free_quota(monkeypatch):
    client = _RoutingClient({
        "https://openrouter.ai/api/v1/credits": {
            "data": {"total_credits": 100.0, "total_usage": 87.5}},
        "https://openrouter.ai/api/v1/key": {
            "data": {
                "limit": 50.0,
                "limit_remaining": 12.5,
                "limit_reset": "monthly",
                "usage": 87.5,
                "usage_daily": 1.25,
                "usage_weekly": 9.5,
                "usage_monthly": 40.0,
                "byok_usage": 4.0,
                "free_model_daily_requests": {"limit": 1000, "remaining": 250},
            }},
    })
    monkeypatch.setattr(account_usage, "httpx", type("H", (), {"Client": lambda timeout=10.0: client}))
    monkeypatch.setattr(
        account_usage, "resolve_runtime_provider",
        lambda requested, explicit_base_url=None, explicit_api_key=None: {
            "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-synthetic-openrouter"})

    snapshot = account_usage.fetch_account_usage("openrouter", read_only=True)

    assert snapshot is not None
    assert snapshot.raw == {
        "credits": {"total_credits": 100.0, "total_usage": 87.5},
        "key": {
            "limit": 50.0,
            "limit_remaining": 12.5,
            "limit_reset": "monthly",
            "usage": 87.5,
            "usage_daily": 1.25,
            "usage_weekly": 9.5,
            "usage_monthly": 40.0,
            "byok_usage": 4.0,
            "free_model_daily_requests": {"limit": 1000, "remaining": 250},
        },
    }
    assert "openrouter" in account_usage._READ_ONLY_USAGE_FETCHERS
    assert snapshot.identity_id is None, "env/config OpenRouter keys are unknown"
    assert snapshot.profile_scope.startswith("home:")
    assert "sk-synthetic-openrouter" not in str(snapshot.raw)
    assert "authorization" not in str(snapshot.raw).lower()


def test_list_account_usage_identities_is_nonsecret_and_per_pool_entry(monkeypatch):
    pool = _Pool([
        _entry("one", "sk-pool-one", label="Work key", request_count=7),
        _entry("two", "sk-pool-two", status="exhausted", label="Personal key"),
        _entry("three", "sk-pool-three", status="dead", label="Retired key"),
    ])
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)

    rows = account_usage.list_account_usage_identities("openrouter")

    assert [row["label"] for row in rows] == ["Work key", "Personal key", "Retired key"]
    assert [row["available"] for row in rows] == [True, False, False]
    assert [row["status"] for row in rows] == ["ok", "exhausted", "dead"]
    assert [row["request_count"] for row in rows] == [7, 3, 3]
    assert [row["credential_id"] for row in rows] == ["one", "two", "three"]
    from agent.account_usage_cache import _identity_id_for
    expected = [_identity_id_for("openrouter", entry) for entry in pool.entries()]
    assert [row["identity_id"] for row in rows] == expected
    assert all(row["identity_id"].startswith("fp:") for row in rows)
    assert all(row["profile_scope"].startswith("home:") for row in rows)
    assert all(row["account_id"] is None for row in rows)
    text = str(rows)
    assert "sk-pool" not in text
    assert "access_token" not in text and "refresh_token" not in text


def test_fetch_account_usage_resolves_identity_inside_core(monkeypatch):
    entry = _entry("one", "sk-pool-one", label="Work key")
    monkeypatch.setattr("agent.credential_pool.load_pool",
                        lambda provider: _Pool([entry, _entry("two", "sk-pool-two")]))
    identity = account_usage.list_account_usage_identities("openrouter")[0]["identity_id"]

    seen = []

    def fake_fetcher(base_url=None, api_key=None):
        seen.append((base_url, api_key))
        return account_usage.AccountUsageSnapshot(
            provider="openrouter", source="test", fetched_at=datetime.now(UTC),
            windows=(account_usage.AccountUsageWindow("Synthetic", 1.0),))

    monkeypatch.setitem(account_usage._USAGE_FETCHERS, "openrouter", fake_fetcher)
    monkeypatch.setitem(account_usage._READ_ONLY_USAGE_FETCHERS, "openrouter", fake_fetcher)

    snapshot = account_usage.fetch_account_usage(
        "openrouter", identity_id=identity, read_only=True)

    assert snapshot is not None
    assert seen == [("https://pool.example/v1", "sk-pool-one")]
    assert "sk-pool-one" not in str(account_usage.list_account_usage_identities("openrouter"))

    assert account_usage.fetch_account_usage(
        "openrouter", identity_id="fp:missing", read_only=True) is None
    assert len(seen) == 1


def test_explicit_api_key_still_wins_over_identity(monkeypatch):
    monkeypatch.setattr("agent.credential_pool.load_pool",
                        lambda provider: _Pool([_entry("one", "sk-pool-one")]))
    identity = account_usage.list_account_usage_identities("openrouter")[0]["identity_id"]
    seen = []

    def fake_fetcher(base_url=None, api_key=None):
        seen.append((base_url, api_key))
        return account_usage.AccountUsageSnapshot(
            provider="openrouter", source="test", fetched_at=datetime.now(UTC),
            windows=(account_usage.AccountUsageWindow("Synthetic", 1.0),))

    monkeypatch.setitem(account_usage._USAGE_FETCHERS, "openrouter", fake_fetcher)

    account_usage.fetch_account_usage(
        "openrouter", identity_id=identity, api_key="explicit-key",
        base_url="https://explicit.example/v1")

    assert seen == [("https://explicit.example/v1", "explicit-key")]
