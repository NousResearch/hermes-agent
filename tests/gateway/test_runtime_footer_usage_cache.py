"""Regression tests for runtime-footer provider/account/quota wiring helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gateway.run as gateway_run
from agent.account_usage import AccountUsageSnapshot


def _reset_cache():
    with gateway_run._FOOTER_ACCOUNT_USAGE_LOCK:
        gateway_run._FOOTER_ACCOUNT_USAGE_CACHE.clear()
        gateway_run._FOOTER_ACCOUNT_USAGE_REFRESHING.clear()


def test_footer_account_usage_cache_key_is_profile_and_credential_scoped(monkeypatch):
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: Path("/profiles/alice"))
    a = gateway_run._footer_account_usage_cache_key(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="token-a",
    )
    b = gateway_run._footer_account_usage_cache_key(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="token-b",
    )
    c = gateway_run._footer_account_usage_cache_key(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex/",
        api_key="token-a",
    )
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: Path("/profiles/bob"))
    d = gateway_run._footer_account_usage_cache_key(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="token-a",
    )

    assert a != b
    assert a == c
    assert a != d
    assert "token-a" not in repr(a)


def test_cold_cache_schedules_once_and_returns_immediately(monkeypatch):
    _reset_cache()
    started = []
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: Path("/profiles/alice"))
    monkeypatch.setattr(gateway_run.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        gateway_run,
        "_start_footer_account_usage_refresh",
        lambda *args: started.append(args),
    )

    first = gateway_run._fetch_footer_account_usage_cached(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )
    second = gateway_run._fetch_footer_account_usage_cached(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )

    assert first is None
    assert second is None
    assert len(started) == 1
    assert started[0][1:] == (
        "openai-codex",
        "https://example.invalid",
        "runtime-token",
    )
    _reset_cache()


def test_fresh_cache_returns_snapshot_without_scheduling(monkeypatch):
    _reset_cache()
    snapshot = AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=None,
        plan="Plus",
        windows=(),
    )
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: Path("/profiles/alice"))
    key = gateway_run._footer_account_usage_cache_key(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )
    gateway_run._FOOTER_ACCOUNT_USAGE_CACHE[key] = (100.0, snapshot)
    monkeypatch.setattr(gateway_run.time, "monotonic", lambda: 110.0)
    monkeypatch.setattr(
        gateway_run,
        "_start_footer_account_usage_refresh",
        lambda *_args: (_ for _ in ()).throw(AssertionError("fresh cache refreshed")),
    )

    result = gateway_run._fetch_footer_account_usage_cached(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )

    assert result is snapshot
    _reset_cache()


def test_stale_cache_returns_stale_snapshot_while_refreshing(monkeypatch):
    _reset_cache()
    snapshot = SimpleNamespace(windows=(SimpleNamespace(label="Session"),))
    started = []
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: Path("/profiles/alice"))
    key = gateway_run._footer_account_usage_cache_key(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )
    gateway_run._FOOTER_ACCOUNT_USAGE_CACHE[key] = (1.0, snapshot)
    monkeypatch.setattr(gateway_run.time, "monotonic", lambda: 500.0)
    monkeypatch.setattr(
        gateway_run,
        "_start_footer_account_usage_refresh",
        lambda *args: started.append(args),
    )

    result = gateway_run._fetch_footer_account_usage_cached(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )

    assert result is snapshot
    assert len(started) == 1
    _reset_cache()


def test_refresh_uses_live_credential_and_updates_cache(monkeypatch):
    _reset_cache()
    calls = []
    snapshot = SimpleNamespace(windows=(SimpleNamespace(label="Session"),))
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: Path("/profiles/alice"))
    key = gateway_run._footer_account_usage_cache_key(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )
    gateway_run._FOOTER_ACCOUNT_USAGE_REFRESHING.add(key)
    monkeypatch.setattr(
        gateway_run,
        "fetch_account_usage",
        lambda *args, **kwargs: calls.append((args, kwargs)) or snapshot,
    )
    monkeypatch.setattr(gateway_run.time, "monotonic", lambda: 321.0)

    gateway_run._refresh_footer_account_usage(
        key, "openai-codex", "https://example.invalid", "runtime-token"
    )

    assert gateway_run._FOOTER_ACCOUNT_USAGE_CACHE[key] == (321.0, snapshot)
    assert key not in gateway_run._FOOTER_ACCOUNT_USAGE_REFRESHING
    assert calls == [
        (
            ("openai-codex",),
            {"base_url": "https://example.invalid", "api_key": "runtime-token"},
        )
    ]
    _reset_cache()


def test_refresh_failure_preserves_last_good_snapshot(monkeypatch):
    _reset_cache()
    stale = SimpleNamespace(windows=(SimpleNamespace(label="Session"),))
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: Path("/profiles/alice"))
    key = gateway_run._footer_account_usage_cache_key(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )
    gateway_run._FOOTER_ACCOUNT_USAGE_CACHE[key] = (1.0, stale)
    gateway_run._FOOTER_ACCOUNT_USAGE_REFRESHING.add(key)
    monkeypatch.setattr(gateway_run, "fetch_account_usage", lambda *_a, **_kw: None)
    monkeypatch.setattr(gateway_run.time, "monotonic", lambda: 500.0)

    gateway_run._refresh_footer_account_usage(
        key, "openai-codex", "https://example.invalid", "runtime-token"
    )

    assert gateway_run._FOOTER_ACCOUNT_USAGE_CACHE[key] == (500.0, stale)
    assert key not in gateway_run._FOOTER_ACCOUNT_USAGE_REFRESHING
    _reset_cache()
