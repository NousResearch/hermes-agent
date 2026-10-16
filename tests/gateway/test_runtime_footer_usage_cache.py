"""Regression tests for runtime-footer provider/account/quota wiring helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gateway.runtime_footer_usage as usage
from agent.account_usage import AccountUsageSnapshot


def _reset_cache():
    usage.clear()


def test_footer_account_usage_cache_key_is_profile_and_credential_scoped(monkeypatch):
    monkeypatch.setattr(usage, "get_hermes_home", lambda: Path("/profiles/alice"))
    a = usage.cache_key(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="token-a",
    )
    b = usage.cache_key(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="token-b",
    )
    c = usage.cache_key(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex/",
        api_key="token-a",
    )
    monkeypatch.setattr(usage, "get_hermes_home", lambda: Path("/profiles/bob"))
    d = usage.cache_key(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="token-a",
    )

    assert a != b
    assert a == c
    assert a != d
    assert "token-a" not in repr(a)


def test_footer_account_usage_cache_key_accepts_captured_profile_home(monkeypatch):
    monkeypatch.setattr(usage, "get_hermes_home", lambda: Path("/profiles/default"))

    key = usage.cache_key(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="token-a",
        hermes_home=Path("/profiles/alice"),
    )

    assert key[0] == "/profiles/alice"


def test_cold_cache_schedules_once_and_returns_immediately(monkeypatch):
    _reset_cache()
    started = []
    monkeypatch.setattr(usage, "get_hermes_home", lambda: Path("/profiles/alice"))
    monkeypatch.setattr(usage.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        usage,
        "_start_refresh",
        lambda *args: started.append(args),
    )

    first = usage.get_cached(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )
    second = usage.get_cached(
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
    monkeypatch.setattr(usage, "get_hermes_home", lambda: Path("/profiles/alice"))
    key = usage.cache_key(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )
    usage._CACHE[key] = (100.0, snapshot)
    monkeypatch.setattr(usage.time, "monotonic", lambda: 110.0)
    monkeypatch.setattr(
        usage,
        "_start_refresh",
        lambda *_args: (_ for _ in ()).throw(AssertionError("fresh cache refreshed")),
    )

    result = usage.get_cached(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )

    assert result is snapshot
    _reset_cache()


def test_stale_cache_returns_stale_snapshot_while_refreshing(monkeypatch):
    _reset_cache()
    snapshot = SimpleNamespace(windows=(SimpleNamespace(label="Session"),))
    started = []
    monkeypatch.setattr(usage, "get_hermes_home", lambda: Path("/profiles/alice"))
    key = usage.cache_key(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )
    usage._CACHE[key] = (1.0, snapshot)
    monkeypatch.setattr(usage.time, "monotonic", lambda: 500.0)
    monkeypatch.setattr(
        usage,
        "_start_refresh",
        lambda *args: started.append(args),
    )

    result = usage.get_cached(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )

    assert result is snapshot
    assert len(started) == 1
    _reset_cache()


def test_refresh_uses_live_credential_and_updates_cache(monkeypatch):
    _reset_cache()
    calls = []
    snapshot = SimpleNamespace(windows=(SimpleNamespace(label="Session"),))
    monkeypatch.setattr(usage, "get_hermes_home", lambda: Path("/profiles/alice"))
    key = usage.cache_key(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )
    usage._REFRESHING.add(key)
    monkeypatch.setattr(
        usage,
        "fetch_account_usage",
        lambda *args, **kwargs: calls.append((args, kwargs)) or snapshot,
    )
    monkeypatch.setattr(usage.time, "monotonic", lambda: 321.0)

    usage._refresh(
        key, "openai-codex", "https://example.invalid", "runtime-token"
    )

    assert usage._CACHE[key] == (321.0, snapshot)
    assert key not in usage._REFRESHING
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
    monkeypatch.setattr(usage, "get_hermes_home", lambda: Path("/profiles/alice"))
    key = usage.cache_key(
        "openai-codex", base_url="https://example.invalid", api_key="runtime-token"
    )
    usage._CACHE[key] = (1.0, stale)
    usage._REFRESHING.add(key)
    monkeypatch.setattr(usage, "fetch_account_usage", lambda *_a, **_kw: None)
    monkeypatch.setattr(usage.time, "monotonic", lambda: 500.0)

    usage._refresh(
        key, "openai-codex", "https://example.invalid", "runtime-token"
    )

    assert usage._CACHE[key] == (500.0, stale)
    assert key not in usage._REFRESHING
    _reset_cache()
