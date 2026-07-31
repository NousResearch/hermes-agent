"""Regression tests for runtime-footer provider/account/quota wiring helpers."""

from __future__ import annotations

import time
from pathlib import Path

import gateway.run as gateway_run
from agent.account_usage import AccountUsageSnapshot


def test_footer_account_usage_cache_key_is_scoped_to_credential():
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
    assert a != b
    assert a == c


def test_fetch_footer_account_usage_cached_reuses_snapshot(monkeypatch):
    gateway_run._FOOTER_ACCOUNT_USAGE_CACHE.clear()
    calls = {"n": 0}
    snapshot = AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=None,
        plan="Plus",
        windows=(),
    )

    def fake_fetch(provider, base_url=None, api_key=None):
        calls["n"] += 1
        assert provider == "openai-codex"
        assert api_key == "runtime-token"
        return snapshot

    monkeypatch.setattr(gateway_run, "fetch_account_usage", fake_fetch)

    first = gateway_run._fetch_footer_account_usage_cached(
        "openai-codex",
        base_url="https://example.invalid",
        api_key="runtime-token",
    )
    second = gateway_run._fetch_footer_account_usage_cached(
        "openai-codex",
        base_url="https://example.invalid",
        api_key="runtime-token",
    )
    assert first is snapshot
    assert second is snapshot
    assert calls["n"] == 1


def test_fetch_footer_account_usage_cached_expires(monkeypatch):
    gateway_run._FOOTER_ACCOUNT_USAGE_CACHE.clear()
    calls = {"n": 0}

    def fake_fetch(provider, base_url=None, api_key=None):
        calls["n"] += 1
        return calls["n"]

    monkeypatch.setattr(gateway_run, "fetch_account_usage", fake_fetch)
    monkeypatch.setattr(gateway_run, "_FOOTER_ACCOUNT_USAGE_TTL_SECONDS", 0.01)

    first = gateway_run._fetch_footer_account_usage_cached("deepseek", api_key="k")
    time.sleep(0.02)
    second = gateway_run._fetch_footer_account_usage_cached("deepseek", api_key="k")
    assert first == 1
    assert second == 2
    assert calls["n"] == 2


def test_run_agent_result_includes_runtime_provider_fields():
    """Static guard: both agent_result returns must expose runtime auth metadata."""
    src = Path(gateway_run.__file__).read_text()
    assert src.count('"provider": getattr(agent, "provider", None)') >= 2
    assert src.count('"base_url": getattr(agent, "base_url", None)') >= 2
    assert src.count('"api_key": getattr(agent, "api_key", None)') >= 2
    assert "_fetch_footer_account_usage_cached" in src
    assert "provider=_footer_provider" in src
    assert "account_usage=_footer_account_usage" in src
