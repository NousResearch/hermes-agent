"""Codex endpoint overrides and reused CLI agents retain credential provenance."""

import pytest

from agent.credential_pool import CredentialPool, PooledCredential
from hermes_cli import runtime_provider


@pytest.fixture
def codex_pool(monkeypatch):
    monkeypatch.setattr("agent.credential_pool.get_pool_strategy", lambda provider: "priority")
    entry = PooledCredential(
        provider="openai-codex", id="test-entry", label="Test", auth_type="oauth",
        priority=0, source="test", access_token="test-token",
        base_url=runtime_provider.DEFAULT_CODEX_BASE_URL,
    )
    pool = CredentialPool("openai-codex", [entry])
    monkeypatch.setattr(pool, "select", lambda **kwargs: entry)
    monkeypatch.setattr(runtime_provider, "load_pool", lambda provider: pool)
    monkeypatch.setattr(runtime_provider._config_mod, "load_config", dict)
    monkeypatch.setattr(runtime_provider, "_get_model_config", dict)
    monkeypatch.setattr(runtime_provider, "get_secret_str", lambda name, default="": default)
    monkeypatch.setattr(runtime_provider, "resolve_codex_runtime_credentials", lambda: {
        "api_key": "test-token", "base_url": runtime_provider.DEFAULT_CODEX_BASE_URL,
    })
    return pool


def test_canonical_codex_url_without_key_retains_pool(codex_pool):
    runtime = runtime_provider.resolve_runtime_provider(
        requested="openai-codex",
        explicit_base_url=runtime_provider.DEFAULT_CODEX_BASE_URL + "/",
        target_model="gpt-5-codex",
    )
    assert runtime["api_key"] == "test-token"
    assert runtime.get("credential_pool") is codex_pool
    assert runtime["base_url"] == runtime_provider.DEFAULT_CODEX_BASE_URL


@pytest.mark.parametrize("env_url", ["", "https://env-proxy.example.test/v1"])
def test_explicit_canonical_codex_url_beats_proxy_overrides(monkeypatch, codex_pool, env_url):
    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {
        "provider": "openai-codex", "base_url": "https://config-proxy.example.test/v1",
    })
    monkeypatch.setattr(runtime_provider, "get_secret_str", lambda name, default="": (
        env_url if name == "HERMES_CODEX_BASE_URL" else default
    ))
    runtime = runtime_provider.resolve_runtime_provider(
        requested="openai-codex",
        explicit_base_url=runtime_provider.DEFAULT_CODEX_BASE_URL + "/",
        target_model="gpt-5-codex",
    )
    assert runtime["base_url"] == runtime_provider.DEFAULT_CODEX_BASE_URL
    assert runtime["api_key"] == "test-token"
    assert runtime["credential_pool"] is codex_pool


@pytest.mark.parametrize("pool_available", [False, True])
def test_explicit_canonical_url_survives_unavailable_pool(monkeypatch, codex_pool, pool_available):
    monkeypatch.setattr(runtime_provider, "load_pool", lambda provider: codex_pool if pool_available else None)
    monkeypatch.setattr(codex_pool, "select", lambda **kwargs: None)
    monkeypatch.setattr(runtime_provider, "resolve_codex_runtime_credentials", lambda: {
        "api_key": "test-token", "base_url": "https://stored-proxy.example.test/v1",
    })
    runtime = runtime_provider.resolve_runtime_provider(
        requested="openai-codex", explicit_base_url=runtime_provider.DEFAULT_CODEX_BASE_URL,
    )
    assert runtime["base_url"] == runtime_provider.DEFAULT_CODEX_BASE_URL
    assert runtime["api_key"] == "test-token"
    assert runtime.get("credential_pool") is None


@pytest.mark.parametrize("api_key,base_url", [
    ("explicit-test-token", runtime_provider.DEFAULT_CODEX_BASE_URL),
    (None, "https://codex-proxy.example.test/v1"),
])
def test_explicit_codex_credentials_or_custom_url_still_bypass_pool(codex_pool, api_key, base_url):
    runtime = runtime_provider.resolve_runtime_provider(
        requested="openai-codex", explicit_api_key=api_key, explicit_base_url=base_url,
    )
    assert runtime["api_key"] == (api_key or "test-token")
    assert runtime["base_url"] == base_url
    assert runtime.get("credential_pool") is None


def test_unchanged_agent_rebinds_pool_and_entry_identity(monkeypatch, codex_pool):
    from cli import HermesCLI
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.api_key = "test-token"
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    shell = HermesCLI.__new__(HermesCLI)
    shell.model = "gpt-5-codex"
    shell.requested_provider = shell.provider = "openai-codex"
    shell.api_mode = "codex_responses"
    shell.api_key = agent.api_key
    shell.base_url = runtime_provider.DEFAULT_CODEX_BASE_URL
    shell.acp_command = None
    shell.acp_args = []
    shell._explicit_api_key = shell._explicit_base_url = None
    shell.agent = agent
    shell._credential_pool = None
    shell._maybe_print_free_tier_available_notice = lambda: None
    shell._normalize_model_for_provider = lambda provider: False
    runtime = dict(provider=shell.provider, api_mode=shell.api_mode,
                   api_key=shell.api_key, base_url=shell.base_url,
                   credential_pool=codex_pool, source="pool")
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda **kwargs: runtime)
    assert shell._ensure_runtime_credentials()
    assert shell.agent is agent
    assert agent._credential_pool is codex_pool
    assert agent._credential_pool_entry_id == "test-entry"
    runtime["credential_pool"] = None
    assert shell._ensure_runtime_credentials()
    assert shell.agent is agent
    assert agent._credential_pool is None
    assert agent._credential_pool_entry_id is None
