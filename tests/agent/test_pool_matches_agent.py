"""Relayer custom-pool identity through real agent construction and recovery."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from agent.credential_pool import (
    CredentialPool,
    PooledCredential,
)
from agent.credential_pool_identity import credential_pool_matches_provider, resolve_runtime_pool_key
from agent.error_classifier import FailoverReason

RELAYER_URL = "https://relayer.example/v1"
CLAUDE_URL = "https://claude.example/v1"
OTHER_URL = "https://minimax.example/v1"


@pytest.fixture(autouse=True)
def custom_config(monkeypatch):
    # conftest isolates HERMES_HOME; use actual config loading/name resolution.
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home())
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "custom_providers:\n"
        "  - name: Claude\n"
        f"    base_url: {CLAUDE_URL}\n"
        "  - name: Minimax\n"
        f"    base_url: {OTHER_URL}\n"
        "agent:\n"
        "  skip_background_review: true\n"
        "model:\n"
        "  context_length: 128000\n",
        encoding="utf-8",
    )


def make_pool(base_url=RELAYER_URL):
    entries = [PooledCredential.from_dict("custom:claude", {
        "id": f"key-{i}", "access_token": f"test-key-{i}",
        "base_url": base_url, "priority": i,
    }) for i in range(2)]
    pool = CredentialPool("custom:claude", entries)
    pool.select()
    return pool


def make_agent(monkeypatch, pool, requested_provider="custom:claude", base_url=RELAYER_URL):
    from run_agent import AIAgent

    # No mocked __init__, init_agent, routing guard, config resolver or pool.
    # Only the external client/tool discovery is stubbed to keep this offline.
    client = MagicMock()
    client.api_key = "test-key-0"
    client.base_url = base_url
    client._default_headers = None
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **kwargs: client)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *a, **kw: {})
    return AIAgent(
        provider="custom", requested_provider=requested_provider,
        base_url=base_url, api_key="test-key-0", model="test-model",
        api_mode="chat_completions", credential_pool=pool,
        enabled_toolsets=[], quiet_mode=True, skip_memory=True,
        skip_context_files=True, skip_background_review=True,
    )


def test_relayer_construction_keeps_pool_and_rotates(monkeypatch):
    pool = make_pool()
    agent = make_agent(monkeypatch, pool)
    assert agent.requested_provider == "custom:claude"
    assert agent._credential_pool is pool

    recovered, retried = agent._recover_with_credential_pool(
        status_code=429, has_retried_429=True,
        classified_reason=FailoverReason.rate_limit,
    )
    assert recovered is True
    assert retried is False
    assert agent.api_key == "test-key-1"
    assert agent.base_url == RELAYER_URL
    assert pool.current().id == "key-1"
    assert pool.entries()[0].last_status == "exhausted"


def test_relayer_pool_survives_fallback_to_next_turn_restore(monkeypatch):
    pool = make_pool()
    agent = make_agent(monkeypatch, pool)
    assert agent._primary_runtime["requested_provider"] == "custom:claude"

    # Model the state left by a cross-provider fallback at the end of a turn.
    agent._fallback_activated = True
    agent._provider_fallback_active = True
    agent.provider = "openrouter"
    agent.requested_provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"

    assert agent._restore_primary_runtime() is True
    assert agent.provider == "custom"
    assert agent.requested_provider == "custom:claude"
    assert agent.base_url == RELAYER_URL
    assert agent._credential_pool is pool
    assert agent._credential_pool.provider == "custom:claude"


@pytest.mark.parametrize("requested_provider", ["custom:minimax", "", None])
def test_relayer_construction_drops_unidentified_or_mismatched_pool(monkeypatch, requested_provider):
    pool = make_pool()
    agent = make_agent(monkeypatch, pool, requested_provider=requested_provider)
    assert agent._credential_pool is None
    assert all(entry.last_status is None for entry in pool.entries())


def test_relayer_rotation_still_rejects_entry_for_another_endpoint(monkeypatch):
    pool = make_pool()
    pool.entries()[1].base_url = OTHER_URL
    agent = make_agent(monkeypatch, pool)
    assert agent._credential_pool is pool
    recovered, _ = agent._recover_with_credential_pool(
        status_code=429, has_retried_429=True,
        classified_reason=FailoverReason.rate_limit,
    )
    assert recovered is False
    assert agent.api_key == "test-key-0"
    assert agent.base_url == RELAYER_URL


@pytest.mark.parametrize("requested", ["custom:claude", "  CUSTOM:CLAUDE  "])
def test_exact_named_relayer_match(requested):
    assert credential_pool_matches_provider(
        "custom:claude", "custom", base_url=RELAYER_URL,
        requested_provider=requested,
    )


def test_exact_named_relayer_identity_resolves_pool_key():
    assert resolve_runtime_pool_key(
        "custom",
        RELAYER_URL,
        requested_provider="custom:claude",
    ) == "custom:claude"


@pytest.mark.parametrize("requested,url", [
    ("custom:minimax", RELAYER_URL),
    ("custom:claude", OTHER_URL),
])
def test_named_identity_does_not_resolve_across_pool_boundaries(requested, url):
    assert resolve_runtime_pool_key(
        "custom",
        url,
        requested_provider=requested,
    ) != "custom:claude"


@pytest.mark.parametrize("provider,pool,requested,url", [
    ("custom", "custom:claude", "custom:minimax", RELAYER_URL),
    ("custom", "custom:claude", None, RELAYER_URL),
    ("custom", "custom:claude", "custom:claude-other", RELAYER_URL),
    ("custom", "custom:claude", "claude", RELAYER_URL),
    ("custom", "custom:claude", "custom:claude", ""),
    ("", "custom:claude", "custom:claude", RELAYER_URL),
    ("openai-codex", "custom:claude", "custom:claude", RELAYER_URL),
    ("custom:claude", "custom:claude", "custom:claude", RELAYER_URL),
    ("custom-other", "custom:claude", "custom:claude", RELAYER_URL),
    ("custom", "custom:claude", "custom:claude", OTHER_URL),
    ("custom", "custom:", "custom:", RELAYER_URL),
    ("custom", "", "custom:claude", RELAYER_URL),
    ("custom", "claude", "claude", RELAYER_URL),
])
def test_named_identity_does_not_override_other_boundaries(provider, pool, requested, url):
    assert not credential_pool_matches_provider(
        pool, provider, base_url=url, requested_provider=requested,
    )


def test_existing_endpoint_and_builtin_matches_unchanged():
    assert credential_pool_matches_provider("custom:claude", "custom", base_url=CLAUDE_URL)
    assert credential_pool_matches_provider("anthropic", "anthropic", requested_provider="custom:claude")
    assert credential_pool_matches_provider(SimpleNamespace(), "custom", base_url=RELAYER_URL)


@pytest.mark.parametrize("provider,url", [
    ("openrouter", "https://openrouter.ai/api/v1"),
    ("custom", OTHER_URL),
])
def test_recovery_rechecks_current_provider_after_construction(monkeypatch, provider, url):
    pool = make_pool()
    agent = make_agent(monkeypatch, pool)
    # A fallback must not mutate the original pool, even with stale requested identity.
    agent.provider = provider
    agent.base_url = url
    recovered, retried = agent._recover_with_credential_pool(
        status_code=429, has_retried_429=True,
        classified_reason=FailoverReason.rate_limit,
    )
    assert recovered is False
    assert retried is True
    assert all(entry.last_status is None for entry in pool.entries())


def test_requested_identity_does_not_bypass_config_lookup_failure(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("config unavailable")

    monkeypatch.setattr("agent.credential_pool.get_custom_provider_pool_key", unavailable)
    assert not credential_pool_matches_provider(
        "custom:claude", "custom", base_url=RELAYER_URL,
        requested_provider="custom:claude",
    )
