"""Credential pools must never cross provider or custom-endpoint boundaries."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.credential_pool import (
    credential_pool_matches_provider,
    get_custom_provider_pool_key,
)
from hermes_cli import runtime_provider as rp


def test_provider_match_requires_exact_non_custom_identity():
    assert credential_pool_matches_provider("deepseek", "deepseek")
    assert not credential_pool_matches_provider("openai-codex", "deepseek")
    assert not credential_pool_matches_provider("", "deepseek")


def test_custom_pool_match_is_scoped_by_endpoint():
    with patch(
        "agent.credential_pool.get_custom_provider_pool_key",
        return_value="custom:lab",
    ):
        assert credential_pool_matches_provider(
            "custom:lab", "custom", base_url="https://lab.example/v1"
        )
        assert not credential_pool_matches_provider(
            "custom:other", "custom", base_url="https://lab.example/v1"
        )


def test_pool_key_name_wins_over_shared_base_url(monkeypatch):
    """Two providers sharing one base_url with different api_keys (issue
    #81789) must resolve to their own pool keys when the name is given."""
    config = {
        "custom_providers": [
            {
                "name": "slomerex-grok",
                "base_url": "https://relay.example/v1",
                "api_key": "keyA",
            },
            {
                "name": "slomerex-alt",
                "base_url": "https://relay.example/v1",
                "api_key": "keyB",
            },
        ]
    }
    monkeypatch.setattr(
        "agent.credential_pool._load_config_safe", lambda: config
    )

    assert (
        get_custom_provider_pool_key("https://relay.example/v1", provider_name="slomerex-alt")
        == "custom:slomerex-alt"
    )
    assert (
        get_custom_provider_pool_key("https://relay.example/v1", provider_name="custom:slomerex-grok")
        == "custom:slomerex-grok"
    )
    # Slug form and raw name both match.
    assert (
        get_custom_provider_pool_key("https://relay.example/v1", provider_name="custom:slomerex-alt")
        == "custom:slomerex-alt"
    )
    # Bare base_url keeps the legacy first-owner behavior.
    assert (
        get_custom_provider_pool_key("https://relay.example/v1")
        == "custom:slomerex-grok"
    )


def test_pool_match_prefers_requested_name_over_shared_url(monkeypatch):
    """credential_pool_matches_provider must use the requested identity so the
    second provider's pool is accepted even though its URL is owned first by
    the other entry."""
    config = {
        "custom_providers": [
            {
                "name": "slomerex-grok",
                "base_url": "https://relay.example/v1",
                "api_key": "keyA",
            },
            {
                "name": "slomerex-alt",
                "base_url": "https://relay.example/v1",
                "api_key": "keyB",
            },
        ]
    }
    monkeypatch.setattr(
        "agent.credential_pool._load_config_safe", lambda: config
    )

    assert credential_pool_matches_provider(
        "custom:slomerex-alt",
        "custom",
        base_url="https://relay.example/v1",
        provider_name="custom:slomerex-alt",
    )
    # Without the name the URL lookup resolves the first owner → mismatch.
    assert not credential_pool_matches_provider(
        "custom:slomerex-alt",
        "custom",
        base_url="https://relay.example/v1",
    )


def test_runtime_ignores_pool_loaded_for_different_provider(monkeypatch):
    entry = SimpleNamespace(
        provider="openai-codex",
        access_token="wrong-token",
        runtime_api_key="wrong-token",
        runtime_base_url="https://chatgpt.com/backend-api/codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )
    pool = SimpleNamespace(
        provider="openai-codex",
        has_credentials=lambda: True,
        select=lambda: entry,
    )
    monkeypatch.setattr(rp, "load_pool", lambda _provider: pool)
    monkeypatch.setattr(rp, "resolve_provider", lambda *_a, **_kw: "deepseek")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "deepseek", "default": "deepseek-chat"},
    )
    monkeypatch.setattr(
        rp,
        "resolve_api_key_provider_credentials",
        lambda _provider: {
            "provider": "deepseek",
            "api_key": "deepseek-key",
            "base_url": "https://api.deepseek.com/v1",
            "source": "env",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="deepseek")

    assert resolved["provider"] == "deepseek"
    assert resolved["api_key"] == "deepseek-key"
    assert resolved["base_url"] == "https://api.deepseek.com/v1"