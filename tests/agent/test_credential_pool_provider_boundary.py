"""Credential pools must never cross provider or custom-endpoint boundaries."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.credential_pool import (
    credential_pool_matches_provider,
    resolve_runtime_pool_key,
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


def test_named_custom_pool_match_requires_configured_identity_and_endpoint():
    configured = [
        (
            "gemini-display",
            {
                "name": "Gemini Display",
                "provider_key": "gemini-no-filter",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/",
            },
        )
    ]
    with patch("agent.credential_pool._iter_custom_providers", return_value=configured):
        assert credential_pool_matches_provider(
            "custom:gemini-display",
            "gemini-no-filter",
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )
        assert credential_pool_matches_provider(
            "custom:gemini-display",
            "custom:gemini-no-filter",
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )
        assert not credential_pool_matches_provider(
            "custom:gemini-display",
            "gemini-no-filter",
            base_url="https://fallback.example/v1",
        )
        assert not credential_pool_matches_provider(
            "custom:gemini-display",
            "custom:gemini-no-filter",
            base_url="https://fallback.example/v1",
        )
        assert not credential_pool_matches_provider(
            "custom:gemini-display",
            "other-provider",
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )


def test_runtime_pool_key_resolves_all_custom_runtime_identities():
    endpoint = "https://generativelanguage.googleapis.com/v1beta"
    configured = [
        (
            "sibling-display",
            {
                "name": "Sibling Display",
                "provider_key": "sibling-provider",
                "base_url": endpoint,
            },
        ),
        (
            "gemini-display",
            {
                "name": "Gemini Display",
                "provider_key": "gemini-no-filter",
                "base_url": endpoint,
            },
        )
    ]
    with patch("agent.credential_pool._iter_custom_providers", return_value=configured):
        assert resolve_runtime_pool_key("custom", endpoint) == "custom:sibling-display"
        assert (
            resolve_runtime_pool_key("gemini-no-filter", endpoint)
            == "custom:gemini-display"
        )
        assert (
            resolve_runtime_pool_key("custom:gemini-no-filter", endpoint)
            == "custom:gemini-display"
        )
        assert (
            resolve_runtime_pool_key(
                "gemini-no-filter",
                "https://fallback.example/v1",
            )
            == "gemini-no-filter"
        )


def test_runtime_pool_key_resolves_modern_provider_in_mixed_config():
    endpoint = "https://generativelanguage.googleapis.com/v1beta"
    config = {
        "custom_providers": [
            {
                "name": "Legacy Provider",
                "base_url": "https://legacy.example/v1",
            }
        ],
        "providers": {
            "gemini-no-filter": {
                "name": "Gemini Display",
                "api": endpoint,
            }
        },
    }

    with patch("agent.credential_pool._load_config_safe", return_value=config):
        assert (
            resolve_runtime_pool_key("gemini-no-filter", endpoint)
            == "custom:gemini-display"
        )
        assert (
            resolve_runtime_pool_key("custom:gemini-no-filter", endpoint)
            == "custom:gemini-display"
        )
        assert (
            resolve_runtime_pool_key(
                "custom:gemini-no-filter",
                "https://fallback.example/v1",
            )
            == "custom:gemini-no-filter"
        )


def test_runtime_pool_key_preserves_non_custom_identity():
    with patch("agent.credential_pool._iter_custom_providers", return_value=[]):
        assert (
            resolve_runtime_pool_key("openai-codex", "https://chatgpt.com/backend-api")
            == "openai-codex"
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