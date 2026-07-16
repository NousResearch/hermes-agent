"""Tests for provider resolution fallback to the ProviderProfile plugin registry.

The step-4 fallback in resolve_provider_full() lets model-provider plugins
registered via register_provider() be discovered by /model --provider without
requiring a HERMES_OVERLAYS entry.
"""

import pytest

import providers as _providers_mod
from hermes_cli.providers import resolve_provider_full
from providers.base import ProviderProfile


def _mock_plugin_profile(name):
    return ProviderProfile(
        name="testplugin",
        display_name="Test Plugin",
        base_url="https://test.example.com/v1",
        env_vars=("TEST_KEY",),
        fallback_models=("model-a", "model-b"),
    )


def _mock_oauth_profile(name):
    return ProviderProfile(
        name="testoauth",
        base_url="https://oauth.example.com/v1",
        auth_type="oauth_device_code",
    )


def _mock_codex_profile(name):
    return ProviderProfile(
        name="testplugin",
        base_url="https://test.example.com/v1",
        api_mode="codex_responses",
    )


def _mock_minimal_profile(name):
    return ProviderProfile(
        name="minimalplugin",
        base_url="https://minimal.example.com/v1",
    )


class TestPluginProviderResolution:
    def test_plugin_provider_resolves(self, monkeypatch):
        """A registered plugin provider with auth_type=api_key is resolved."""
        monkeypatch.setattr(
            _providers_mod, "get_provider_profile",
            lambda name: _mock_plugin_profile(name) if name == "testplugin" else None,
        )

        result = resolve_provider_full("testplugin")
        assert result is not None
        assert result.id == "testplugin"
        assert result.name == "Test Plugin"
        assert result.transport == "openai_chat"
        assert result.base_url == "https://test.example.com/v1"
        assert result.api_key_env_vars == ("TEST_KEY",)
        assert result.auth_type == "api_key"
        assert result.source == "plugin"

    def test_plugin_provider_custom_api_mode(self, monkeypatch):
        """A plugin with api_mode=codex_responses maps to correct transport."""
        monkeypatch.setattr(
            _providers_mod, "get_provider_profile",
            lambda name: _mock_codex_profile(name) if name == "testplugin" else None,
        )

        result = resolve_provider_full("testplugin")
        assert result is not None
        assert result.transport == "codex_responses"

    def test_unknown_provider_returns_none(self):
        """A provider not in any registry returns None (no crash)."""
        result = resolve_provider_full("nonexistent_xyz_123456")
        assert result is None

    def test_existing_provider_still_resolves(self):
        """Built-in providers (openrouter) still resolve normally."""
        result = resolve_provider_full("openrouter")
        assert result is not None
        assert result.id == "openrouter"
        assert result.transport == "openai_chat"

    def test_plugin_oauth_provider_skipped(self, monkeypatch):
        """A plugin with auth_type=oauth_device_code is NOT resolved
        (only api_key plugins get the fallback)."""
        monkeypatch.setattr(
            _providers_mod, "get_provider_profile",
            lambda name: _mock_oauth_profile(name) if name == "testoauth" else None,
        )

        result = resolve_provider_full("testoauth")
        assert result is None

    def test_empty_display_name_falls_back_to_id(self, monkeypatch):
        """When display_name is empty, the provider id is used as name."""
        monkeypatch.setattr(
            _providers_mod, "get_provider_profile",
            lambda name: _mock_minimal_profile(name) if name == "minimalplugin" else None,
        )

        result = resolve_provider_full("minimalplugin")
        assert result is not None
        assert result.name == "minimalplugin"
