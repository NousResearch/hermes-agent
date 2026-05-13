"""Tests for the Cohere native provider integration.

Covers:
  - CohereProfile registration via the plugin discovery path
  - HermesOverlay entry shape (transport, env vars, base URL)
  - alias resolution: "cohere", "command", "command-r", "command-a"
  - determine_api_mode() → "cohere_chat" for the provider and the URL
  - PROVIDER_REGISTRY auto-extension picks up Cohere
"""

from __future__ import annotations

import pytest

from hermes_cli.providers import (
    HERMES_OVERLAYS,
    TRANSPORT_TO_API_MODE,
    ALIASES,
    _LABEL_OVERRIDES,
    determine_api_mode,
    get_label,
    normalize_provider,
)


# ── Overlay shape ───────────────────────────────────────────────────────


class TestCohereOverlay:
    def test_cohere_in_overlays(self):
        assert "cohere" in HERMES_OVERLAYS

    def test_cohere_transport_is_cohere_chat(self):
        assert HERMES_OVERLAYS["cohere"].transport == "cohere_chat"

    def test_cohere_env_vars(self):
        ev = HERMES_OVERLAYS["cohere"].extra_env_vars
        assert "COHERE_API_KEY" in ev
        assert "CO_API_KEY" in ev

    def test_cohere_base_url_override(self):
        assert HERMES_OVERLAYS["cohere"].base_url_override == "https://api.cohere.com"

    def test_cohere_base_url_env_var(self):
        assert HERMES_OVERLAYS["cohere"].base_url_env_var == "COHERE_BASE_URL"

    def test_cohere_label(self):
        assert _LABEL_OVERRIDES.get("cohere") == "Cohere"
        assert get_label("cohere") == "Cohere"


# ── Transport → api_mode wiring ────────────────────────────────────────


class TestTransportApiMode:
    def test_cohere_chat_in_transport_map(self):
        assert TRANSPORT_TO_API_MODE.get("cohere_chat") == "cohere_chat"


# ── Aliases ─────────────────────────────────────────────────────────────


class TestCohereAliases:
    @pytest.mark.parametrize("alias", ["command", "command-r", "command-a", "cohere-ai"])
    def test_alias_resolves_to_cohere(self, alias):
        assert ALIASES.get(alias) == "cohere"
        assert normalize_provider(alias) == "cohere"

    def test_canonical_unchanged(self):
        assert normalize_provider("cohere") == "cohere"


# ── determine_api_mode ──────────────────────────────────────────────────


class TestDetermineApiMode:
    def test_provider_cohere_returns_cohere_chat(self):
        assert determine_api_mode("cohere") == "cohere_chat"

    def test_provider_command_alias_returns_cohere_chat(self):
        assert determine_api_mode("command-r") == "cohere_chat"

    def test_url_api_cohere_com_returns_cohere_chat(self):
        assert determine_api_mode("custom", "https://api.cohere.com/v2") == "cohere_chat"

    def test_url_api_cohere_ai_legacy_host_returns_cohere_chat(self):
        assert determine_api_mode("custom", "https://api.cohere.ai/v1") == "cohere_chat"


# ── Profile registration (via providers/ discovery) ────────────────────


class TestCohereProfileRegistered:
    def test_profile_present(self):
        from providers import get_provider_profile

        profile = get_provider_profile("cohere")
        assert profile is not None
        assert profile.name == "cohere"
        assert profile.api_mode == "cohere_chat"
        assert profile.base_url == "https://api.cohere.com"
        assert "COHERE_API_KEY" in profile.env_vars

    def test_profile_aliases_route_back_to_cohere(self):
        from providers import get_provider_profile

        for alias in ("command", "command-r", "command-a", "cohere-ai"):
            profile = get_provider_profile(alias)
            assert profile is not None, f"alias {alias} should resolve"
            assert profile.name == "cohere"

    def test_profile_fallback_models_contain_chat_capable_entries(self):
        from providers import get_provider_profile

        profile = get_provider_profile("cohere")
        assert profile is not None
        models = profile.fallback_models
        assert isinstance(models, tuple)
        assert len(models) >= 1
        # Every fallback model should start with "command-" (chat-capable)
        assert all(m.startswith("command-") for m in models)


# ── PROVIDER_REGISTRY auto-extension ──────────────────────────────────


class TestPROVIDER_REGISTRY_AutoExtension:
    def test_cohere_in_provider_registry(self):
        # Force discovery first
        from providers import list_providers
        list_providers()
        from hermes_cli.auth import PROVIDER_REGISTRY
        assert "cohere" in PROVIDER_REGISTRY

    def test_cohere_provider_registry_auth_type(self):
        from providers import list_providers
        list_providers()
        from hermes_cli.auth import PROVIDER_REGISTRY
        pcfg = PROVIDER_REGISTRY["cohere"]
        assert pcfg.auth_type == "api_key"
        assert "COHERE_API_KEY" in pcfg.api_key_env_vars
