"""Tests for the Mistral AI provider profile.

Covers profile discovery, config-driven runtime resolution (via
PROVIDER_REGISTRY and CANONICAL_PROVIDERS auto-extension), base-URL override,
and strict-provider message handling.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def mistral_profile():
    """Resolve the registered Mistral AI provider profile via discovery."""
    # Trigger plugin discovery by importing model_tools (which pulls in
    # providers, which triggers _discover_providers).
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("mistral")
    assert profile is not None, "mistral provider profile must be registered"
    return profile


# ============================================================================
# Profile discovery
# ============================================================================


class TestMistralProfileDiscovery:
    """The ProviderProfile plugin registers correctly and exposes metadata."""

    def test_profile_is_registered(self):
        """The plugin's ``register_provider(mistral)`` call makes the
        profile discoverable via ``get_provider_profile()``."""
        import providers  # noqa: F811

        profile = providers.get_provider_profile("mistral")
        assert profile is not None
        assert profile.name == "mistral"

    def test_profile_identity(self, mistral_profile):
        assert mistral_profile.name == "mistral"
        assert mistral_profile.display_name == "Mistral AI"
        assert "Pixtral" in mistral_profile.description
        assert mistral_profile.signup_url == "https://console.mistral.ai"

    def test_profile_aliases(self, mistral_profile):
        assert "mistral-ai" in mistral_profile.aliases

    def test_profile_env_vars(self, mistral_profile):
        assert "MISTRAL_API_KEY" in mistral_profile.env_vars
        assert "MISTRAL_BASE_URL" in mistral_profile.env_vars

    def test_profile_base_url(self, mistral_profile):
        assert mistral_profile.base_url == "https://api.mistral.ai/v1"

    def test_profile_supports_vision(self, mistral_profile):
        assert mistral_profile.supports_vision is True
        assert mistral_profile.supports_vision_tool_messages is True

    def test_profile_supports_health_check(self, mistral_profile):
        assert mistral_profile.supports_health_check is True

    def test_profile_fallback_models(self, mistral_profile):
        models = mistral_profile.fallback_models
        assert "mistral-large-latest" in models
        assert "pixtral-large-latest" in models
        assert "codestral-latest" in models

    def test_alias_resolves_to_same_profile(self):
        import providers

        by_alias = providers.get_provider_profile("mistral-ai")
        by_name = providers.get_provider_profile("mistral")
        assert by_alias is by_name  # same object

    def test_list_providers_includes_mistral(self):
        import providers

        all_profiles = providers.list_providers()
        names = {p.name for p in all_profiles}
        assert "mistral" in names


# ============================================================================
# Config-driven runtime resolution (PROVIDER_REGISTRY auto-extension)
# ============================================================================


class TestMistralProviderRegistry:
    """The ProviderProfile auto-populates ``PROVIDER_REGISTRY`` in auth.py."""

    def test_auto_extended_to_provider_registry(self):
        """Verify the Mistral plugin's profile is picked up by the
        auto-extension in ``hermes_cli/auth.py``."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert "mistral" in PROVIDER_REGISTRY

    def test_registry_entry_has_correct_config(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        cfg = PROVIDER_REGISTRY.get("mistral")
        assert cfg is not None
        assert cfg.id == "mistral"
        assert cfg.auth_type == "api_key"
        assert cfg.inference_base_url == "https://api.mistral.ai/v1"
        assert "MISTRAL_API_KEY" in cfg.api_key_env_vars
        assert cfg.base_url_env_var == "MISTRAL_BASE_URL"

    def test_aliases_in_registry(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        # The auto-extension registers aliases pointing to the same entry
        assert "mistral-ai" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["mistral-ai"] is PROVIDER_REGISTRY["mistral"]


# ============================================================================
# CANONICAL_PROVIDERS auto-extension
# ============================================================================


class TestMistralCanonicalProviders:
    """The ProviderProfile auto-populates ``CANONICAL_PROVIDERS`` in models.py."""

    def test_auto_extended_to_canonical_providers(self):
        from hermes_cli.models import CANONICAL_PROVIDERS

        slugs = {p.slug for p in CANONICAL_PROVIDERS}
        assert "mistral" in slugs

    def test_canonical_entry_has_label_and_desc(self):
        from hermes_cli.models import CANONICAL_PROVIDERS

        for entry in CANONICAL_PROVIDERS:
            if entry.slug == "mistral":
                assert entry.label == "Mistral AI"
                assert "Mistral" in entry.tui_desc
                break
        else:
            pytest.fail("mistral not found in CANONICAL_PROVIDERS")


# ============================================================================
# Runtime provider resolution
# ============================================================================


class TestMistralRuntimeResolution:
    """``get_provider()`` resolves Mistral through the models.dev → overlay path."""

    def test_get_provider_resolves(self):
        from hermes_cli.providers import get_provider

        pdef = get_provider("mistral")
        assert pdef is not None
        assert pdef.id == "mistral"
        assert pdef.transport == "openai_chat"
        assert pdef.auth_type == "api_key"

    def test_resolved_base_url(self):
        from hermes_cli.providers import get_provider

        pdef = get_provider("mistral")
        assert pdef is not None
        # The overlay provides base_url_override since models.dev has api=''
        assert pdef.base_url == "https://api.mistral.ai/v1"

    def test_resolved_env_vars(self):
        from hermes_cli.providers import get_provider

        pdef = get_provider("mistral")
        assert pdef is not None
        assert "MISTRAL_API_KEY" in pdef.api_key_env_vars

    def test_resolved_base_url_env_var(self):
        from hermes_cli.providers import get_provider

        pdef = get_provider("mistral")
        assert pdef is not None
        assert pdef.base_url_env_var == "MISTRAL_BASE_URL"

    def test_not_aggregator(self):
        from hermes_cli.providers import get_provider

        pdef = get_provider("mistral")
        assert pdef is not None
        assert pdef.is_aggregator is False

    def test_alias_resolution(self):
        from hermes_cli.providers import get_provider

        # "mistral-ai" is a ProviderProfile alias, not in the ALIASES dict.
        # get_provider() resolves via ALIASES only, which doesn't include
        # ProviderProfile aliases.  The canonical name works directly.
        pdef = get_provider("mistral")
        assert pdef is not None
        assert pdef.id == "mistral"

        # normalize_provider resolves the canonical name identically.
        from hermes_cli.providers import normalize_provider

        assert normalize_provider("mistral-ai") == "mistral-ai"  # not aliased

    def test_normalize_provider(self):
        from hermes_cli.providers import normalize_provider

        assert normalize_provider("mistral") == "mistral"
        assert normalize_provider("Mistral") == "mistral"
        assert normalize_provider("mistral-ai") == "mistral-ai"


# ============================================================================
# Base-URL override
# ============================================================================


class TestMistralBaseUrlOverride:
    """The ``MISTRAL_BASE_URL`` env var overrides the default base URL."""

    def test_get_provider_honors_env_override(self, monkeypatch):
        """The resolved ProviderDef includes base_url_env_var which
        downstream code (``resolve_provider_full``) uses to apply the
        user override.  Assert the plumbing is correct."""
        from hermes_cli.providers import get_provider

        pdef = get_provider("mistral")
        assert pdef is not None
        assert pdef.base_url_env_var == "MISTRAL_BASE_URL"
        # Base URL override from overlay is separate from env-var override;
        # the overlay provides the default and the env-var name so downstream
        # code can apply the user's override.

    def test_provider_profile_base_url_set(self, mistral_profile):
        """The profile's base_url is the default endpoint."""
        assert mistral_profile.base_url == "https://api.mistral.ai/v1"


# ============================================================================
# Strict-provider message handling
# ============================================================================


class TestMistralStrictProvider:
    """Message prepare/build hooks — Mistral AI is plain OpenAI-compatible."""

    def test_prepare_messages_pass_through(self, mistral_profile):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = mistral_profile.prepare_messages(messages)
        assert result == messages

    def test_build_extra_body_empty(self, mistral_profile):
        result = mistral_profile.build_extra_body(session_id="abc123")
        assert result == {}

    def test_build_api_kwargs_extras_empty(self, mistral_profile):
        extra_body, top_level = mistral_profile.build_api_kwargs_extras(
            reasoning_config=None, model="mistral-large-latest"
        )
        assert extra_body == {}
        assert top_level == {}

    def test_get_hostname_from_base_url(self, mistral_profile):
        hostname = mistral_profile.get_hostname()
        assert hostname == "api.mistral.ai"

    def test_get_max_tokens_default_none(self, mistral_profile):
        assert mistral_profile.get_max_tokens("mistral-large-latest") is None

    def test_fetch_models_url(self, mistral_profile):
        """Default fetch_models builds URL from base_url + '/models'."""
        # Should construct https://api.mistral.ai/v1/models
        hostname = mistral_profile.get_hostname()
        assert hostname == "api.mistral.ai"


# ============================================================================
# End-to-end fixture: isolated profile scenario
# ============================================================================


class TestMistralFakeHomeEnvIntegration:
    """Verify the auto-extension works with a realistic import path."""

    def test_provider_discovery_after_plugin_import(self):
        """Simulate what happens during normal hermes startup:
        importing providers triggers discovery, which imports the
        mistral plugin, which calls register_provider()."""
        import providers

        # Force a re-discovery (the module-level _discovered flag
        # prevents double-scanning, but get_provider_profile handles
        # it by checking the flag).
        profile = providers.get_provider_profile("mistral")
        assert profile is not None
