"""Tests for DarkBloom provider support — provider-profile plugin (plugins/model-providers/darkbloom/).

DarkBloom is declared once as a ``ProviderProfile``; every layer (auth registry, picker catalog,
Settings tabs, env-var registration, URL→provider mapping) is expected to auto-wire from that
single declaration. These tests assert those contracts, not the declaration's exact field values.
"""

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
    resolve_provider,
)

_OTHER_PROVIDER_KEYS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "DASHSCOPE_API_KEY",
    "XAI_API_KEY", "KIMI_API_KEY", "KIMI_CN_API_KEY",
    "MINIMAX_API_KEY", "MINIMAX_CN_API_KEY", "AI_GATEWAY_API_KEY",
    "KILOCODE_API_KEY", "HF_TOKEN", "GLM_API_KEY", "ZAI_API_KEY",
    "XIAOMI_API_KEY", "TOKENHUB_API_KEY", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
)


# =============================================================================
# Declaration → registry
# =============================================================================


class TestDarkBloomRegistry:
    def test_registered_from_profile(self):
        assert "darkbloom" in PROVIDER_REGISTRY


    def test_inference_base_url(self):
        assert PROVIDER_REGISTRY["darkbloom"].inference_base_url == "https://api.darkbloom.dev/v1"


    def test_api_key_env_var_declared(self):
        assert "DARKBLOOM_API_KEY" in PROVIDER_REGISTRY["darkbloom"].api_key_env_vars


    def test_base_url_override_env_var_derived(self):
        """The ``*_BASE_URL`` entry of the profile's ``env_vars`` becomes the override slot."""
        assert PROVIDER_REGISTRY["darkbloom"].base_url_env_var == "DARKBLOOM_BASE_URL"


# =============================================================================
# Aliases (``provider:model`` / --provider)
# =============================================================================


class TestDarkBloomAliases:
    @pytest.mark.parametrize("alias", ["darkbloom", "dark", "db"])
    def test_alias_resolves(self, alias, monkeypatch):
        for key in _OTHER_PROVIDER_KEYS + ("OPENROUTER_API_KEY",):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("DARKBLOOM_API_KEY", "db-test-12345")
        assert resolve_provider(alias) == "darkbloom"


    def test_leading_and_trailing_space_tolerated(self, monkeypatch):
        monkeypatch.setenv("DARKBLOOM_API_KEY", "db-test-12345")
        assert resolve_provider("  darkbloom  ") == "darkbloom"


# =============================================================================
# Credentials
# =============================================================================


class TestDarkBloomCredentials:
    def test_status_configured(self, monkeypatch):
        monkeypatch.setenv("DARKBLOOM_API_KEY", "db-test")
        assert get_api_key_provider_status("darkbloom")["configured"]


    def test_openrouter_key_does_not_make_darkbloom_configured(self, monkeypatch):
        """A user with only an OpenRouter key must not see DarkBloom as configured."""
        monkeypatch.delenv("DARKBLOOM_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        assert not get_api_key_provider_status("darkbloom")["configured"]


    def test_resolve_credentials_default_endpoint(self, monkeypatch):
        monkeypatch.setenv("DARKBLOOM_API_KEY", "db-direct-key")
        monkeypatch.delenv("DARKBLOOM_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("darkbloom")
        assert creds["api_key"] == "db-direct-key"
        assert creds["base_url"] == "https://api.darkbloom.dev/v1"


    def test_resolve_credentials_base_url_override(self, monkeypatch):
        monkeypatch.setenv("DARKBLOOM_API_KEY", "db-direct-key")
        monkeypatch.setenv("DARKBLOOM_BASE_URL", "https://proxy.internal.example/v1")
        creds = resolve_api_key_provider_credentials("darkbloom")
        assert creds["base_url"] == "https://proxy.internal.example/v1"


# =============================================================================
# Picker catalog — CLI ``hermes model`` and the desktop Settings tabs
# =============================================================================


class TestDarkBloomCatalog:
    def test_in_canonical_providers(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        assert "darkbloom" in [p.slug for p in CANONICAL_PROVIDERS]


    def test_settings_descriptor_routes_to_keys_tab(self):
        from hermes_cli.provider_catalog import provider_catalog
        by_slug = {d.slug: d for d in provider_catalog()}
        assert "darkbloom" in by_slug
        d = by_slug["darkbloom"]
        assert d.auth_type == "api_key"
        assert d.tab == "keys"
        assert "DARKBLOOM_API_KEY" in d.api_key_env_vars
        assert d.label


    def test_settings_payload_lists_the_key_row(self):
        """The Keys tab renders a card per descriptor — the env var must be present in its payload."""
        from hermes_cli.web_routers.config_env import _get_env_vars_sync
        assert "DARKBLOOM_API_KEY" in _get_env_vars_sync(None)


    def test_env_var_registered_for_setup_wizard(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS
        entry = OPTIONAL_ENV_VARS.get("DARKBLOOM_API_KEY")
        assert entry is not None
        assert entry["category"] == "provider"
        assert entry["password"] is True
        assert entry["url"], "setup wizard needs a signup/console URL"


# =============================================================================
# URL → provider reverse mapping (token budgeting, trajectory compression)
# =============================================================================


class TestDarkBloomURLMapping:
    def test_hostname_maps_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER
        assert _URL_TO_PROVIDER.get("api.darkbloom.dev") == "darkbloom"


    def test_provider_prefix_registered(self):
        from agent.model_metadata import _PROVIDER_PREFIXES
        assert "darkbloom" in _PROVIDER_PREFIXES
