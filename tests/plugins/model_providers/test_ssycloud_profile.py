"""Tests for the SSYCloud (胜算云) model-provider plugin."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_user_config(tmp_path, monkeypatch):
    """Keep developer credentials and user plugins out of these tests."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("SSYCLOUD_API_KEY", raising=False)


@pytest.fixture
def ssycloud_profile():
    """Resolve SSYCloud through the real provider discovery path."""
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("ssycloud")
    assert profile is not None, "SSYCloud provider profile must be registered"
    return profile


class TestSSYCloudProfile:
    def test_identity_endpoint_and_auth(self, ssycloud_profile):
        assert ssycloud_profile.name == "ssycloud"
        assert ssycloud_profile.api_mode == "chat_completions"
        assert ssycloud_profile.auth_type == "api_key"
        assert ssycloud_profile.base_url == "https://router.shengsuanyun.com/api/v1"
        assert ssycloud_profile.get_hostname() == "router.shengsuanyun.com"
        assert ssycloud_profile.env_vars == ("SSYCLOUD_API_KEY",)

    @pytest.mark.parametrize("alias", ["shengsuanyun", "ssy-cloud"])
    def test_aliases_resolve(self, ssycloud_profile, alias):
        import providers

        assert providers.get_provider_profile(alias) is ssycloud_profile

    def test_fallback_models_keep_native_vendor_prefix(self, ssycloud_profile):
        assert ssycloud_profile.fallback_models
        assert all("/" in model for model in ssycloud_profile.fallback_models)


class TestSSYCloudAutoWiring:
    def test_auth_registry_and_credentials(self, monkeypatch):
        monkeypatch.setenv("SSYCLOUD_API_KEY", "ssy-test-key")

        from hermes_cli.auth import (
            PROVIDER_REGISTRY,
            resolve_api_key_provider_credentials,
        )

        config = PROVIDER_REGISTRY["ssycloud"]
        assert config.api_key_env_vars == ("SSYCLOUD_API_KEY",)
        assert config.base_url_env_var == ""

        credentials = resolve_api_key_provider_credentials("ssycloud")
        assert credentials["api_key"] == "ssy-test-key"
        assert credentials["base_url"] == "https://router.shengsuanyun.com/api/v1"

    def test_catalog_and_overlay_treat_ssycloud_as_aggregator(self):
        from hermes_cli.models import (
            CANONICAL_PROVIDERS,
            _AGGREGATOR_PROVIDERS,
            normalize_provider as normalize_model_provider,
        )
        from hermes_cli.providers import (
            HERMES_OVERLAYS,
            is_aggregator,
            normalize_provider as normalize_runtime_provider,
        )

        assert "ssycloud" in {entry.slug for entry in CANONICAL_PROVIDERS}
        assert "ssycloud" in _AGGREGATOR_PROVIDERS
        assert HERMES_OVERLAYS["ssycloud"].is_aggregator is True
        assert is_aggregator("ssycloud") is True
        assert normalize_model_provider("shengsuanyun") == "ssycloud"
        assert normalize_runtime_provider("shengsuanyun") == "ssycloud"

    def test_model_normalization_preserves_router_slug(self):
        from hermes_cli.model_normalize import normalize_model_for_provider

        assert normalize_model_for_provider(
            "deepseek/deepseek-v4-flash", "ssycloud"
        ) == "deepseek/deepseek-v4-flash"
