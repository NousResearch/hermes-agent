"""Unit tests for the LongCat provider profile and its CLI wiring.

LongCat (Meituan) exposes ``LongCat-2.0`` over an OpenAI-compatible Chat
Completions API. It is a reasoning model: every response also carries a
``reasoning_content`` field. That field is additive, so the standard
OpenAI-compatible path handles it without special transport wiring, and
multi-turn requests do not need ``reasoning_content`` echoed back. These
tests pin the profile metadata and the provider resolution so the
``longcat`` provider keeps resolving through the standard api-key path.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def longcat_profile():
    """Resolve the registered LongCat profile through the global registry.

    Importing ``model_tools`` triggers plugin discovery, which is what
    registers the LongCat profile.
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("longcat")
    assert profile is not None, "longcat provider profile must be registered"
    return profile


class TestLongCatProfileMetadata:
    def test_identity(self, longcat_profile):
        assert longcat_profile.name == "longcat"
        assert longcat_profile.display_name == "LongCat"

    def test_base_url(self, longcat_profile):
        assert longcat_profile.base_url == "https://api.longcat.chat/openai/v1"

    def test_auth_type(self, longcat_profile):
        assert longcat_profile.auth_type == "api_key"

    def test_env_vars(self, longcat_profile):
        assert longcat_profile.env_vars == ("LONGCAT_API_KEY", "LONGCAT_BASE_URL")

    def test_aliases(self, longcat_profile):
        assert "long-cat" in longcat_profile.aliases
        assert "meituan-longcat" in longcat_profile.aliases

    def test_aux_model(self, longcat_profile):
        assert longcat_profile.default_aux_model == "LongCat-2.0"

    def test_fallback_models(self, longcat_profile):
        assert longcat_profile.fallback_models == ("LongCat-2.0",)


class TestLongCatAuxConsumer:
    """The aux helper returns LongCat's single model rather than empty."""

    def test_consumer_api_returns_longcat_2(self):
        from agent.auxiliary_client import _get_aux_model_for_provider

        assert _get_aux_model_for_provider("longcat") == "LongCat-2.0"


class TestLongCatProviderRegistry:
    """The overlay surfaces LongCat as a simple api-key provider."""

    def test_registered(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert "longcat" in PROVIDER_REGISTRY

    def test_env_vars(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY["longcat"]
        assert pconfig.api_key_env_vars == ("LONGCAT_API_KEY",)
        assert pconfig.base_url_env_var == "LONGCAT_BASE_URL"

    def test_inference_base_url(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert (
            PROVIDER_REGISTRY["longcat"].inference_base_url
            == "https://api.longcat.chat/openai/v1"
        )


class TestLongCatResolution:
    def test_explicit(self):
        from hermes_cli.auth import resolve_provider

        assert resolve_provider("longcat") == "longcat"

    @pytest.mark.parametrize("alias", ["long-cat", "meituan-longcat"])
    def test_aliases_resolve(self, alias):
        from hermes_cli.auth import resolve_provider

        assert resolve_provider(alias) == "longcat"

    def test_auto_detects_longcat_key(self, monkeypatch):
        from hermes_cli.auth import resolve_provider

        # Isolate detection on LongCat's key only.
        for noise in (
            "OPENROUTER_API_KEY",
            "STEPFUN_API_KEY",
            "GMI_API_KEY",
            "NVIDIA_API_KEY",
            "MINIMAX_API_KEY",
            "DEEPSEEK_API_KEY",
        ):
            monkeypatch.delenv(noise, raising=False)
        monkeypatch.setenv("LONGCAT_API_KEY", "test-longcat-key")
        assert resolve_provider("auto") == "longcat"

    def test_resolve_credentials_uses_default_base_url(self, monkeypatch):
        from hermes_cli.auth import resolve_api_key_provider_credentials

        monkeypatch.setenv("LONGCAT_API_KEY", "longcat-secret-key")
        creds = resolve_api_key_provider_credentials("longcat")
        assert creds["provider"] == "longcat"
        assert creds["api_key"] == "longcat-secret-key"
        assert creds["base_url"] == "https://api.longcat.chat/openai/v1"

    def test_resolve_credentials_custom_base_url(self, monkeypatch):
        from hermes_cli.auth import resolve_api_key_provider_credentials

        monkeypatch.setenv("LONGCAT_API_KEY", "longcat-key")
        monkeypatch.setenv("LONGCAT_BASE_URL", "https://custom.longcat.example/v1")
        creds = resolve_api_key_provider_credentials("longcat")
        assert creds["base_url"] == "https://custom.longcat.example/v1"


class TestLongCatHostMapping:
    """The LongCat host maps back to the provider for metadata lookups."""

    def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER

        assert _URL_TO_PROVIDER.get("api.longcat.chat") == "longcat"
