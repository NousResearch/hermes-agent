"""Tests for Sakana AI Fugu provider support."""

from __future__ import annotations

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
    resolve_provider,
)
from hermes_cli.models import _PROVIDER_LABELS, normalize_provider


@pytest.fixture(autouse=True)
def _clear_sakana_env(monkeypatch):
    for key in (
        "SAKANA_API_KEY",
        "SAKANA_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


class TestSakanaProviderProfile:
    def test_registered_in_provider_registry(self):
        assert "sakana" in PROVIDER_REGISTRY
        cfg = PROVIDER_REGISTRY["sakana"]
        assert cfg.name == "Sakana AI"
        assert cfg.auth_type == "api_key"
        assert cfg.inference_base_url == "https://api.sakana.ai/v1"
        assert cfg.api_key_env_vars == ("SAKANA_API_KEY",)
        assert cfg.base_url_env_var == "SAKANA_BASE_URL"

    def test_profile_fields(self):
        from providers import get_provider_profile

        profile = get_provider_profile("sakana")
        assert profile is not None
        assert profile.display_name == "Sakana AI"
        assert profile.base_url == "https://api.sakana.ai/v1"
        assert profile.fallback_models == ("fugu", "fugu-ultra")
        assert profile.default_headers["User-Agent"].startswith("HermesAgent/")


class TestSakanaAliases:
    @pytest.mark.parametrize("alias", ["sakana", "sakana-ai", "fugu"])
    def test_alias_resolves(self, alias, monkeypatch):
        monkeypatch.setenv("SAKANA_API_KEY", "sakana-test-key")
        assert resolve_provider(alias) == "sakana"

    def test_models_normalize_provider(self):
        assert normalize_provider("sakana-ai") == "sakana"
        assert normalize_provider("fugu") == "sakana"

    def test_providers_normalize_provider(self):
        from hermes_cli.providers import normalize_provider as normalize_provider_in_providers

        assert normalize_provider_in_providers("sakana-ai") == "sakana"
        assert normalize_provider_in_providers("fugu") == "sakana"


class TestSakanaCredentials:
    def test_status_configured_from_sakana_api_key(self, monkeypatch):
        monkeypatch.setenv("SAKANA_API_KEY", "sk-sakana-test")
        status = get_api_key_provider_status("sakana")
        assert status["configured"]

    def test_resolve_credentials_default_base_url(self, monkeypatch):
        monkeypatch.setenv("SAKANA_API_KEY", "sk-sakana-test")
        creds = resolve_api_key_provider_credentials("sakana")
        assert creds["api_key"] == "sk-sakana-test"
        assert creds["base_url"] == "https://api.sakana.ai/v1"

    def test_resolve_credentials_base_url_override(self, monkeypatch):
        monkeypatch.setenv("SAKANA_API_KEY", "sk-sakana-test")
        monkeypatch.setenv("SAKANA_BASE_URL", "https://sakana.example/v1")
        creds = resolve_api_key_provider_credentials("sakana")
        assert creds["base_url"] == "https://sakana.example/v1"


class TestSakanaModelCatalog:
    def test_static_model_list_fallback(self):
        from hermes_cli.models import provider_model_ids

        assert _PROVIDER_LABELS["sakana"] == "Sakana AI"
        assert provider_model_ids("sakana") == ["fugu", "fugu-ultra"]
