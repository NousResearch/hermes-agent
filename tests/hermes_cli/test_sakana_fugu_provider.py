"""Focused tests for first-class Sakana Fugu provider wiring."""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.auth import resolve_provider
from hermes_cli.models import (
    _PROVIDER_MODELS,
    get_default_model_for_provider,
    normalize_provider,
    provider_model_ids,
)
from hermes_cli.provider_catalog import provider_catalog_by_slug
from hermes_cli.runtime_provider import resolve_runtime_provider
from providers import get_provider_profile


class TestSakanaFuguProfile:
    def test_profile_metadata_and_aliases(self):
        profile = get_provider_profile("sakana-fugu")

        assert profile is not None
        assert profile.name == "sakana-fugu"
        assert profile.display_name == "Sakana Fugu"
        assert profile.api_mode == "codex_responses"
        assert profile.base_url == "https://api.sakana.ai/v1"
        assert profile.env_vars == ("SAKANA_API_KEY", "SAKANA_BASE_URL")
        assert profile.default_aux_model == "fugu"
        assert profile.default_stale_timeout_seconds == 7200.0
        assert profile.fallback_models == ("fugu-ultra", "fugu")
        sakana_alias = get_provider_profile("sakana")
        fugu_alias = get_provider_profile("fugu")
        assert sakana_alias is not None
        assert fugu_alias is not None
        assert sakana_alias.name == "sakana-fugu"
        assert fugu_alias.name == "sakana-fugu"

    def test_reasoning_effort_uses_official_high_level_for_responses(self):
        from agent.transports.codex import ResponsesApiTransport

        profile = get_provider_profile("sakana-fugu")
        kwargs = ResponsesApiTransport().build_kwargs(
            model="fugu-ultra",
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
            reasoning_config={"enabled": True, "effort": "medium"},
            provider_profile=profile,
            base_url="https://api.sakana.ai/v1",
        )

        assert kwargs["reasoning"] == {"effort": "high"}

        kwargs = ResponsesApiTransport().build_kwargs(
            model="fugu-ultra",
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
            reasoning_config={"enabled": True, "effort": "xhigh"},
            provider_profile=profile,
            base_url="https://api.sakana.ai/v1",
        )

        assert kwargs["reasoning"] == {"effort": "high"}


class TestSakanaFuguRegistries:
    def test_aliases_resolve_across_registries(self, monkeypatch):
        monkeypatch.setenv("SAKANA_API_KEY", "sakana-test-key")

        assert resolve_provider("sakana-fugu") == "sakana-fugu"
        assert resolve_provider("sakana") == "sakana-fugu"
        assert resolve_provider("fugu") == "sakana-fugu"
        assert normalize_provider("sakana") == "sakana-fugu"
        assert normalize_provider("fugu") == "sakana-fugu"

    def test_provider_catalog_exposes_api_key_setup(self):
        descriptor = provider_catalog_by_slug()["sakana-fugu"]

        assert descriptor.label == "Sakana Fugu"
        assert descriptor.tab == "keys"
        assert descriptor.auth_type == "api_key"
        assert descriptor.api_key_env_vars == ("SAKANA_API_KEY",)
        assert descriptor.base_url_env_var == "SAKANA_BASE_URL"
        assert descriptor.signup_url == "https://console.sakana.ai/"

    def test_static_model_catalog_and_default_model(self):
        assert _PROVIDER_MODELS["sakana-fugu"] == ["fugu-ultra", "fugu"]
        assert get_default_model_for_provider("sakana-fugu") == "fugu-ultra"
        assert provider_model_ids("sakana-fugu") == ["fugu-ultra", "fugu"]

    def test_runtime_provider_uses_sakana_responses_api(self, monkeypatch):
        monkeypatch.setenv("SAKANA_API_KEY", "sakana-test-key")
        monkeypatch.delenv("SAKANA_BASE_URL", raising=False)

        resolved = resolve_runtime_provider(requested="sakana-fugu")

        assert resolved["provider"] == "sakana-fugu"
        assert resolved["api_mode"] == "codex_responses"
        assert resolved["base_url"] == "https://api.sakana.ai/v1"
        assert resolved["api_key"] == "sakana-test-key"

    def test_live_models_merge_after_curated_models(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id,
                "api_key": "sakana-live-key",
                "base_url": "https://api.sakana.ai/v1",
                "source": "SAKANA_API_KEY",
            },
        )
        with patch("providers.base.ProviderProfile.fetch_models", return_value=["fugu", "fugu-experimental"]):
            assert provider_model_ids("sakana-fugu") == [
                "fugu-ultra",
                "fugu",
                "fugu-experimental",
            ]


class TestSakanaFuguMetadata:
    def test_provider_prefix_and_endpoint_inference(self):
        from agent.model_metadata import _infer_provider_from_url, _strip_provider_prefix

        assert _strip_provider_prefix("sakana-fugu:fugu-ultra") == "fugu-ultra"
        assert _strip_provider_prefix("fugu:fugu-ultra") == "fugu-ultra"
        assert _infer_provider_from_url("https://api.sakana.ai/v1") == "sakana-fugu"


class TestSakanaFuguEnvironmentDocs:
    def test_optional_env_vars_include_sakana(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert OPTIONAL_ENV_VARS["SAKANA_API_KEY"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["SAKANA_API_KEY"]["password"] is True
        assert OPTIONAL_ENV_VARS["SAKANA_API_KEY"]["url"] == "https://console.sakana.ai/"
        assert OPTIONAL_ENV_VARS["SAKANA_BASE_URL"]["password"] is False

    def test_doctor_provider_env_hints_include_sakana(self):
        from hermes_cli.doctor import _PROVIDER_ENV_HINTS

        assert "SAKANA_API_KEY" in _PROVIDER_ENV_HINTS
