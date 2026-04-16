"""Tests for Qiniu provider support."""

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider, resolve_api_key_provider_credentials


class TestQiniuProviderRegistry:
    def test_registered(self):
        assert "qiniu" in PROVIDER_REGISTRY

    def test_registry_fields(self):
        pconfig = PROVIDER_REGISTRY["qiniu"]
        assert pconfig.name == "Qiniu"
        assert pconfig.auth_type == "api_key"
        assert pconfig.inference_base_url == "https://api.qnaigc.com/v1"
        assert pconfig.api_key_env_vars == ("QINIU_API_KEY",)
        assert pconfig.base_url_env_var == "QINIU_BASE_URL"


class TestQiniuCredentialResolution:
    def test_resolve_credentials(self, monkeypatch):
        monkeypatch.setenv("QINIU_API_KEY", "qiniu-key")
        monkeypatch.delenv("QINIU_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("qiniu")
        assert creds["api_key"] == "qiniu-key"
        assert creds["base_url"] == "https://api.qnaigc.com/v1"

    def test_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("QINIU_API_KEY", "qiniu-key")
        monkeypatch.setenv("QINIU_BASE_URL", "https://custom.qiniu.example/v1")
        creds = resolve_api_key_provider_credentials("qiniu")
        assert creds["base_url"] == "https://custom.qiniu.example/v1"


class TestQiniuModelCatalog:
    def test_static_model_fallback_exists(self):
        from hermes_cli.models import _PROVIDER_MODELS

        assert "qiniu" in _PROVIDER_MODELS
        assert _PROVIDER_MODELS["qiniu"] == ["deepseek-v3"]

    def test_canonical_provider_entry(self):
        from hermes_cli.models import CANONICAL_PROVIDERS

        slugs = [entry.slug for entry in CANONICAL_PROVIDERS]
        assert "qiniu" in slugs

    def test_provider_model_ids_prefers_live_list(self, monkeypatch):
        from hermes_cli import models as models_mod

        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda _provider: {"api_key": "qiniu-key", "base_url": "https://api.qnaigc.com/v1"},
        )
        monkeypatch.setattr(models_mod, "fetch_api_models", lambda *_args, **_kwargs: ["deepseek-v3", "deepseek-r1"])

        assert models_mod.provider_model_ids("qiniu") == ["deepseek-v3", "deepseek-r1"]


class TestQiniuNormalizationAndMetadata:
    def test_resolve_provider(self):
        assert resolve_provider("qiniu") == "qiniu"

    def test_matching_prefix_strip_provider(self):
        from hermes_cli.model_normalize import _MATCHING_PREFIX_STRIP_PROVIDERS, normalize_model_for_provider

        assert "qiniu" in _MATCHING_PREFIX_STRIP_PROVIDERS
        assert normalize_model_for_provider("qiniu/deepseek-v3", "qiniu") == "deepseek-v3"

    def test_model_metadata_url_mapping(self):
        from agent.model_metadata import _URL_TO_PROVIDER, _infer_provider_from_url

        assert _URL_TO_PROVIDER["api.qnaigc.com"] == "qiniu"
        assert _infer_provider_from_url("https://api.qnaigc.com/v1") == "qiniu"

    def test_provider_overlay(self):
        from hermes_cli.providers import HERMES_OVERLAYS, get_label

        overlay = HERMES_OVERLAYS["qiniu"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_override == "https://api.qnaigc.com/v1"
        assert overlay.base_url_env_var == "QINIU_BASE_URL"
        assert get_label("qiniu") == "Qiniu"
