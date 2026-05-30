"""Tests for Yandex AI Studio provider support."""

from __future__ import annotations

import pytest

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider
from plugins.model_providers.yandex import normalize_yandex_model


class TestYandexProviderRegistry:
    def test_registered(self):
        assert "yandex" in PROVIDER_REGISTRY

    def test_inference_base_url(self):
        assert (
            PROVIDER_REGISTRY["yandex"].inference_base_url
            == "https://llm.api.cloud.yandex.net/v1"
        )

    def test_api_key_env_vars(self):
        assert "YANDEX_API_KEY" in PROVIDER_REGISTRY["yandex"].api_key_env_vars


class TestYandexAliases:
    @pytest.mark.parametrize("alias", ["yandex", "yandex-ai-studio", "yandex-aistudio"])
    def test_alias_resolves(self, alias, monkeypatch):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key-1234567890123456")
        monkeypatch.setenv("YANDEX_FOLDER_ID", "b1folder")
        assert resolve_provider(alias) == "yandex"


class TestNormalizeYandexModel:
    def test_bare_model_becomes_gpt_uri(self, monkeypatch):
        monkeypatch.setenv("YANDEX_FOLDER_ID", "b1folder")
        assert (
            normalize_yandex_model("deepseek-v4-flash/latest")
            == "gpt://b1folder/deepseek-v4-flash/latest"
        )

    def test_expands_folder_env_placeholder(self, monkeypatch):
        monkeypatch.setenv("YANDEX_FOLDER_ID", "b1folder")
        assert (
            normalize_yandex_model("gpt://${YANDEX_FOLDER_ID}/deepseek-v4-flash/latest")
            == "gpt://b1folder/deepseek-v4-flash/latest"
        )

    def test_passthrough_existing_gpt_uri(self, monkeypatch):
        monkeypatch.setenv("YANDEX_FOLDER_ID", "b1folder")
        uri = "gpt://b1folder/deepseek-v4-flash/latest"
        assert normalize_yandex_model(uri) == uri

    def test_missing_folder_leaves_placeholder(self, monkeypatch):
        monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
        raw = "gpt://${YANDEX_FOLDER_ID}/deepseek-v4-flash/latest"
        assert normalize_yandex_model(raw) == raw

    def test_model_normalize_integration(self, monkeypatch):
        from hermes_cli.model_normalize import normalize_model_for_provider

        monkeypatch.setenv("YANDEX_FOLDER_ID", "b1folder")
        assert (
            normalize_model_for_provider("deepseek-v4-flash/latest", "yandex")
            == "gpt://b1folder/deepseek-v4-flash/latest"
        )


class TestYandexProfileHooks:
    def test_no_thinking_extra_body(self):
        from providers import get_provider_profile

        profile = get_provider_profile("yandex")
        extra, top = profile.build_api_kwargs_extras(
            model="gpt://b1folder/deepseek-v4-flash/latest",
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        assert extra == {}
        assert top == {}

    def test_dynamic_folder_header(self, monkeypatch):
        from providers import get_provider_profile

        monkeypatch.setenv("YANDEX_FOLDER_ID", "b1folder")
        profile = get_provider_profile("yandex")
        assert profile.default_headers["x-folder-id"] == "b1folder"
        assert profile.default_headers["x-data-logging-enabled"] == "false"
