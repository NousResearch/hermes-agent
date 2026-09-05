"""Tests for the OpenAI-compatible image generation provider."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

import plugins.image_gen.openai_compatible as openai_compatible_plugin


_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


def _fake_response(*, b64=None, url=None, revised_prompt=None):
    item = SimpleNamespace(b64_json=b64, url=url, revised_prompt=revised_prompt)
    return SimpleNamespace(data=[item])


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _write_config(tmp_path: Path, config: dict) -> None:
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _fake_openai_module(fake_client: MagicMock):
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    return fake_openai


class TestMetadata:
    def test_name(self):
        provider = openai_compatible_plugin.OpenAICompatibleImageGenProvider()
        assert provider.name == "openai-compatible"
        assert provider.display_name == "OpenAI-Compatible"

    def test_default_model(self):
        provider = openai_compatible_plugin.OpenAICompatibleImageGenProvider()
        assert provider.default_model() == "gpt-image-2-medium"


class TestConfig:
    def test_env_client_config(self, monkeypatch):
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "env-key")
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "https://gateway.test/v1")
        assert openai_compatible_plugin._resolve_client_config() == (
            "env-key",
            "https://gateway.test/v1",
        )

    def test_provider_config_client_config(self, tmp_path):
        _write_config(
            tmp_path,
            {
                "image_gen": {
                    "openai_compatible": {
                        "api_key": "cfg-key",
                        "base_url": "https://gateway.test/v1",
                    }
                }
            },
        )
        assert openai_compatible_plugin._resolve_client_config() == (
            "cfg-key",
            "https://gateway.test/v1",
        )

    def test_inherits_custom_model_provider(self, tmp_path):
        _write_config(
            tmp_path,
            {
                "model": {
                    "provider": "custom:bifrost",
                    "api_key": "model-key",
                    "base_url": "https://bifrost.test/v1",
                },
                "image_gen": {"provider": "openai-compatible"},
            },
        )
        assert openai_compatible_plugin._resolve_client_config() == (
            "model-key",
            "https://bifrost.test/v1",
        )

    def test_model_resolution_from_top_level_image_model(self, tmp_path):
        _write_config(tmp_path, {"image_gen": {"model": "gpt-image-2-high"}})
        model_id, meta = openai_compatible_plugin._resolve_model()
        assert model_id == "gpt-image-2-high"
        assert meta["quality"] == "high"

    def test_api_model_override(self, tmp_path):
        _write_config(
            tmp_path,
            {"image_gen": {"openai_compatible": {"api_model": "gateway-image"}}},
        )
        assert openai_compatible_plugin._resolve_api_model() == "gateway-image"


class TestAvailability:
    def test_requires_key_and_base_url(self, monkeypatch):
        monkeypatch.delenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", raising=False)
        provider = openai_compatible_plugin.OpenAICompatibleImageGenProvider()
        assert provider.is_available() is False

    def test_available_with_key_and_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "key")
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "https://gateway.test/v1")
        provider = openai_compatible_plugin.OpenAICompatibleImageGenProvider()
        assert provider.is_available() is True


class TestGenerate:
    def test_missing_config_returns_auth_error(self):
        provider = openai_compatible_plugin.OpenAICompatibleImageGenProvider()
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_uses_openai_sdk_with_custom_base_url(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "key")
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "https://gateway.test/v1")
        _write_config(
            tmp_path,
            {
                "image_gen": {
                    "provider": "openai-compatible",
                    "model": "gpt-image-2-low",
                    "openai_compatible": {"api_model": "gpt-image-2"},
                }
            },
        )

        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        fake_openai = _fake_openai_module(fake_client)
        with patch.dict("sys.modules", {"openai": fake_openai}):
            provider = openai_compatible_plugin.OpenAICompatibleImageGenProvider()
            result = provider.generate("a cat", aspect_ratio="square")

        assert result["success"] is True
        assert result["provider"] == "openai-compatible"
        assert result["model"] == "gpt-image-2-low"
        assert result["quality"] == "low"
        assert result["api_model"] == "gpt-image-2"
        assert Path(result["image"]).exists()

        fake_openai.OpenAI.assert_called_once_with(
            api_key="key",
            base_url="https://gateway.test/v1",
        )
        call_kwargs = fake_client.images.generate.call_args.kwargs
        assert call_kwargs["model"] == "gpt-image-2"
        assert call_kwargs["quality"] == "low"
        assert call_kwargs["size"] == "1024x1024"

    def test_url_response_passes_through(self, monkeypatch):
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "key")
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "https://gateway.test/v1")
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            url="https://gateway.test/image.png"
        )

        fake_openai = _fake_openai_module(fake_client)
        with patch.dict("sys.modules", {"openai": fake_openai}):
            provider = openai_compatible_plugin.OpenAICompatibleImageGenProvider()
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["image"] == "https://gateway.test/image.png"
