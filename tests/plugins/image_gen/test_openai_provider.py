"""Tests for the bundled OpenAI image_gen plugin (gpt-image-2 only)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import plugins.image_gen.openai as openai_plugin


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return openai_plugin.OpenAIImageGenProvider()


def _fake_response(*, b64: str | None = None, url: str | None = None,
                   revised_prompt: str | None = None):
    item = SimpleNamespace(b64_json=b64, url=url, revised_prompt=revised_prompt)
    return SimpleNamespace(data=[item])


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "openai"

    def test_display_name(self, provider):
        assert provider.display_name == "OpenAI"

    def test_default_model(self, provider):
        assert provider.default_model() == "gpt-image-2"

    def test_list_models_just_gpt_image_2(self, provider):
        models = provider.list_models()
        assert len(models) == 1
        assert models[0]["id"] == "gpt-image-2"


class TestAvailability:
    def test_no_api_key_unavailable(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert openai_plugin.OpenAIImageGenProvider().is_available() is False

    def test_api_key_set_available(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        assert openai_plugin.OpenAIImageGenProvider().is_available() is True


class TestGenerate:
    def test_empty_prompt_rejected(self, provider):
        result = provider.generate("", aspect_ratio="square")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = openai_plugin.OpenAIImageGenProvider().generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_b64_saves_to_cache(self, provider, tmp_path):
        import base64
        png_bytes = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
            "ae426082"
        )
        b64 = base64.b64encode(png_bytes).decode()

        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=b64)

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "gpt-image-2"
        assert result["aspect_ratio"] == "landscape"
        assert result["provider"] == "openai"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        assert saved.read_bytes() == png_bytes

        call_kwargs = fake_client.images.generate.call_args.kwargs
        assert call_kwargs["model"] == "gpt-image-2"
        assert call_kwargs["size"] == "1536x1024"
        # gpt-image-2 rejects response_format — we must NOT send it.
        assert "response_format" not in call_kwargs

    @pytest.mark.parametrize("aspect,expected_size", [
        ("landscape", "1536x1024"),
        ("square", "1024x1024"),
        ("portrait", "1024x1536"),
    ])
    def test_aspect_ratio_mapping(self, provider, aspect, expected_size):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64="")

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            provider.generate("a cat", aspect_ratio=aspect)

        assert fake_client.images.generate.call_args.kwargs["size"] == expected_size

    def test_quality_override_from_config(self, provider, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"quality": "high"}}})
        )

        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64="")

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            provider.generate("a cat")

        assert fake_client.images.generate.call_args.kwargs["quality"] == "high"

    def test_quality_auto_not_sent(self, provider, tmp_path):
        """``quality: auto`` is the API default — we shouldn't send it."""
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"quality": "auto"}}})
        )

        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64="")

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            provider.generate("a cat")

        assert "quality" not in fake_client.images.generate.call_args.kwargs

    def test_revised_prompt_passed_through(self, provider):
        import base64
        png_bytes = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
            "ae426082"
        )
        b64 = base64.b64encode(png_bytes).decode()

        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=b64,
            revised_prompt="A photo of a cat",
        )

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            result = provider.generate("a cat")

        assert result["revised_prompt"] == "A photo of a cat"

    def test_api_error_returns_error_response(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.side_effect = RuntimeError("boom")

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "boom" in result["error"]

    def test_empty_response_data(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = SimpleNamespace(data=[])

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_url_fallback_if_api_changes(self, provider):
        """Defensive: if OpenAI ever returns URL instead of b64, pass through."""
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=None, url="https://example.com/img.png"
        )

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["image"] == "https://example.com/img.png"
