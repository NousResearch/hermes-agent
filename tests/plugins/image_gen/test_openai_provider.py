"""Tests for the bundled OpenAI image_gen plugin (gpt-image-2, three tiers)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import plugins.image_gen.openai as openai_plugin


# 1×1 transparent PNG — valid bytes for save_b64_image()
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    import base64
    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


def _fake_response(*, b64=None, url=None, revised_prompt=None):
    item = SimpleNamespace(b64_json=b64, url=url, revised_prompt=revised_prompt)
    return SimpleNamespace(data=[item])


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return openai_plugin.OpenAIImageGenProvider()


def _patched_openai(fake_client: MagicMock):
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    return patch.dict("sys.modules", {"openai": fake_openai})


# ── Metadata ────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "openai"

    def test_default_model(self, provider):
        assert provider.default_model() == "gpt-image-2-medium"

    def test_list_models_three_tiers(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        assert ids == ["gpt-image-2-low", "gpt-image-2-medium", "gpt-image-2-high"]

    def test_catalog_entries_have_display_speed_strengths(self, provider):
        for entry in provider.list_models():
            assert entry["display"].startswith("GPT Image 2")
            assert entry["speed"]
            assert entry["strengths"]


# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_no_api_key_unavailable(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert openai_plugin.OpenAIImageGenProvider().is_available() is False

    def test_api_key_set_available(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        assert openai_plugin.OpenAIImageGenProvider().is_available() is True


# ── Model resolution ────────────────────────────────────────────────────────


class TestModelResolution:
    def test_default_is_medium(self):
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-medium"
        assert meta["quality"] == "medium"

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-2-high")
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-high"
        assert meta["quality"] == "high"

    def test_env_var_unknown_falls_back(self, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "bogus-tier")
        model_id, _ = openai_plugin._resolve_model()
        assert model_id == openai_plugin.DEFAULT_MODEL

    def test_config_openai_model(self, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"model": "gpt-image-2-low"}}})
        )
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-low"
        assert meta["quality"] == "low"

    def test_config_top_level_model(self, tmp_path):
        """``image_gen.model: gpt-image-2-high`` also works (top-level)."""
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"model": "gpt-image-2-high"}})
        )
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-high"
        assert meta["quality"] == "high"


# ── Generate ────────────────────────────────────────────────────────────────


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
        png_bytes = bytes.fromhex(_PNG_HEX)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "gpt-image-2-medium"
        assert result["aspect_ratio"] == "landscape"
        assert result["provider"] == "openai"
        assert result["quality"] == "medium"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        assert saved.read_bytes() == png_bytes

        call_kwargs = fake_client.images.generate.call_args.kwargs
        # All tiers hit the single underlying API model.
        assert call_kwargs["model"] == "gpt-image-2"
        assert call_kwargs["quality"] == "medium"
        assert call_kwargs["size"] == "1536x1024"
        # gpt-image-2 rejects response_format — we must NOT send it.
        assert "response_format" not in call_kwargs

    @pytest.mark.parametrize("tier,expected_quality", [
        ("gpt-image-2-low", "low"),
        ("gpt-image-2-medium", "medium"),
        ("gpt-image-2-high", "high"),
    ])
    def test_tier_maps_to_quality(self, provider, monkeypatch, tier, expected_quality):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", tier)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["model"] == tier
        assert result["quality"] == expected_quality
        assert fake_client.images.generate.call_args.kwargs["quality"] == expected_quality
        # Always the same underlying API model regardless of tier.
        assert fake_client.images.generate.call_args.kwargs["model"] == "gpt-image-2"

    @pytest.mark.parametrize("aspect,expected_size", [
        ("landscape", "1536x1024"),
        ("square", "1024x1024"),
        ("portrait", "1024x1536"),
    ])
    def test_aspect_ratio_mapping(self, provider, aspect, expected_size):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            provider.generate("a cat", aspect_ratio=aspect)

        assert fake_client.images.generate.call_args.kwargs["size"] == expected_size

    def test_revised_prompt_passed_through(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=_b64_png(), revised_prompt="A photo of a cat",
        )

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["revised_prompt"] == "A photo of a cat"

    def test_api_error_returns_error_response(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.side_effect = RuntimeError("boom")

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "boom" in result["error"]

    def test_empty_response_data(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = SimpleNamespace(data=[])

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_url_fallback_if_api_changes(self, provider):
        """Defensive: if OpenAI ever returns URL instead of b64, pass through."""
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=None, url="https://example.com/img.png",
        )

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["image"] == "https://example.com/img.png"
