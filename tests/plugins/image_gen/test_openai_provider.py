"""Tests for the bundled OpenAI image_gen plugin (gpt-image-1/1.5/2 models)."""

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

    def test_list_models_includes_all_models(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        # Must include all 8 entries: 3 gpt-image-2 tiers + 3 gpt-image-1.5 tiers + gpt-image-1 + gpt-image-1-mini
        assert "gpt-image-2-low" in ids
        assert "gpt-image-2-medium" in ids
        assert "gpt-image-2-high" in ids
        assert "gpt-image-1.5-low" in ids
        assert "gpt-image-1.5-medium" in ids
        assert "gpt-image-1.5-high" in ids
        assert "gpt-image-1" in ids
        assert "gpt-image-1-mini" in ids
        assert len(ids) == 8

    def test_catalog_entries_have_required_fields(self, provider):
        for entry in provider.list_models():
            assert entry["display"]
            assert entry["speed"]
            assert entry["strengths"]
            assert entry["price"]


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
        assert meta["api_model"] == "gpt-image-2"

    def test_env_var_override_tier_id(self, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-2-high")
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-high"
        assert meta["quality"] == "high"
        assert meta["api_model"] == "gpt-image-2"

    def test_env_var_override_bare_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-1"
        assert meta["api_model"] == "gpt-image-1"
        assert "quality" not in meta  # gpt-image-1 has no quality tiers

    def test_env_var_override_unknown_model_passthrough(self, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "some-future-model")
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "some-future-model"
        assert meta["api_model"] == "some-future-model"
        assert "quality" not in meta

    def test_env_var_override_empty_falls_back(self, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "")
        # Empty string env var should not override; falls back to default
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-medium"

    def test_config_openai_model(self, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"model": "gpt-image-2-low"}}})
        )
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-low"
        assert meta["quality"] == "low"
        assert meta["api_model"] == "gpt-image-2"

    def test_config_top_level_model(self, tmp_path):
        """``image_gen.model: gpt-image-2-high`` also works (top-level)."""
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"model": "gpt-image-2-high"}})
        )
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-high"
        assert meta["quality"] == "high"

    def test_config_bare_model_name(self, tmp_path):
        """Bare model names like ``gpt-image-1`` pass through config."""
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"model": "gpt-image-1"}})
        )
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-1"
        assert meta["api_model"] == "gpt-image-1"
        assert "quality" not in meta

    def test_config_custom_model_passthrough(self, tmp_path):
        """Unknown model names from config pass through with default metadata."""
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"model": "my-custom-model"}})
        )
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "my-custom-model"
        assert meta["api_model"] == "my-custom-model"


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
        assert result["api_model"] == "gpt-image-2"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        assert saved.read_bytes() == png_bytes

        call_kwargs = fake_client.images.generate.call_args.kwargs
        # Model sent to API should be the api_model, not the tier ID.
        assert call_kwargs["model"] == "gpt-image-2"
        assert call_kwargs["quality"] == "medium"
        assert call_kwargs["size"] == "1536x1024"
        # gpt-image-2 rejects response_format — we must NOT send it.
        assert "response_format" not in call_kwargs

    @pytest.mark.parametrize("tier,api_model,expected_quality", [
        ("gpt-image-2-low", "gpt-image-2", "low"),
        ("gpt-image-2-medium", "gpt-image-2", "medium"),
        ("gpt-image-2-high", "gpt-image-2", "high"),
    ])
    def test_gpt_image_2_tier_maps_correctly(self, provider, monkeypatch, tier, api_model, expected_quality):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", tier)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["model"] == tier
        assert result["quality"] == expected_quality
        assert result["api_model"] == api_model
        assert fake_client.images.generate.call_args.kwargs["quality"] == expected_quality
        assert fake_client.images.generate.call_args.kwargs["model"] == api_model

    @pytest.mark.parametrize("tier,api_model,expected_quality", [
        ("gpt-image-1.5-low", "gpt-image-1.5", "low"),
        ("gpt-image-1.5-medium", "gpt-image-1.5", "medium"),
        ("gpt-image-1.5-high", "gpt-image-1.5", "high"),
    ])
    def test_gpt_image_1_5_tier_maps_correctly(self, provider, monkeypatch, tier, api_model, expected_quality):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", tier)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["model"] == tier
        assert result["quality"] == expected_quality
        assert result["api_model"] == api_model
        assert fake_client.images.generate.call_args.kwargs["quality"] == expected_quality
        assert fake_client.images.generate.call_args.kwargs["model"] == api_model

    @pytest.mark.parametrize("model_id", ["gpt-image-1", "gpt-image-1-mini"])
    def test_flat_models_no_quality_param(self, provider, monkeypatch, model_id):
        """gpt-image-1 and gpt-image-1-mini do NOT accept the quality parameter."""
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", model_id)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["model"] == model_id
        assert result["api_model"] == model_id
        assert "quality" not in result
        call_kwargs = fake_client.images.generate.call_args.kwargs
        assert call_kwargs["model"] == model_id
        assert "quality" not in call_kwargs

    def test_custom_model_passthrough_generate(self, provider, monkeypatch):
        """Unknown model names from env should pass through to the API."""
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "future-model-x")
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["api_model"] == "future-model-x"
        assert fake_client.images.generate.call_args.kwargs["model"] == "future-model-x"
        assert "quality" not in fake_client.images.generate.call_args.kwargs

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

    def test_url_response_is_cached_locally(self, provider):
        """OpenAI URL response (if API ever returns one) is cached locally.

        Pre-fix this asserted the bare URL passed through; symmetric to the
        xAI #26942 fix.  Even though gpt-image-2 returns b64 today, every
        ``image_gen`` provider must guarantee the gateway gets a stable
        file path so ephemeral signed URLs can't expire mid-flight.
        """
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=None, url="https://example.com/img.png",
        )

        with _patched_openai(fake_client), patch(
            "plugins.image_gen.openai.save_url_image",
            return_value=Path("/tmp/openai_gpt-image-2_20260524_000000_deadbeef.png"),
        ) as mock_save_url:
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["image"].startswith("/")
        assert "example.com" not in result["image"]
        mock_save_url.assert_called_once()

    def test_url_response_falls_back_to_bare_url_when_download_fails(self, provider):
        """Cache failure must not turn into a tool error — symmetric with xAI."""
        import requests as req_lib

        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=None, url="https://example.com/img.png",
        )

        with _patched_openai(fake_client), patch(
            "plugins.image_gen.openai.save_url_image",
            side_effect=req_lib.HTTPError("404 from CDN"),
        ):
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["image"] == "https://example.com/img.png"