"""Tests for the bundled Meta Model API image_gen plugin (muse-image)."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# The plugin directory uses a hyphen, which is not a valid Python identifier
# for the dotted-import form. Load it via importlib so tests don't need to
# touch sys.path or rename the directory.
meta_plugin = importlib.import_module("plugins.image_gen.meta-ai")


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
    # Clear every auth + override env var so tests start from a clean slate.
    for env in ("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY",
                "META_BASE_URL", "META_IMAGE_MODEL"):
        monkeypatch.delenv(env, raising=False)
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("META_MODEL_API_KEY", "test-key")
    return meta_plugin.MetaImageGenProvider()


def _patched_openai(fake_client: MagicMock):
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    return patch.dict("sys.modules", {"openai": fake_openai})


# ── Metadata ────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "meta-ai"

    def test_display_name(self, provider):
        assert provider.display_name == "Meta Model API"

    def test_default_model(self, provider):
        assert provider.default_model() == "muse-image-1.0"

    def test_list_models(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        assert ids == ["muse-image-1.0"]

    def test_catalog_entries_have_display_speed_strengths_price(self, provider):
        for entry in provider.list_models():
            assert entry["display"]
            assert entry["speed"]
            assert entry["strengths"]
            assert entry["price"]

    def test_text_only_capabilities(self, provider):
        caps = provider.capabilities()
        assert caps["modalities"] == ["text"]
        assert caps["max_reference_images"] == 0


# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_no_api_key_unavailable(self):
        assert meta_plugin.MetaImageGenProvider().is_available() is False

    @pytest.mark.parametrize(
        "env", ["MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY"]
    )
    def test_each_auth_alias_makes_available(self, monkeypatch, env):
        monkeypatch.setenv(env, "test")
        assert meta_plugin.MetaImageGenProvider().is_available() is True


# ── Auth / base-url resolution ────────────────────────────────────────────────


class TestResolution:
    def test_api_key_priority_order(self, monkeypatch):
        # MODEL_API_KEY wins over the aliases.
        monkeypatch.setenv("META_MODEL_API_KEY", "third")
        monkeypatch.setenv("META_API_KEY", "second")
        monkeypatch.setenv("MODEL_API_KEY", "first")
        assert meta_plugin._resolve_api_key() == "first"

    def test_default_base_url(self):
        assert meta_plugin._resolve_base_url() == "https://api.meta.ai/v1"

    def test_base_url_override(self, monkeypatch):
        monkeypatch.setenv("META_BASE_URL", "https://proxy.internal/v1")
        assert meta_plugin._resolve_base_url() == "https://proxy.internal/v1"


# ── Model resolution ──────────────────────────────────────────────────────────


class TestModelResolution:
    def test_default(self):
        model_id, _meta = meta_plugin._resolve_model()
        assert model_id == "muse-image-1.0"

    def test_env_var_override_ignores_unknown(self, monkeypatch):
        monkeypatch.setenv("META_IMAGE_MODEL", "not-a-real-model")
        model_id, _meta = meta_plugin._resolve_model()
        # Unknown id is ignored; falls through to the default.
        assert model_id == "muse-image-1.0"


# ── Generate ──────────────────────────────────────────────────────────────────


class TestGenerate:
    def test_empty_prompt_rejected(self, provider):
        result = provider.generate("", aspect_ratio="square")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        assert result["provider"] == "meta-ai"

    def test_missing_api_key(self):
        result = meta_plugin.MetaImageGenProvider().generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_b64_saves_to_cache(self, provider, tmp_path):
        png_bytes = bytes.fromhex(_PNG_HEX)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "muse-image-1.0"
        assert result["aspect_ratio"] == "landscape"
        assert result["provider"] == "meta-ai"
        assert result["modality"] == "text"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        assert saved.read_bytes() == png_bytes

        call_kwargs = fake_client.images.generate.call_args.kwargs
        assert call_kwargs["model"] == "muse-image-1.0"
        assert call_kwargs["size"] == "1536x1024"
        assert call_kwargs["n"] == 1

    def test_client_uses_meta_base_url(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())
        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client

        with patch.dict("sys.modules", {"openai": fake_openai}):
            provider.generate("a cat")

        assert fake_openai.OpenAI.call_args.kwargs["base_url"] == "https://api.meta.ai/v1"

    def test_base_url_override_reaches_client(self, provider, monkeypatch):
        monkeypatch.setenv("META_BASE_URL", "https://proxy.internal/v1")
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())
        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client

        with patch.dict("sys.modules", {"openai": fake_openai}):
            provider.generate("a cat")

        assert fake_openai.OpenAI.call_args.kwargs["base_url"] == "https://proxy.internal/v1"

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

    def test_url_response_is_cached_locally(self, provider):
        """A URL response is materialized locally (symmetric to the openai/xai
        providers) so ephemeral signed URLs can't expire mid-flight."""
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=None, url="https://example.com/img.webp",
        )

        with _patched_openai(fake_client), patch.object(
            meta_plugin, "save_url_image",
            return_value=Path("/tmp/meta_20260524_000000_deadbeef.webp"),
        ) as mock_save_url:
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["image"].startswith("/")
        assert "example.com" not in result["image"]
        mock_save_url.assert_called_once()

    def test_empty_response_errors(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=None, url=None)

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_api_error_surfaced(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.side_effect = RuntimeError("boom")

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "boom" in result["error"]
