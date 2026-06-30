#!/usr/bin/env python3
"""Tests for xAI image generation provider."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch, tmp_path):
    """Ensure XAI_API_KEY is set for all tests."""
    monkeypatch.setenv("XAI_API_KEY", "test-key-12345")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    try:
        import hermes_cli.config as cfg_mod

        if hasattr(cfg_mod, "_invalidate_load_config_cache"):
            cfg_mod._invalidate_load_config_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Provider class tests
# ---------------------------------------------------------------------------


class TestXAIImageGenProvider:
    def test_name(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        assert provider.name == "xai"

    def test_display_name(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        assert provider.display_name == "xAI (Grok)"

    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "sk-xxx")
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        assert provider.is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        assert provider.is_available() is False

    def test_list_models(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        models = provider.list_models()
        assert len(models) >= 1
        assert models[0]["id"] == "grok-imagine-image"

    def test_default_model(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        assert provider.default_model() == "grok-imagine-image"

    def test_get_setup_schema(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        schema = provider.get_setup_schema()
        assert schema["name"] == "xAI Grok Imagine (image)"
        assert schema["badge"] == "paid"
        # Auth resolution is delegated to the shared "xai_grok" post_setup
        # hook so the picker doesn't blindly prompt for XAI_API_KEY when the
        # user is already signed in via xAI Grok OAuth.
        assert schema["env_vars"] == []
        assert schema["post_setup"] == "xai_grok"

    def test_capabilities_expose_total_source_image_limit(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        caps = XAIImageGenProvider().capabilities()
        assert caps["max_reference_images"] == 2
        assert caps["max_source_images"] == 3


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_model(self):
        from plugins.image_gen.xai import _resolve_model

        model_id, meta = _resolve_model()
        assert model_id == "grok-imagine-image"

    def test_default_resolution(self):
        from plugins.image_gen.xai import _resolve_resolution

        assert _resolve_resolution() == "1k"

    def test_custom_model(self, monkeypatch):
        monkeypatch.setenv("XAI_IMAGE_MODEL", "grok-imagine-image")
        from plugins.image_gen.xai import _resolve_model

        model_id, _ = _resolve_model()
        assert model_id == "grok-imagine-image"

    def test_scoped_config_model(self, monkeypatch):
        monkeypatch.delenv("XAI_IMAGE_MODEL", raising=False)
        from plugins.image_gen.xai import _resolve_model

        with patch(
            "plugins.image_gen.xai._load_xai_config",
            return_value={"model": "grok-imagine-image-quality"},
        ):
            model_id, _ = _resolve_model()
        assert model_id == "grok-imagine-image-quality"

    def test_top_level_config_model(self, monkeypatch):
        """A model picked via ``hermes tools`` is persisted to top-level
        ``image_gen.model``; the xAI provider must honor it, not only the
        scoped ``image_gen.xai.model``.
        """
        monkeypatch.delenv("XAI_IMAGE_MODEL", raising=False)
        from plugins.image_gen.xai import _resolve_model

        with patch("plugins.image_gen.xai._load_xai_config", return_value={}), patch(
            "plugins.image_gen.xai._load_image_gen_config",
            return_value={"model": "grok-imagine-image-quality"},
        ):
            model_id, _ = _resolve_model()
        assert model_id == "grok-imagine-image-quality"

    def test_explicit_model_kwarg_wins_over_config(self, monkeypatch):
        monkeypatch.delenv("XAI_IMAGE_MODEL", raising=False)
        from plugins.image_gen.xai import _resolve_model

        with patch("plugins.image_gen.xai._load_xai_config", return_value={}), patch(
            "plugins.image_gen.xai._load_image_gen_config",
            return_value={"model": "grok-imagine-image"},
        ):
            model_id, _ = _resolve_model("grok-imagine-image-quality")
        assert model_id == "grok-imagine-image-quality"


# ---------------------------------------------------------------------------
# Generate tests
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        result = provider.generate(prompt="test")
        assert result["success"] is False
        assert "XAI_API_KEY" in result["error"]

    def test_successful_generation(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"b64_json": "dGVzdC1pbWFnZS1kYXRh"}],  # base64 "test-image-data"
        }

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp):
            with patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/test.png"):
                provider = XAIImageGenProvider()
                result = provider.generate(prompt="A cat playing piano")

        assert result["success"] is True
        assert result["image"] == "/tmp/test.png"
        assert result["provider"] == "xai"
        assert result["model"] == "grok-imagine-image"

    def test_successful_url_response(self):
        """xAI URL response is cached locally — #26942 contract.

        Pre-fix this asserted ``result["image"] == "<the bare URL>"``, which
        was exactly the bug: xAI's ``imgen.x.ai/xai-tmp-*`` URLs expire fast
        and the gateway 404'd by ``send_photo`` time.  Post-fix the URL
        bytes are downloaded at tool-completion and the result carries an
        absolute filesystem path the gateway can upload from.
        """
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"url": "https://imgen.x.ai/xai-tmp-imgen-test.jpeg"}],
        }

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp), \
             patch(
                 "plugins.image_gen.xai.save_url_image",
                 return_value=Path("/tmp/xai_grok-imagine-image_20260524_000000_deadbeef.jpg"),
             ) as mock_save_url:
            provider = XAIImageGenProvider()
            result = provider.generate(prompt="A cat playing piano")

        assert result["success"] is True
        assert result["image"].startswith("/"), (
            f"URL response must be cached to an absolute path, got {result['image']!r}"
        )
        assert "imgen.x.ai" not in result["image"], (
            "ephemeral xAI URL must not leak into result.image — caller will 404"
        )
        # The downloader should have been called exactly once with the URL
        # and an xai-prefixed cache filename.
        mock_save_url.assert_called_once()
        call_args, call_kwargs = mock_save_url.call_args
        assert call_args[0] == "https://imgen.x.ai/xai-tmp-imgen-test.jpeg"
        assert call_kwargs.get("prefix", "").startswith("xai_")

    def test_url_response_falls_back_to_bare_url_when_download_fails(self):
        """If caching the URL fails (network blip, 404 in-flight), the
        provider must NOT hard-error — fall through to returning the bare
        URL so the agent surface at least sees *something*.  The gateway's
        existing URL-send fallback then has a chance to succeed; if it
        too 404s, the user gets the original (now legible) error rather
        than an opaque "image generation failed" tool result.
        """
        import requests as req_lib
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"url": "https://imgen.x.ai/xai-tmp-imgen-already-404.jpeg"}],
        }

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp), \
             patch(
                 "plugins.image_gen.xai.save_url_image",
                 side_effect=req_lib.HTTPError("404 from CDN"),
             ):
            provider = XAIImageGenProvider()
            result = provider.generate(prompt="A cat playing piano")

        assert result["success"] is True, (
            "Cache failure must not turn into a tool error — gateway gets a chance to retry"
        )
        assert result["image"] == "https://imgen.x.ai/xai-tmp-imgen-already-404.jpeg"

    def test_api_error(self):
        import requests as req_lib
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_resp.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_resp.raise_for_status.side_effect = req_lib.HTTPError(response=mock_resp)

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp):
            provider = XAIImageGenProvider()
            result = provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"

    def test_api_error_preserves_real_response_status(self):
        import requests as req_lib
        from plugins.image_gen.xai import XAIImageGenProvider

        response = req_lib.Response()
        response.status_code = 401
        response._content = json.dumps({"error": {"message": "Invalid API key"}}).encode()
        response.headers["Content-Type"] = "application/json"

        response.raise_for_status = MagicMock(
            side_effect=req_lib.HTTPError(response=response)
        )

        with patch("plugins.image_gen.xai.requests.post", return_value=response):
            provider = XAIImageGenProvider()
            result = provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "xAI image generation failed (401): Invalid API key" in result["error"]

    def test_timeout(self):
        import requests as req_lib

        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post", side_effect=req_lib.Timeout()):
            provider = XAIImageGenProvider()
            result = provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "timeout"

    def test_empty_response(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": []}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp):
            provider = XAIImageGenProvider()
            result = provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_auth_header(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"url": "https://xai.image/test.png"}],
        }

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post:
            provider = XAIImageGenProvider()
            provider.generate(prompt="test")

        call_args = mock_post.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert "Bearer test-key-12345" in headers["Authorization"]
        assert "Hermes-Agent" in headers["User-Agent"]

    def test_payload_resolution_is_literal_1k_or_2k(self):
        """Regression: xAI API rejects numeric resolutions ("1024"/"2048") with 422.

        The endpoint expects the literal strings "1k" or "2k". Ensure the wire
        payload carries that literal — not a numeric mapping. See PR #18678.
        """
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"url": "https://xai.image/test.png"}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post:
            provider = XAIImageGenProvider()
            provider.generate(prompt="test")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["resolution"] in {"1k", "2k"}, (
            f"resolution must be the literal '1k' or '2k', got {payload['resolution']!r}"
        )

    def test_image_edit_rejects_bare_file_id_input(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"url": "https://xai.image/edited.png"}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.xai.save_url_image", return_value="/tmp/edited.png"):
            provider = XAIImageGenProvider()
            result = provider.generate(
                prompt="make the robot red",
                image_url="file_03eb65b1-aa97-482f-9ef0-b04f9172ea00",
            )

        assert result["success"] is False
        assert result["error_type"] == "invalid_image_url"
        mock_post.assert_not_called()

    def test_image_edit_accepts_public_https_url(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"url": "https://xai.image/edited.png"}]}

        public_url = "https://files-cdn.x.ai/token/file_abc.png"
        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.xai.save_url_image", return_value="/tmp/edited.png"):
            provider = XAIImageGenProvider()
            result = provider.generate(
                prompt="make the robot red",
                image_url=public_url,
            )

        assert result["success"] is True
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["image"] == {"url": public_url, "type": "image_url"}

    def test_multi_image_edit_rejects_bare_file_id_inputs(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"url": "https://xai.image/edited.png"}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.xai.save_url_image", return_value="/tmp/edited.png"):
            provider = XAIImageGenProvider()
            result = provider.generate(
                prompt="combine these robots into one product shot",
                image_url="file_03eb65b1-aa97-482f-9ef0-b04f9172ea00",
                reference_image_urls=[
                    "file_54b48d6d-28ad-4982-9d72-bd3ac677c9bc",
                    "file_aa11bb22-cc33-44dd-88ee-ff0011223344",
                ],
            )

        assert result["success"] is False
        assert result["error_type"] == "invalid_image_url"
        mock_post.assert_not_called()

    def test_multi_image_edit_rejects_more_than_three_sources(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        result = provider.generate(
            prompt="combine too many references",
            image_url="file_1",
            reference_image_urls=["file_2", "file_3", "file_4"],
        )

        assert result["success"] is False
        assert result["error_type"] == "too_many_references"

    def test_storage_options_are_sent_by_default(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/test.png"):
            provider = XAIImageGenProvider()
            provider.generate(prompt="test")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["storage_options"]["public_url"] is True
        assert "expires_after" not in payload["storage_options"]
        assert payload["storage_options"]["filename"].endswith(".png")

    def test_public_url_file_output_wins_over_temporary_url(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{
                "url": "https://imgen.x.ai/xai-tmp-imgen-test.jpeg",
                "file_output": {
                    "file_id": "file-123",
                    "filename": "stored.png",
                    "public_url": "https://xai-files.example/stored.png",
                    "public_url_expires_at": 1234567890,
                },
            }],
        }

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp), \
             patch("plugins.image_gen.xai.save_url_image") as mock_save_url:
            provider = XAIImageGenProvider()
            result = provider.generate(prompt="A cat playing piano")

        assert result["success"] is True
        assert result["image"] == "https://xai-files.example/stored.png"
        assert result["public_url"] == "https://xai-files.example/stored.png"
        assert "file_id" not in result
        mock_save_url.assert_not_called()


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register(self):
        from plugins.image_gen.xai import XAIImageGenProvider, register

        mock_ctx = MagicMock()
        register(mock_ctx)
        mock_ctx.register_image_gen_provider.assert_called_once()
        provider = mock_ctx.register_image_gen_provider.call_args[0][0]
        assert isinstance(provider, XAIImageGenProvider)
        assert provider.name == "xai"


def test_xai_image_field_expands_user_home(tmp_path, monkeypatch):
    """A ~-prefixed local image path must load (expanduser), not raise io_error.

    Pre-flight validation uses ``Path(source).expanduser()`` so a ``~/...`` path
    passes; ``_xai_image_field`` must expand it too or the load fails spuriously.
    """
    from plugins.image_gen.xai import _xai_image_field

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    field = _xai_image_field("~/pic.png")
    assert field["type"] == "image_url"
    assert field["url"].startswith("data:image/png;base64,")
