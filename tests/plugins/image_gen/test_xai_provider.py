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
def _fake_api_key(monkeypatch):
    """Ensure XAI_API_KEY is set for all tests."""
    monkeypatch.setenv("XAI_API_KEY", "test-key-12345")


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
        # FR1: quality model is listed first (current official recommendation).
        assert models[0]["id"] == "grok-imagine-image-quality"
        # The lower-cost/fast model stays selectable.
        assert "grok-imagine-image" in {m["id"] for m in models}

    def test_default_model(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        assert provider.default_model() == "grok-imagine-image-quality"

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


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_model(self):
        from plugins.image_gen.xai import _resolve_model

        model_id, meta = _resolve_model()
        assert model_id == "grok-imagine-image-quality"

    def test_default_resolution(self):
        from plugins.image_gen.xai import _resolve_resolution

        assert _resolve_resolution() == "1k"

    def test_custom_model(self, monkeypatch):
        monkeypatch.setenv("XAI_IMAGE_MODEL", "grok-imagine-image")
        from plugins.image_gen.xai import _resolve_model

        model_id, _ = _resolve_model()
        assert model_id == "grok-imagine-image"


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
        assert result["model"] == "grok-imagine-image-quality"

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

    def test_payload_requests_base64_response(self):
        """xAI image gen should request b64_json so Hermes never depends on imgen.x.ai temp URLs."""
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdC1pbWFnZS1kYXRh"}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/test.png"):
            provider = XAIImageGenProvider()
            provider.generate(prompt="test")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["response_format"] == "b64_json"

    def test_generate_with_image_url_routes_to_edit_endpoint(self):
        """Backward compatibility: image_generate image_url still reaches xAI edits."""
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/edited.png"):
            provider = XAIImageGenProvider()
            result = provider.generate(
                prompt="make it blue",
                image_url="https://x.example/source.png",
                aspect_ratio="square",
                n=2,
            )

        url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args[0][0]
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert result["success"] is True
        assert result["modality"] == "image"
        assert url.endswith("/images/edits")
        assert payload["image"] == {"url": "https://x.example/source.png", "type": "image_url"}
        assert payload["aspect_ratio"] == "1:1"
        assert payload["n"] == 2
        assert payload["response_format"] == "b64_json"

    def test_generate_with_reference_images_routes_to_multi_image_edit(self):
        """image_generate reference_image_urls remain compatible with xAI multi-image edits."""
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/edited.png"):
            provider = XAIImageGenProvider()
            provider.generate(
                prompt="blend them",
                image_url="https://x.example/source.png",
                reference_image_urls=[" https://x.example/ref.png "],
            )

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert "image" not in payload
        assert payload["images"] == [
            {"url": "https://x.example/source.png", "type": "image_url"},
            {"url": "https://x.example/ref.png", "type": "image_url"},
        ]


# ---------------------------------------------------------------------------
# Edit / image-to-image tests
# ---------------------------------------------------------------------------


# Smallest byte sequence that passes the PNG magic-byte sniff.
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class TestEdit:
    def test_supports_edit_is_true(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        assert XAIImageGenProvider().supports_edit() is True

    def test_edit_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        result = provider.edit(prompt="make it blue", image="https://x/a.png")
        assert result["success"] is False
        assert result["error_type"] == "missing_api_key"

    def test_edit_requires_prompt(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        result = provider.edit(prompt="   ", image="https://x/a.png")
        assert result["success"] is False
        assert result["error_type"] == "invalid_input"

    def test_edit_requires_image(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        result = provider.edit(prompt="make it blue", image="")
        assert result["success"] is False
        assert result["error_type"] == "invalid_input"

    def test_edit_calls_images_edits_endpoint(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/edited.png"):
            provider = XAIImageGenProvider()
            provider.edit(prompt="make it blue", image="https://x/a.png")

        url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args[0][0]
        assert url.endswith("/images/edits"), f"edit must POST /v1/images/edits, got {url!r}"

    def test_edit_url_input_uses_image_object(self):
        """http(s) URLs go in the official ``image`` object shape.

        xAI's ``/v1/images/edits`` request JSON wraps the source image in an
        object: ``"image": {"url": "<url-or-data-uri>", "type": "image_url"}``
        (https://docs.x.ai/developers/model-capabilities/images/editing). The
        earlier top-level ``image_url`` / raw-string ``image`` shapes were
        wrong and are rejected by the endpoint.
        """
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/edited.png"):
            provider = XAIImageGenProvider()
            result = provider.edit(prompt="make it blue", image="https://imgen.x.ai/a.png")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["model"] == "grok-imagine-image-quality"
        assert payload["prompt"] == "make it blue"
        assert payload["response_format"] == "b64_json"
        assert payload["image"] == {
            "url": "https://imgen.x.ai/a.png",
            "type": "image_url",
        }, "http URLs must be wrapped in the {url, type:image_url} object"
        assert "image_url" not in payload, "no top-level image_url field — use image.url"
        assert result["success"] is True
        assert result["image"] == "/tmp/edited.png"
        assert result["provider"] == "xai"
        assert result["modality"] == "image"

    def test_edit_local_path_becomes_data_uri(self, tmp_path):
        """Local files are inlined as a base64 data URI inside the image object.

        Per the docs the ``url`` field accepts a public URL *or* a
        base64-encoded data URI, so a validated local file rides the same
        ``{"url": ..., "type": "image_url"}`` object shape.
        """
        from plugins.image_gen.xai import XAIImageGenProvider

        src = tmp_path / "input.png"
        src.write_bytes(_PNG_BYTES)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/edited.png"):
            provider = XAIImageGenProvider()
            result = provider.edit(prompt="make it blue", image=str(src))

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert isinstance(payload["image"], dict)
        assert payload["image"]["type"] == "image_url"
        assert payload["image"]["url"].startswith("data:image/png;base64,")
        assert "image_url" not in payload, "local files must be inlined as a data URI in image.url"
        assert result["success"] is True

    def test_edit_data_uri_passthrough(self):
        """A data URI is passed verbatim as the image object's ``url``."""
        from plugins.image_gen.xai import XAIImageGenProvider

        data_uri = "data:image/png;base64,dGVzdA=="
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/edited.png"):
            provider = XAIImageGenProvider()
            provider.edit(prompt="make it blue", image=data_uri)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["image"] == {"url": data_uri, "type": "image_url"}

    def test_edit_local_path_missing_is_invalid_input(self, tmp_path):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        result = provider.edit(prompt="x", image=str(tmp_path / "nope.png"))
        assert result["success"] is False
        assert result["error_type"] == "invalid_input"

    def test_edit_local_path_non_image_is_invalid_input(self, tmp_path):
        from plugins.image_gen.xai import XAIImageGenProvider

        bad = tmp_path / "fake.png"
        bad.write_bytes(b"definitely not an image")
        provider = XAIImageGenProvider()
        result = provider.edit(prompt="x", image=str(bad))
        assert result["success"] is False
        assert result["error_type"] == "invalid_input"

    def test_edit_url_response_is_cached(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"url": "https://imgen.x.ai/xai-tmp-edit.jpeg"}],
        }

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp), \
             patch(
                 "plugins.image_gen.xai.save_url_image",
                 return_value=Path("/tmp/xai_edit_20260607_000000_deadbeef.jpg"),
             ) as mock_save_url:
            provider = XAIImageGenProvider()
            result = provider.edit(prompt="x", image="https://x/a.png")

        assert result["success"] is True
        assert result["image"].startswith("/")
        assert "imgen.x.ai" not in result["image"]
        mock_save_url.assert_called_once()

    def test_edit_api_error(self):
        import requests as req_lib
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.text = "Unprocessable"
        mock_resp.json.return_value = {"error": {"message": "bad image"}}
        mock_resp.raise_for_status.side_effect = req_lib.HTTPError(response=mock_resp)

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp):
            provider = XAIImageGenProvider()
            result = provider.edit(prompt="x", image="https://x/a.png")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "edit" in result["error"].lower()

    def test_edit_timeout(self):
        import requests as req_lib
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post", side_effect=req_lib.Timeout()):
            provider = XAIImageGenProvider()
            result = provider.edit(prompt="x", image="https://x/a.png")

        assert result["success"] is False
        assert result["error_type"] == "timeout"

    def test_edit_requests_base64_response(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/edited.png"):
            provider = XAIImageGenProvider()
            provider.edit(prompt="x", image="https://x/a.png")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["response_format"] == "b64_json"

    def test_edit_payload_uses_configured_resolution(self):
        """Reference-image edits should honor image_gen.xai.resolution like generation."""
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}

        with patch("plugins.image_gen.xai._load_xai_config", return_value={"resolution": "2k"}), \
             patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/edited.png"):
            result = XAIImageGenProvider().edit(prompt="x", image="https://x/a.png")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["resolution"] == "2k"
        assert result["resolution"] == "2k"


# ---------------------------------------------------------------------------
# FR1 — Model catalog and default
# ---------------------------------------------------------------------------


class TestModelCatalogFR1:
    def test_default_model_constant_is_quality(self):
        from plugins.image_gen import xai

        assert xai.DEFAULT_MODEL == "grok-imagine-image-quality"

    def test_resolve_model_default_is_quality(self):
        from plugins.image_gen.xai import _resolve_model

        model_id, meta = _resolve_model()
        assert model_id == "grok-imagine-image-quality"
        assert isinstance(meta, dict)

    def test_explicit_env_selects_standard_model(self, monkeypatch):
        monkeypatch.setenv("XAI_IMAGE_MODEL", "grok-imagine-image")
        from plugins.image_gen.xai import _resolve_model

        model_id, _ = _resolve_model()
        assert model_id == "grok-imagine-image"

    def test_deprecated_pro_model_falls_back_to_quality(self, monkeypatch):
        # grok-imagine-image-pro is officially deprecated (2026-05-15); it must
        # never resolve as the active model — fall back to the quality default.
        monkeypatch.setenv("XAI_IMAGE_MODEL", "grok-imagine-image-pro")
        from plugins.image_gen.xai import _resolve_model

        model_id, _ = _resolve_model()
        assert model_id == "grok-imagine-image-quality"

    def test_unknown_model_falls_back_to_quality(self, monkeypatch):
        monkeypatch.setenv("XAI_IMAGE_MODEL", "grok-imagine-image-9000")
        from plugins.image_gen.xai import _resolve_model

        model_id, _ = _resolve_model()
        assert model_id == "grok-imagine-image-quality"

    def test_pro_model_not_in_catalog(self):
        # Deprecated model must not be offered as a selectable catalog entry.
        from plugins.image_gen.xai import XAIImageGenProvider

        ids = {m["id"] for m in XAIImageGenProvider().list_models()}
        assert "grok-imagine-image-pro" not in ids


# ---------------------------------------------------------------------------
# FR2 — xAI aspect-ratio handling
# ---------------------------------------------------------------------------


class TestAspectResolverFR2:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("landscape", "16:9"),
            ("portrait", "9:16"),
            ("square", "1:1"),
            ("4:3", "4:3"),
            ("20:9", "20:9"),
            ("9:19.5", "9:19.5"),
            ("auto", "auto"),
            ("AUTO", "auto"),
            ("  16:9  ", "16:9"),
        ],
    )
    def test_resolver_maps_aliases_and_official_values(self, value, expected):
        from plugins.image_gen.xai import _resolve_xai_aspect_ratio

        assert _resolve_xai_aspect_ratio(value) == expected

    def test_resolver_invalid_soft_falls_back_to_default(self):
        from plugins.image_gen.xai import _resolve_xai_aspect_ratio

        # Invalid input must not crash; it falls back to the landscape default
        # wire ratio rather than raising.
        assert _resolve_xai_aspect_ratio("cinemascope") == "16:9"
        assert _resolve_xai_aspect_ratio(None) == "16:9"
        assert _resolve_xai_aspect_ratio(123) == "16:9"

    def _post_mock(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}
        return mock_resp

    @pytest.mark.parametrize(
        "requested,wire",
        [("landscape", "16:9"), ("portrait", "9:16"), ("4:3", "4:3"), ("20:9", "20:9"), ("auto", "auto")],
    )
    def test_generate_payload_uses_official_wire_ratio(self, requested, wire):
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post", return_value=self._post_mock()) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/x.png"):
            XAIImageGenProvider().generate(prompt="t", aspect_ratio=requested)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["aspect_ratio"] == wire

    def test_generate_echo_preserves_alias_for_backward_compat(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post", return_value=self._post_mock()), \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/x.png"):
            result = XAIImageGenProvider().generate(prompt="t", aspect_ratio="square")

        # Echoed aspect_ratio keeps the caller's alias (FR10), even though the
        # wire payload carried the official 1:1 ratio.
        assert result["aspect_ratio"] == "square"


# ---------------------------------------------------------------------------
# FR3 — Multi-output generation and edit
# ---------------------------------------------------------------------------


class TestMultiOutputFR3:
    def _multi_b64_mock(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"b64_json": "dGVzdA=="}, {"b64_json": "dGVzdDI="}],
        }
        return mock_resp

    def test_generate_n_sets_payload_and_returns_all_outputs(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post", return_value=self._multi_b64_mock()) as mock_post, \
             patch(
                 "plugins.image_gen.xai.save_b64_image",
                 side_effect=["/tmp/a.png", "/tmp/b.png"],
             ):
            result = XAIImageGenProvider().generate(prompt="t", n=2)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["n"] == 2
        # Backward compat: first output stays in result["image"].
        assert result["image"] == "/tmp/a.png"
        # FR3: every returned output is cached and reported in result["images"].
        refs = [item["image"] for item in result["images"]]
        assert refs == ["/tmp/a.png", "/tmp/b.png"]

    def test_generate_default_has_no_n_and_no_images_list(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/a.png"):
            result = XAIImageGenProvider().generate(prompt="t")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert "n" not in payload
        # Single output keeps the legacy shape — no "images" key.
        assert "images" not in result

    def test_edit_n_sets_payload_and_returns_all_outputs(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post", return_value=self._multi_b64_mock()) as mock_post, \
             patch(
                 "plugins.image_gen.xai.save_b64_image",
                 side_effect=["/tmp/e1.png", "/tmp/e2.png"],
             ):
            result = XAIImageGenProvider().edit(prompt="x", image="https://x/a.png", n=2)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["n"] == 2
        assert result["image"] == "/tmp/e1.png"
        assert [item["image"] for item in result["images"]] == ["/tmp/e1.png", "/tmp/e2.png"]


# ---------------------------------------------------------------------------
# FR5 / FR6 — File-id input and multi-image edit
# ---------------------------------------------------------------------------


class TestFileIdAndMultiImageFR56:
    def _ok_mock(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}
        return mock_resp

    def test_single_file_id_uses_file_id_object(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post", return_value=self._ok_mock()) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/edited.png"):
            result = XAIImageGenProvider().edit(prompt="x", image="file_abc123")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["image"] == {"file_id": "file_abc123"}
        assert "url" not in payload["image"]
        assert "images" not in payload
        assert result["success"] is True

    def test_multi_image_mixed_inputs_build_images_array(self, tmp_path):
        from plugins.image_gen.xai import XAIImageGenProvider

        src = tmp_path / "in.png"
        src.write_bytes(_PNG_BYTES)
        data_uri = "data:image/png;base64,dGVzdA=="

        with patch("plugins.image_gen.xai.requests.post", return_value=self._ok_mock()) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/edited.png"):
            result = XAIImageGenProvider().edit(
                prompt="combine",
                images=[str(src), "file_xyz", "https://x/a.png", data_uri][:3],
            )

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert "image" not in payload, "multi-image edit must not send a top-level image object"
        imgs = payload["images"]
        assert len(imgs) == 3
        assert imgs[0]["type"] == "image_url"
        assert imgs[0]["url"].startswith("data:image/png;base64,")
        assert imgs[1] == {"file_id": "file_xyz"}
        assert imgs[2] == {"url": "https://x/a.png", "type": "image_url"}
        assert result["success"] is True

    def test_both_image_and_images_is_invalid_input(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post") as mock_post:
            result = XAIImageGenProvider().edit(
                prompt="x", image="https://x/a.png", images=["https://x/b.png"]
            )
        assert result["success"] is False
        assert result["error_type"] == "invalid_input"
        mock_post.assert_not_called()

    def test_empty_images_list_is_invalid_input(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post") as mock_post:
            result = XAIImageGenProvider().edit(prompt="x", images=[])
        assert result["success"] is False
        assert result["error_type"] == "invalid_input"
        mock_post.assert_not_called()

    def test_more_than_three_images_is_invalid_input_before_network(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post") as mock_post:
            result = XAIImageGenProvider().edit(
                prompt="x",
                images=["https://x/1.png", "https://x/2.png", "https://x/3.png", "https://x/4.png"],
            )
        assert result["success"] is False
        assert result["error_type"] == "invalid_input"
        mock_post.assert_not_called()

    def test_empty_file_id_is_invalid_input(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post") as mock_post:
            result = XAIImageGenProvider().edit(prompt="x", image="file_")
        assert result["success"] is False
        assert result["error_type"] == "invalid_input"
        mock_post.assert_not_called()

    def test_bad_local_path_in_images_is_invalid_input(self, tmp_path):
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post") as mock_post:
            result = XAIImageGenProvider().edit(
                prompt="x", images=["https://x/a.png", str(tmp_path / "nope.png")]
            )
        assert result["success"] is False
        assert result["error_type"] == "invalid_input"
        mock_post.assert_not_called()

    def test_file_id_not_treated_as_local_path(self):
        # A file id must never be probed on the local filesystem.
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post", return_value=self._ok_mock()) as mock_post, \
             patch("plugins.image_gen.xai.local_image_to_data_uri") as mock_local, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/edited.png"):
            XAIImageGenProvider().edit(prompt="x", image="file_realid")

        mock_local.assert_not_called()
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["image"] == {"file_id": "file_realid"}


# ---------------------------------------------------------------------------
# FR7 — storage_options pass-through (default-off, no public URL by default)
# ---------------------------------------------------------------------------


class TestStorageOptionsFR7:
    def _ok_mock(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}
        return mock_resp

    def test_no_storage_options_by_default(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post", return_value=self._ok_mock()) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/x.png"):
            XAIImageGenProvider().generate(prompt="t")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert "storage_options" not in payload

    def test_valid_storage_options_pass_through(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        opts = {"filename": "out.png", "expires_after": 7200, "public_url": True}
        with patch("plugins.image_gen.xai.requests.post", return_value=self._ok_mock()) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/x.png"):
            XAIImageGenProvider().generate(prompt="t", storage_options=opts)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["storage_options"] == opts

    def test_storage_options_does_not_inject_public_url(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        opts = {"filename": "out.png"}
        with patch("plugins.image_gen.xai.requests.post", return_value=self._ok_mock()) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/x.png"):
            XAIImageGenProvider().generate(prompt="t", storage_options=opts)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert "public_url" not in payload["storage_options"]

    @pytest.mark.parametrize(
        "opts",
        [
            "notadict",
            {"expires_after": 7200},  # missing filename
            {"filename": "out.png", "expires_after": 60},  # below 3600
            {"filename": "out.png", "expires_after": 2592001},  # above max
            {"filename": "out.png", "expires_after": "lots"},  # wrong type
            {"filename": "out.png", "public_url": "yes"},  # wrong public_url type
            {"filename": "out.png", "public_url": {"expires_after": 1}},  # nested out of range
        ],
    )
    def test_invalid_storage_options_rejected_before_network(self, opts):
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post") as mock_post:
            result = XAIImageGenProvider().generate(prompt="t", storage_options=opts)
        assert result["success"] is False
        assert result["error_type"] == "invalid_input"
        mock_post.assert_not_called()

    def test_edit_valid_storage_options_pass_through(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        opts = {"filename": "edit.png"}
        with patch("plugins.image_gen.xai.requests.post", return_value=self._ok_mock()) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/e.png"):
            XAIImageGenProvider().edit(prompt="x", image="https://x/a.png", storage_options=opts)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["storage_options"] == opts


# ---------------------------------------------------------------------------
# FR8 — MIME-aware cache + response metadata preservation
# ---------------------------------------------------------------------------


class TestMimeAndMetadataFR8:
    def test_webp_mime_selects_webp_extension(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"b64_json": "dGVzdA==", "mime_type": "image/webp"}],
        }

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp), \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/x.webp") as mock_save:
            result = XAIImageGenProvider().generate(prompt="t")

        _args, kwargs = mock_save.call_args
        assert kwargs.get("extension") == "webp"
        assert result["mime_type"] == "image/webp"

    def test_jpeg_mime_selects_jpg_extension(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"b64_json": "dGVzdA==", "mime_type": "image/jpeg"}],
        }

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp), \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/x.jpg") as mock_save:
            XAIImageGenProvider().generate(prompt="t")

        assert mock_save.call_args.kwargs.get("extension") == "jpg"

    def test_unknown_mime_defaults_to_png_extension(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"b64_json": "dGVzdA==", "mime_type": "image/tiff"}],
        }

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp), \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/x.png") as mock_save:
            XAIImageGenProvider().generate(prompt="t")

        assert mock_save.call_args.kwargs.get("extension") == "png"

    def test_metadata_fields_preserved(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        file_output = {
            "file_id": "file_out1",
            "filename": "out.png",
            "expires_at": 1234567890,
            "public_url_error": "rate_limited",
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"b64_json": "dGVzdA==", "mime_type": "image/png", "file_output": file_output}],
            "storage_error": "quota",
            "public_url_error": "blocked",
            "usage": {"cost_in_usd_ticks": 42},
        }

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp), \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/x.png"):
            result = XAIImageGenProvider().generate(prompt="t")

        assert result["file_output"] == file_output
        assert result["storage_error"] == "quota"
        assert result["public_url_error"] == "blocked"
        assert result["usage"] == {"cost_in_usd_ticks": 42}

    def test_per_image_mime_and_file_output_in_images_list(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"b64_json": "dGVzdA==", "mime_type": "image/png"},
                {"b64_json": "dGVzdDI=", "mime_type": "image/webp", "file_output": {"file_id": "f2"}},
            ],
        }

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp), \
             patch("plugins.image_gen.xai.save_b64_image", side_effect=["/tmp/a.png", "/tmp/b.webp"]):
            result = XAIImageGenProvider().generate(prompt="t", n=2)

        images = result["images"]
        assert images[0]["mime_type"] == "image/png"
        assert images[1]["mime_type"] == "image/webp"
        assert images[1]["file_output"] == {"file_id": "f2"}


# ---------------------------------------------------------------------------
# FR9 — service_tier stays absent by default
# ---------------------------------------------------------------------------


class TestServiceTierAbsentFR9:
    def _ok_mock(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}
        return mock_resp

    def test_generate_payload_has_no_service_tier(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post", return_value=self._ok_mock()) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/x.png"):
            XAIImageGenProvider().generate(prompt="t")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert "service_tier" not in payload

    def test_edit_payload_has_no_service_tier(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        with patch("plugins.image_gen.xai.requests.post", return_value=self._ok_mock()) as mock_post, \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/tmp/e.png"):
            XAIImageGenProvider().edit(prompt="x", image="https://x/a.png")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert "service_tier" not in payload


# ---------------------------------------------------------------------------
# FR10 — Backward compatibility of the simple success shape
# ---------------------------------------------------------------------------


class TestBackwardCompatFR10:
    def test_simple_generate_shape_unchanged(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp), \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/abs/cache/path.png"):
            result = XAIImageGenProvider().generate(prompt="A cat", aspect_ratio="square")

        for key in ("success", "image", "model", "prompt", "aspect_ratio", "provider"):
            assert key in result
        assert result["success"] is True
        assert result["image"] == "/abs/cache/path.png"
        assert result["model"] == "grok-imagine-image-quality"
        assert result["prompt"] == "A cat"
        assert result["aspect_ratio"] == "square"
        assert result["provider"] == "xai"

    def test_simple_edit_shape_unchanged(self, tmp_path):
        from plugins.image_gen.xai import XAIImageGenProvider

        src = tmp_path / "input.png"
        src.write_bytes(_PNG_BYTES)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp), \
             patch("plugins.image_gen.xai.save_b64_image", return_value="/abs/cache/edit.png"):
            result = XAIImageGenProvider().edit(prompt="make it blue", image=str(src))

        assert result["success"] is True
        assert result["image"] == "/abs/cache/edit.png"
        assert result["provider"] == "xai"
        assert result["model"] == "grok-imagine-image-quality"


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
