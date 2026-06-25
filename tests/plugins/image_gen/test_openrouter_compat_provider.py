#!/usr/bin/env python3
"""Tests for the OpenRouter-compatible image gen provider (OpenRouter + Nous)."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_RUNTIME = "hermes_cli.runtime_provider.resolve_runtime_provider"
_PNG_DATA_URI = "data:image/png;base64,dGVzdC1pbWFnZS1kYXRh"  # "test-image-data"


def _runtime_ok(**over):
    base = {
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "sk-or-test",
        "source": "env",
    }
    base.update(over)
    return base


def _mock_chat_response(images):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "images": [
                        {"type": "image_url", "image_url": {"url": u}} for u in images
                    ],
                }
            }
        ]
    }
    return resp


def _openrouter():
    from plugins.image_gen.openrouter import OpenRouterCompatImageProvider

    return OpenRouterCompatImageProvider(
        provider_name="openrouter",
        display_name="OpenRouter",
        runtime_name="openrouter",
        config_key="openrouter",
        model_env_var="OPENROUTER_IMAGE_MODEL",
        setup_schema={"name": "OpenRouter (image)", "badge": "paid", "env_vars": []},
    )


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class TestProviderClass:
    def test_names(self):
        from plugins.image_gen.openrouter import _build_providers

        names = {p.name for p in _build_providers()}
        assert names == {"openrouter", "nous"}

    def test_display_names(self):
        from plugins.image_gen.openrouter import _build_providers

        by_name = {p.name: p for p in _build_providers()}
        assert by_name["openrouter"].display_name == "OpenRouter"
        assert by_name["nous"].display_name == "Nous Portal"

    def test_capabilities_support_image_input(self):
        caps = _openrouter().capabilities()
        assert "image" in caps["modalities"]
        assert caps["max_reference_images"] >= 1

    def test_is_available_with_key(self):
        with patch(_RUNTIME, return_value=_runtime_ok()):
            assert _openrouter().is_available() is True

    def test_is_available_without_key(self):
        with patch(_RUNTIME, return_value=_runtime_ok(api_key="")):
            assert _openrouter().is_available() is False

    def test_is_available_on_resolution_error(self):
        with patch(_RUNTIME, side_effect=RuntimeError("boom")):
            assert _openrouter().is_available() is False

    def test_default_model(self):
        from plugins.image_gen.openrouter import DEFAULT_MODEL

        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value={}):
            assert _openrouter().default_model() == DEFAULT_MODEL
            assert DEFAULT_MODEL == "google/gemini-2.5-flash-image"

    def test_model_env_override(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_IMAGE_MODEL", "black-forest-labs/flux.2-pro")
        assert _openrouter()._resolve_model() == "black-forest-labs/flux.2-pro"

    def test_model_config_override(self):
        cfg = {"openrouter": {"model": "google/gemini-3.1-flash-image-preview"}}
        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value=cfg):
            assert _openrouter()._resolve_model() == "google/gemini-3.1-flash-image-preview"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_to_image_url_part_passthrough_url(self):
        from plugins.image_gen.openrouter import _to_image_url_part

        assert _to_image_url_part("https://x/y.png") == "https://x/y.png"
        assert _to_image_url_part("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"

    def test_to_image_url_part_inlines_local_file(self, tmp_path):
        from plugins.image_gen.openrouter import _to_image_url_part

        f = tmp_path / "base.png"
        f.write_bytes(b"\x89PNG\r\n")
        part = _to_image_url_part(str(f))
        assert part.startswith("data:image/png;base64,")
        decoded = base64.b64decode(part.split(",", 1)[1])
        assert decoded == b"\x89PNG\r\n"

    def test_to_image_url_part_missing_file(self):
        from plugins.image_gen.openrouter import _to_image_url_part

        assert _to_image_url_part("/no/such/file.png") is None

    def test_extract_images(self):
        from plugins.image_gen.openrouter import _extract_images

        payload = {
            "choices": [
                {"message": {"images": [{"image_url": {"url": "data:image/png;base64,AA"}}]}}
            ]
        }
        assert _extract_images(payload) == ["data:image/png;base64,AA"]

    def test_extract_images_empty(self):
        from plugins.image_gen.openrouter import _extract_images

        assert _extract_images({"choices": [{"message": {"content": "no image"}}]}) == []


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_missing_credentials(self):
        with patch(_RUNTIME, return_value=_runtime_ok(api_key="")):
            result = _openrouter().generate(prompt="a pet")
        assert result["success"] is False
        assert result["error_type"] == "missing_api_key"

    def test_success_data_uri(self):
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_chat_response([_PNG_DATA_URI])), \
             patch(
                 "plugins.image_gen.openrouter.save_b64_image",
                 return_value=Path("/tmp/openrouter_gen.png"),
             ) as mock_save:
            result = _openrouter().generate(prompt="a pet")

        assert result["success"] is True
        assert result["image"] == "/tmp/openrouter_gen.png"
        assert result["provider"] == "openrouter"
        mock_save.assert_called_once()

    def test_success_http_url(self):
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_chat_response(["https://cdn/x.png"])), \
             patch(
                 "plugins.image_gen.openrouter.save_url_image",
                 return_value=Path("/tmp/openrouter_gen_url.png"),
             ) as mock_save_url:
            result = _openrouter().generate(prompt="a pet")

        assert result["success"] is True
        assert result["image"] == "/tmp/openrouter_gen_url.png"
        mock_save_url.assert_called_once()

    def test_empty_response(self):
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_chat_response([])):
            result = _openrouter().generate(prompt="a pet")
        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_payload_shape_and_references(self, tmp_path):
        """Wire payload must carry image modalities, aspect_ratio, and the
        reference image inlined as a data URI (this is what makes pet rows
        stay on-model)."""
        ref = tmp_path / "base.png"
        ref.write_bytes(b"\x89PNG\r\n")

        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_chat_response([_PNG_DATA_URI])) as mock_post, \
             patch("plugins.image_gen.openrouter.save_b64_image", return_value=Path("/tmp/x.png")):
            _openrouter().generate(
                prompt="a pet", aspect_ratio="square", reference_images=[str(ref)]
            )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["modalities"] == ["image", "text"]
        assert payload["image_config"]["aspect_ratio"] == "1:1"
        content = payload["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "a pet"}
        image_parts = [c for c in content if c["type"] == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_auth_header(self):
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_chat_response([_PNG_DATA_URI])) as mock_post, \
             patch("plugins.image_gen.openrouter.save_b64_image", return_value=Path("/tmp/x.png")):
            _openrouter().generate(prompt="a pet")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-or-test"

    def test_posts_to_resolved_base_url(self):
        """Nous routes to its own base URL — proves the same code serves both."""
        nous_runtime = _runtime_ok(
            provider="nous", base_url="https://inference.nousresearch.com/v1", api_key="nous-tok"
        )
        with patch(_RUNTIME, return_value=nous_runtime), \
             patch("requests.post", return_value=_mock_chat_response([_PNG_DATA_URI])) as mock_post, \
             patch("plugins.image_gen.openrouter.save_b64_image", return_value=Path("/tmp/x.png")):
            from plugins.image_gen.openrouter import _build_providers

            nous = {p.name: p for p in _build_providers()}["nous"]
            result = nous.generate(prompt="a pet")

        assert result["success"] is True
        assert result["provider"] == "nous"
        url = mock_post.call_args[0][0]
        assert url == "https://inference.nousresearch.com/v1/chat/completions"

    def test_api_error(self):
        import requests as req_lib

        resp = MagicMock()
        resp.status_code = 401
        resp.text = "Unauthorized"
        resp.json.return_value = {"error": {"message": "Invalid API key"}}
        resp.raise_for_status.side_effect = req_lib.HTTPError(response=resp)

        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=resp):
            result = _openrouter().generate(prompt="a pet")
        assert result["success"] is False
        assert result["error_type"] == "api_error"

    def test_timeout(self):
        import requests as req_lib

        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", side_effect=req_lib.Timeout()):
            result = _openrouter().generate(prompt="a pet")
        assert result["success"] is False
        assert result["error_type"] == "timeout"


# ---------------------------------------------------------------------------
# Registration + pet integration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_both(self):
        from plugins.image_gen.openrouter import register

        ctx = MagicMock()
        register(ctx)
        registered = [c.args[0].name for c in ctx.register_image_gen_provider.call_args_list]
        assert set(registered) == {"openrouter", "nous"}

    def test_both_are_reference_capable_for_pets(self):
        from agent.pet.generate.imagegen import _REF_CAPABLE

        assert "openrouter" in _REF_CAPABLE
        assert "nous" in _REF_CAPABLE
