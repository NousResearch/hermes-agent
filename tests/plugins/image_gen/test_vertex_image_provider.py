#!/usr/bin/env python3
"""Tests for Google Vertex AI image generation provider plugin."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _fake_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    try:
        import hermes_cli.config as cfg_mod
        if hasattr(cfg_mod, "_invalidate_load_config_cache"):
            cfg_mod._invalidate_load_config_cache()
    except Exception:
        pass


class TestVertexImageGenProvider:
    def test_name(self):
        from plugins.image_gen.vertex import VertexImageGenProvider

        provider = VertexImageGenProvider()
        assert provider.name == "vertex"

    def test_display_name(self):
        from plugins.image_gen.vertex import VertexImageGenProvider

        provider = VertexImageGenProvider()
        assert provider.display_name == "Google Vertex AI"

    def test_is_available_with_gcloud(self):
        from plugins.image_gen.vertex import VertexImageGenProvider

        provider = VertexImageGenProvider()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert provider.is_available() is True

    def test_is_available_without_gcloud(self):
        from plugins.image_gen.vertex import VertexImageGenProvider

        provider = VertexImageGenProvider()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert provider.is_available() is False

    def test_list_models(self):
        from plugins.image_gen.vertex import VertexImageGenProvider

        provider = VertexImageGenProvider()
        models = provider.list_models()
        assert len(models) == 2
        assert models[0]["id"] == "gemini-3.1-flash-image"
        assert models[1]["id"] == "gemini-3-pro-image"

    def test_generate_missing_gcloud_token(self):
        from plugins.image_gen.vertex import VertexImageGenProvider

        provider = VertexImageGenProvider()
        with patch("subprocess.check_output", side_effect=Exception("gcloud not auth")):
            result = provider.generate(prompt="test")
        assert result["success"] is False
        assert "gcloud_token_error" in result["error_type"]

    def test_successful_generation(self):
        from plugins.image_gen.vertex import VertexImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "data": "dGVzdC1pbWFnZS1kYXRh"  # base64 "test-image-data"
                                }
                            }
                        ]
                    }
                }
            ]
        }

        provider = VertexImageGenProvider()
        with patch("subprocess.check_output", return_value=b"fake-token\n"), \
             patch("plugins.image_gen.vertex.requests.post", return_value=mock_resp) as mock_post, \
             patch("plugins.image_gen.vertex.save_b64_image", return_value="/tmp/vertex.png"):
            result = provider.generate(prompt="A beautiful sunset", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["image"] == "/tmp/vertex.png"
        assert result["provider"] == "vertex"
        assert result["model"] == "gemini-3.1-flash-image"

        # Verify payload structure
        call_args = mock_post.call_args
        json_payload = call_args.kwargs.get("json") or call_args[1].get("json")
        prompt_text = json_payload["contents"][0]["parts"][0]["text"]
        assert "A beautiful sunset" in prompt_text
        assert "Landscape aspect ratio" in prompt_text


class TestRegistration:
    def test_register(self):
        from plugins.image_gen.vertex import VertexImageGenProvider, register

        mock_ctx = MagicMock()
        register(mock_ctx)
        mock_ctx.register_image_gen_provider.assert_called_once()
        provider = mock_ctx.register_image_gen_provider.call_args[0][0]
        assert isinstance(provider, VertexImageGenProvider)
        assert provider.name == "vertex"
