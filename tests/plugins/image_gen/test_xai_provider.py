#!/usr/bin/env python3
"""Tests for xAI image generation provider."""

from __future__ import annotations

import json
import os
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
        assert models[0]["id"] == "grok-imagine-image"

    def test_default_model(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        assert provider.default_model() == "grok-imagine-image"

    def test_get_setup_schema(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        schema = provider.get_setup_schema()
        assert schema["name"] == "xAI (Grok)"
        assert schema["badge"] == "paid"
        assert len(schema["env_vars"]) == 1
        assert schema["env_vars"][0]["key"] == "XAI_API_KEY"


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
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"url": "https://xai.image/result.png"}],
        }

        with patch("plugins.image_gen.xai.requests.post", return_value=mock_resp):
            provider = XAIImageGenProvider()
            result = provider.generate(prompt="A cat playing piano")

        assert result["success"] is True
        assert result["image"] == "https://xai.image/result.png"

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
