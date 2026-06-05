#!/usr/bin/env python3
"""Tests for Leonardo.AI image generation provider."""

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
    """Ensure LEONARDO_API_KEY is set for all tests."""
    monkeypatch.setenv("LEONARDO_API_KEY", "test-key-12345")


# ---------------------------------------------------------------------------
# Provider class tests
# ---------------------------------------------------------------------------


class TestLeonardoImageGenProvider:
    def test_name(self):
        from plugins.image_gen.leonardo import LeonardoImageGenProvider

        provider = LeonardoImageGenProvider()
        assert provider.name == "leonardo"

    def test_display_name(self):
        from plugins.image_gen.leonardo import LeonardoImageGenProvider

        provider = LeonardoImageGenProvider()
        assert provider.display_name == "Leonardo.AI"

    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("LEONARDO_API_KEY", "leo-xxx")
        from plugins.image_gen.leonardo import LeonardoImageGenProvider

        provider = LeonardoImageGenProvider()
        assert provider.is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("LEONARDO_API_KEY", raising=False)
        from plugins.image_gen.leonardo import LeonardoImageGenProvider

        provider = LeonardoImageGenProvider()
        assert provider.is_available() is False

    def test_list_models(self):
        from plugins.image_gen.leonardo import LeonardoImageGenProvider

        provider = LeonardoImageGenProvider()
        models = provider.list_models()
        assert len(models) == 3
        model_ids = [m["id"] for m in models]
        assert "phoenix" in model_ids
        assert "sdxl" in model_ids
        assert "flux" in model_ids

    def test_default_model(self):
        from plugins.image_gen.leonardo import LeonardoImageGenProvider

        provider = LeonardoImageGenProvider()
        assert provider.default_model() == "phoenix"

    def test_get_setup_schema(self):
        from plugins.image_gen.leonardo import LeonardoImageGenProvider

        provider = LeonardoImageGenProvider()
        schema = provider.get_setup_schema()
        assert schema["name"] == "Leonardo.AI"
        assert schema["badge"] == "free-tier"
        assert len(schema["env_vars"]) == 1
        assert schema["env_vars"][0]["key"] == "LEONARDO_API_KEY"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_model(self):
        from plugins.image_gen.leonardo import _resolve_model

        model_key, model_id, sd_version = _resolve_model()
        assert model_key == "phoenix"
        assert sd_version == "PHOENIX"

    def test_custom_model_via_env(self, monkeypatch):
        monkeypatch.setenv("LEONARDO_IMAGE_MODEL", "flux")
        from plugins.image_gen.leonardo import _resolve_model

        model_key, _, sd_version = _resolve_model()
        assert model_key == "flux"
        assert sd_version == "FLUX"


# ---------------------------------------------------------------------------
# Generate tests
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("LEONARDO_API_KEY", raising=False)
        from plugins.image_gen.leonardo import LeonardoImageGenProvider

        provider = LeonardoImageGenProvider()
        result = provider.generate(prompt="test")
        assert result["success"] is False
        assert "LEONARDO_API_KEY" in result["error"]

    def test_successful_generation(self):
        from plugins.image_gen.leonardo import LeonardoImageGenProvider

        # Mock the POST /generations response
        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.raise_for_status = MagicMock()
        create_resp.json.return_value = {
            "sdGenerationJob": {
                "generationId": "gen-123",
                "apiCreditCost": 5,
            }
        }

        # Mock the GET /generations/{id} polling response
        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.raise_for_status = MagicMock()
        poll_resp.json.return_value = {
            "generations_by_pk": {
                "status": "COMPLETE",
                "generated_images": [
                    {"url": "https://cdn.leonardo.ai/gen-123_0.png"},
                ],
            }
        }

        with (
            patch("plugins.image_gen.leonardo.requests.post", return_value=create_resp),
            patch("plugins.image_gen.leonardo.requests.get", return_value=poll_resp),
            patch(
                "plugins.image_gen.leonardo.save_url_image",
                return_value=Path("/tmp/test.png"),
            ),
            patch("plugins.image_gen.leonardo.time.sleep"),
        ):
            provider = LeonardoImageGenProvider()
            result = provider.generate(prompt="A cat playing piano")

        assert result["success"] is True
        assert result["image"] == "/tmp/test.png"
        assert result["provider"] == "leonardo"
        assert result["model"] == "phoenix"

    def test_api_error(self):
        import requests as req_lib
        from plugins.image_gen.leonardo import LeonardoImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_resp.raise_for_status.side_effect = req_lib.HTTPError(response=mock_resp)

        with patch("plugins.image_gen.leonardo.requests.post", return_value=mock_resp):
            provider = LeonardoImageGenProvider()
            result = provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"

    def test_generation_failed_status(self):
        from plugins.image_gen.leonardo import LeonardoImageGenProvider

        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.raise_for_status = MagicMock()
        create_resp.json.return_value = {
            "sdGenerationJob": {"generationId": "gen-fail", "apiCreditCost": 0}
        }

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.raise_for_status = MagicMock()
        poll_resp.json.return_value = {"generations_by_pk": {"status": "FAILED"}}

        with (
            patch("plugins.image_gen.leonardo.requests.post", return_value=create_resp),
            patch("plugins.image_gen.leonardo.requests.get", return_value=poll_resp),
            patch("plugins.image_gen.leonardo.time.sleep"),
        ):
            provider = LeonardoImageGenProvider()
            result = provider.generate(prompt="test")

        assert result["success"] is False
        assert "failed" in result["error"].lower()

    def test_timeout(self):
        from plugins.image_gen.leonardo import LeonardoImageGenProvider

        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.raise_for_status = MagicMock()
        create_resp.json.return_value = {
            "sdGenerationJob": {"generationId": "gen-slow", "apiCreditCost": 0}
        }

        # Poll always returns PENDING
        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.raise_for_status = MagicMock()
        poll_resp.json.return_value = {"generations_by_pk": {"status": "PENDING"}}

        # time.time is also called by logging internals, so use a
        # counter function that returns past-deadline after the first
        # poll cycle instead of a finite side_effect list.
        call_count = 0

        def fake_time():
            nonlocal call_count
            call_count += 1
            # First call: start of poll (0). Second call: past deadline (999).
            # All subsequent calls (logging, etc.) also get 999.
            return 0.0 if call_count <= 1 else 999.0

        with (
            patch("plugins.image_gen.leonardo.requests.post", return_value=create_resp),
            patch("plugins.image_gen.leonardo.requests.get", return_value=poll_resp),
            patch("plugins.image_gen.leonardo.time.sleep"),
            patch("plugins.image_gen.leonardo.time.time", side_effect=fake_time),
        ):
            provider = LeonardoImageGenProvider()
            result = provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "timeout"

    def test_auth_header(self):
        from plugins.image_gen.leonardo import LeonardoImageGenProvider

        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.raise_for_status = MagicMock()
        create_resp.json.return_value = {
            "sdGenerationJob": {"generationId": "gen-auth", "apiCreditCost": 1}
        }

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.raise_for_status = MagicMock()
        poll_resp.json.return_value = {
            "generations_by_pk": {
                "status": "COMPLETE",
                "generated_images": [{"url": "https://cdn.leonardo.ai/test.png"}],
            }
        }

        with (
            patch(
                "plugins.image_gen.leonardo.requests.post", return_value=create_resp
            ) as mock_post,
            patch("plugins.image_gen.leonardo.requests.get", return_value=poll_resp),
            patch(
                "plugins.image_gen.leonardo.save_url_image",
                return_value=Path("/tmp/test.png"),
            ),
            patch("plugins.image_gen.leonardo.time.sleep"),
        ):
            provider = LeonardoImageGenProvider()
            provider.generate(prompt="test")

        call_args = mock_post.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert "Bearer test-key-12345" in headers["Authorization"]

    def test_payload_contains_required_fields(self):
        from plugins.image_gen.leonardo import LeonardoImageGenProvider

        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.raise_for_status = MagicMock()
        create_resp.json.return_value = {
            "sdGenerationJob": {"generationId": "gen-fields", "apiCreditCost": 1}
        }

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {
            "generations_by_pk": {
                "status": "COMPLETE",
                "generated_images": [{"url": "https://cdn.leonardo.ai/test.png"}],
            }
        }

        with (
            patch(
                "plugins.image_gen.leonardo.requests.post", return_value=create_resp
            ) as mock_post,
            patch("plugins.image_gen.leonardo.requests.get", return_value=poll_resp),
            patch(
                "plugins.image_gen.leonardo.save_url_image",
                return_value=Path("/tmp/test.png"),
            ),
            patch("plugins.image_gen.leonardo.time.sleep"),
        ):
            provider = LeonardoImageGenProvider()
            provider.generate(prompt="A majestic cat", aspect_ratio="square")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get(
            "json"
        )
        assert payload["prompt"] == "A majestic cat"
        assert payload["width"] == 1024
        assert payload["height"] == 1024
        assert payload["modelId"] is not None
        assert payload["num_images"] == 1

    def test_aspect_ratio_landscape(self):
        from plugins.image_gen.leonardo import _ASPECT_SIZES

        assert _ASPECT_SIZES["landscape"]["width"] == 1024
        assert _ASPECT_SIZES["landscape"]["height"] == 768

    def test_aspect_ratio_portrait(self):
        from plugins.image_gen.leonardo import _ASPECT_SIZES

        assert _ASPECT_SIZES["portrait"]["width"] == 768
        assert _ASPECT_SIZES["portrait"]["height"] == 1024


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register(self):
        from plugins.image_gen.leonardo import LeonardoImageGenProvider, register

        mock_ctx = MagicMock()
        register(mock_ctx)
        mock_ctx.register_image_gen_provider.assert_called_once()
        provider = mock_ctx.register_image_gen_provider.call_args[0][0]
        assert isinstance(provider, LeonardoImageGenProvider)
        assert provider.name == "leonardo"
