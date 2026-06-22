#!/usr/bin/env python3
"""Basic tests for OpenRouter image generation provider."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Ensure OPENROUTER_API_KEY is set for all tests."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-12345")


class TestOpenRouterImageGenProvider:
    def test_name(self):
        from plugins.image_gen.openrouter import OpenRouterImageGenProvider

        provider = OpenRouterImageGenProvider()
        assert provider.name == "openrouter"

    def test_display_name(self):
        from plugins.image_gen.openrouter import OpenRouterImageGenProvider

        provider = OpenRouterImageGenProvider()
        assert provider.display_name == "OpenRouter"

    def test_is_available_with_key(self):
        from plugins.image_gen.openrouter import OpenRouterImageGenProvider

        provider = OpenRouterImageGenProvider()
        assert provider.is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from plugins.image_gen.openrouter import OpenRouterImageGenProvider

        provider = OpenRouterImageGenProvider()
        assert provider.is_available() is False

    def test_list_models(self):
        from plugins.image_gen.openrouter import OpenRouterImageGenProvider

        provider = OpenRouterImageGenProvider()
        models = provider.list_models()
        assert len(models) >= 3
        ids = [m["id"] for m in models]
        assert "google/gemini-2.5-flash-image-preview" in ids
        assert "black-forest-labs/flux.2-pro" in ids

    def test_default_model(self):
        from plugins.image_gen.openrouter import OpenRouterImageGenProvider

        provider = OpenRouterImageGenProvider()
        assert provider.default_model() == "google/gemini-2.5-flash-image-preview"

    def test_get_setup_schema(self):
        from plugins.image_gen.openrouter import OpenRouterImageGenProvider

        provider = OpenRouterImageGenProvider()
        schema = provider.get_setup_schema()
        assert schema["name"] == "OpenRouter"
        assert schema["badge"] == "paid"
        assert any(v["key"] == "OPENROUTER_API_KEY" for v in schema.get("env_vars", []))

    def test_capabilities(self):
        from plugins.image_gen.openrouter import OpenRouterImageGenProvider

        provider = OpenRouterImageGenProvider()
        caps = provider.capabilities()
        assert "text" in caps.get("modalities", [])
        assert "image" in caps.get("modalities", [])
        assert caps.get("max_reference_images", 0) >= 4

    def test_generate_text_to_image_success(self, monkeypatch):
        """Smoke test for generate path (mocked network)."""
        from plugins.image_gen.openrouter import OpenRouterImageGenProvider

        provider = OpenRouterImageGenProvider()

        mock_response = {
            "choices": [
                {
                    "message": {
                        "images": [
                            {"image_url": {"url": "https://example.com/generated.png"}}
                        ]
                    }
                }
            ]
        }

        with patch("plugins.image_gen.openrouter.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None

            result = provider.generate("A cute robot in a garden", aspect_ratio="square")

            assert result["success"] is True
            assert result["modality"] == "text"
            assert result["model"] == "google/gemini-2.5-flash-image-preview"
            assert "image" in result

    def test_generate_image_to_image_routes_correctly(self, monkeypatch):
        from plugins.image_gen.openrouter import OpenRouterImageGenProvider

        provider = OpenRouterImageGenProvider()

        mock_response = {
            "choices": [
                {
                    "message": {
                        "images": [
                            {"image_url": {"url": "https://example.com/edited.png"}}
                        ]
                    }
                }
            ]
        }

        with patch("plugins.image_gen.openrouter.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None

            result = provider.generate(
                "Make the background transparent",
                image_url="https://example.com/source.png",
            )

            assert result["success"] is True
            assert result["modality"] == "image"
