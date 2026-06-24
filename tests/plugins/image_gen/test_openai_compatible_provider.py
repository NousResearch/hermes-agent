from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestOpenAICompatibleImageGenProvider:
    def test_name(self):
        from plugins.image_gen.openai_compatible import OpenAICompatibleImageGenProvider

        provider = OpenAICompatibleImageGenProvider()
        assert provider.name == "openai-compatible"

    def test_is_available_uses_configured_base_url(self, monkeypatch):
        from plugins.image_gen.openai_compatible import OpenAICompatibleImageGenProvider

        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "http://localhost:20128/v1")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        with patch("plugins.image_gen.openai_compatible.requests.get", return_value=mock_resp) as mock_get:
            assert OpenAICompatibleImageGenProvider().is_available() is True

        assert mock_get.call_args.args[0] == "http://localhost:20128/v1/images/generations"

    def test_list_models_reads_image_generation_catalog(self, monkeypatch):
        from plugins.image_gen.openai_compatible import OpenAICompatibleImageGenProvider

        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "http://localhost:20128/v1")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "id": "together/black-forest-labs/FLUX.2-pro",
                    "input_modalities": ["text", "image"],
                    "supported_sizes": ["1024x1024"],
                },
                {
                    "id": "together/black-forest-labs/FLUX.2-dev",
                    "input_modalities": ["text"],
                    "supported_sizes": ["1024x1024"],
                },
            ]
        }

        with patch("plugins.image_gen.openai_compatible.requests.get", return_value=mock_resp):
            models = OpenAICompatibleImageGenProvider().list_models()

        assert models == [
            {
                "id": "together/black-forest-labs/FLUX.2-pro",
                "display": "together/black-forest-labs/FLUX.2-pro",
                "speed": "",
                "strengths": "text + image",
                "price": "local proxy",
            },
            {
                "id": "together/black-forest-labs/FLUX.2-dev",
                "display": "together/black-forest-labs/FLUX.2-dev",
                "speed": "",
                "strengths": "text",
                "price": "local proxy",
            },
        ]

    def test_reference_image_payload_uses_image_url_file_uri(self, monkeypatch, tmp_path):
        from plugins.image_gen.openai_compatible import OpenAICompatibleImageGenProvider

        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "http://localhost:20128/v1")
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_MODEL", "together/black-forest-labs/FLUX.2-pro")
        reference = tmp_path / "reference.png"
        reference.write_bytes(b"fake image bytes")

        catalog_resp = MagicMock()
        catalog_resp.raise_for_status = MagicMock()
        catalog_resp.json.return_value = {
            "data": [{
                "id": "together/black-forest-labs/FLUX.2-pro",
                "input_modalities": ["text", "image"],
                "supported_sizes": ["1024x1024"],
            }]
        }
        gen_resp = MagicMock()
        gen_resp.raise_for_status = MagicMock()
        gen_resp.json.return_value = {"data": [{"url": "http://localhost/result.png"}]}

        with patch("plugins.image_gen.openai_compatible.requests.get", return_value=catalog_resp), \
             patch("plugins.image_gen.openai_compatible.requests.post", return_value=gen_resp) as mock_post:
            result = OpenAICompatibleImageGenProvider().generate(
                prompt="use reference",
                aspect_ratio="square",
                image_url=str(reference),
            )

        assert result["success"] is True
        assert result["image"] == "http://localhost/result.png"
        sent = mock_post.call_args.kwargs["json"]
        assert sent["model"] == "together/black-forest-labs/FLUX.2-pro"
        assert sent["prompt"] == "use reference"
        assert sent["image_url"] == reference.resolve().as_uri()
        assert sent["size"] == "1024x1024"

    def test_reference_image_rejected_when_selected_model_is_text_only(self, monkeypatch, tmp_path):
        from plugins.image_gen.openai_compatible import OpenAICompatibleImageGenProvider

        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_MODEL", "text-only-model")
        reference = tmp_path / "reference.png"
        reference.write_bytes(b"fake image bytes")

        catalog_resp = MagicMock()
        catalog_resp.raise_for_status = MagicMock()
        catalog_resp.json.return_value = {
            "data": [{
                "id": "text-only-model",
                "input_modalities": ["text"],
                "supported_sizes": ["1024x1024"],
            }]
        }

        with patch("plugins.image_gen.openai_compatible.requests.get", return_value=catalog_resp), \
             patch("plugins.image_gen.openai_compatible.requests.post") as mock_post:
            result = OpenAICompatibleImageGenProvider().generate(
                prompt="use reference",
                image_url=str(reference),
            )

        assert result["success"] is False
        assert result["error_type"] == "unsupported_reference_image"
        mock_post.assert_not_called()

    def test_missing_reference_image_rejected_before_network(self, monkeypatch, tmp_path):
        from plugins.image_gen.openai_compatible import OpenAICompatibleImageGenProvider

        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_MODEL", "together/black-forest-labs/FLUX.2-pro")
        missing = tmp_path / "missing.png"

        with patch("plugins.image_gen.openai_compatible.requests.get") as mock_get, \
             patch("plugins.image_gen.openai_compatible.requests.post") as mock_post:
            result = OpenAICompatibleImageGenProvider().generate(
                prompt="use reference",
                image_url=str(missing),
            )

        assert result["success"] is False
        assert result["error_type"] == "invalid_reference_image"
        mock_get.assert_not_called()
        mock_post.assert_not_called()

    def test_generation_timeout_reads_image_gen_config(self, monkeypatch):
        from plugins.image_gen.openai_compatible import OpenAICompatibleImageGenProvider

        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_MODEL", "cx/gpt-5.5")

        catalog_resp = MagicMock()
        catalog_resp.raise_for_status = MagicMock()
        catalog_resp.json.return_value = {
            "data": [{
                "id": "cx/gpt-5.5",
                "input_modalities": ["text"],
                "supported_sizes": ["1024x1024"],
            }]
        }
        gen_resp = MagicMock()
        gen_resp.raise_for_status = MagicMock()
        gen_resp.json.return_value = {"data": [{"url": "http://localhost/result.png"}]}

        with patch("hermes_cli.config.load_config", return_value={"image_gen": {"timeout": 420}}), \
             patch("plugins.image_gen.openai_compatible.requests.get", return_value=catalog_resp), \
             patch("plugins.image_gen.openai_compatible.requests.post", return_value=gen_resp) as mock_post:
            result = OpenAICompatibleImageGenProvider().generate(
                prompt="slow codex image generation",
                aspect_ratio="square",
            )

        assert result["success"] is True
        assert mock_post.call_args.kwargs["timeout"] == 420

    def test_register(self):
        from plugins.image_gen.openai_compatible import OpenAICompatibleImageGenProvider, register

        mock_ctx = MagicMock()
        register(mock_ctx)
        provider = mock_ctx.register_image_gen_provider.call_args.args[0]
        assert isinstance(provider, OpenAICompatibleImageGenProvider)
        assert provider.name == "openai-compatible"
