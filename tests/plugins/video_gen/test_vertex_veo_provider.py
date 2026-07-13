#!/usr/bin/env python3
"""Tests for Google Vertex AI Veo video generation provider plugin."""

from __future__ import annotations

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


class TestVertexVeoVideoGenProvider:
    def test_name(self):
        from plugins.video_gen.vertex_veo import VertexVeoVideoGenProvider

        provider = VertexVeoVideoGenProvider()
        assert provider.name == "vertex_veo"

    def test_display_name(self):
        from plugins.video_gen.vertex_veo import VertexVeoVideoGenProvider

        provider = VertexVeoVideoGenProvider()
        assert provider.display_name == "Google Vertex AI Veo"

    def test_is_available_with_gcloud(self):
        from plugins.video_gen.vertex_veo import VertexVeoVideoGenProvider

        provider = VertexVeoVideoGenProvider()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert provider.is_available() is True

    def test_is_available_without_gcloud(self):
        from plugins.video_gen.vertex_veo import VertexVeoVideoGenProvider

        provider = VertexVeoVideoGenProvider()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert provider.is_available() is False

    def test_list_models(self):
        from plugins.video_gen.vertex_veo import VertexVeoVideoGenProvider

        provider = VertexVeoVideoGenProvider()
        models = provider.list_models()
        assert len(models) == 1
        assert models[0]["id"] == "veo-2.0-generate-001"

    def test_default_model(self):
        from plugins.video_gen.vertex_veo import VertexVeoVideoGenProvider

        provider = VertexVeoVideoGenProvider()
        assert provider.default_model() == "veo-2.0-generate-001"

    def test_capabilities(self):
        from plugins.video_gen.vertex_veo import VertexVeoVideoGenProvider

        provider = VertexVeoVideoGenProvider()
        caps = provider.capabilities()
        assert caps["aspect_ratios"] == ["16:9", "9:16"]
        assert caps["resolutions"] == ["720p", "1080p"]

    def test_get_setup_schema(self):
        from plugins.video_gen.vertex_veo import VertexVeoVideoGenProvider

        provider = VertexVeoVideoGenProvider()
        schema = provider.get_setup_schema()
        assert "gcloud" in schema["badge"]

    def test_generate_missing_gcloud_token(self):
        from plugins.video_gen.vertex_veo import VertexVeoVideoGenProvider

        provider = VertexVeoVideoGenProvider()
        with patch("subprocess.check_output", side_effect=Exception("gcloud not auth")):
            result = provider.generate(prompt="test")
        assert result["success"] is False
        assert "gcloud_token_error" in result["error_type"]

    def test_successful_generation(self):
        from plugins.video_gen.vertex_veo import VertexVeoVideoGenProvider

        # Mock predict response (initiating LRO)
        mock_predict_resp = MagicMock()
        mock_predict_resp.status_code = 200
        mock_predict_resp.json.return_value = {
            "name": "projects/winter-environs-427409-r8/locations/us-central1/operations/12345"
        }

        # Mock polling response (completed LRO)
        mock_poll_resp = MagicMock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {
            "done": True,
            "response": {
                "videos": [
                    {
                        "bytesBase64Encoded": "dGVzdC12aWRlbw=="  # base64 data
                    }
                ]
            }
        }

        provider = VertexVeoVideoGenProvider()
        with patch("subprocess.check_output", return_value=b"fake-token\n"), \
             patch("plugins.video_gen.vertex_veo.requests.post") as mock_post, \
             patch("plugins.video_gen.vertex_veo.save_b64_video", return_value="/tmp/veo.mp4"), \
             patch("time.sleep") as mock_sleep:  # mock sleep to speed up test
            
            # The first post is prediction request, second is polling request
            mock_post.side_effect = [mock_predict_resp, mock_poll_resp]
            
            result = provider.generate(prompt="A cinematic video of a flying dragon", duration=5)

        assert result["success"] is True
        assert result["video"] == "/tmp/veo.mp4"
        assert result["provider"] == "vertex_veo"
        assert result["model"] == "veo-2.0-generate-001"

        # Verify that post was called twice: once for submission, once for polling
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(10)


class TestRegistration:
    def test_register(self):
        from plugins.video_gen.vertex_veo import VertexVeoVideoGenProvider, register

        mock_ctx = MagicMock()
        register(mock_ctx)
        mock_ctx.register_video_gen_provider.assert_called_once()
        provider = mock_ctx.register_video_gen_provider.call_args[0][0]
        assert isinstance(provider, VertexVeoVideoGenProvider)
        assert provider.name == "vertex_veo"
