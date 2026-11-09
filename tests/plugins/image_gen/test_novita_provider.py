#!/usr/bin/env python3
"""Tests for NovitaAI image generation provider."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Ensure NOVITA_API_KEY is set for all tests."""
    monkeypatch.setenv("NOVITA_API_KEY", "test-key-12345")


def _submit_response(task_id: str = "00000000-0000-0000-0000-000000000abc"):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"task_id": task_id}
    return resp


def _poll_response(body: dict):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = body
    return resp


def _succeeded_task(url: str = "https://novita.cdn/img.jpeg") -> dict:
    return {
        "extra": {},
        "task": {
            "task_id": "00000000-0000-0000-0000-000000000abc",
            "status": "TASK_STATUS_SUCCEED",
            "reason": "",
            "task_type": "TXT_TO_IMG",
            "eta": 0,
            "progress_percent": 100,
        },
        "images": [{"image_url": url, "image_url_ttl": 3600, "image_type": "jpeg"}],
        "videos": [],
        "audios": [],
    }


def _failed_task(reason: str = "unknown error") -> dict:
    return {
        "extra": {},
        "task": {
            "task_id": "00000000-0000-0000-0000-000000000abc",
            "status": "TASK_STATUS_FAILED",
            "reason": reason,
            "task_type": "TXT_TO_IMG",
            "eta": 0,
            "progress_percent": 0,
        },
        "images": [],
        "videos": [],
        "audios": [],
    }


# ---------------------------------------------------------------------------
# Provider class tests
# ---------------------------------------------------------------------------


class TestNovitaImageGenProvider:
    def test_name(self):
        from plugins.image_gen.novita import NovitaImageGenProvider

        assert NovitaImageGenProvider().name == "novita"

    def test_display_name(self):
        from plugins.image_gen.novita import NovitaImageGenProvider

        assert NovitaImageGenProvider().display_name == "NovitaAI"

    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("NOVITA_API_KEY", "sk-test")
        from plugins.image_gen.novita import NovitaImageGenProvider

        assert NovitaImageGenProvider().is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("NOVITA_API_KEY", raising=False)
        from plugins.image_gen.novita import NovitaImageGenProvider

        assert NovitaImageGenProvider().is_available() is False

    def test_list_models(self):
        from plugins.image_gen.novita import NovitaImageGenProvider

        models = NovitaImageGenProvider().list_models()
        ids = {m["id"] for m in models}
        assert {"z-image-turbo", "qwen-image-txt2img"} <= ids
        for m in models:
            assert m["display"]
            assert m["speed"]
            assert m["strengths"]

    def test_default_model_is_z_image_turbo(self):
        from plugins.image_gen.novita import NovitaImageGenProvider

        assert NovitaImageGenProvider().default_model() == "z-image-turbo"

    def test_get_setup_schema(self):
        from plugins.image_gen.novita import NovitaImageGenProvider

        schema = NovitaImageGenProvider().get_setup_schema()
        assert schema["name"] == "NovitaAI"
        assert schema["badge"] == "paid"
        env_vars = schema["env_vars"]
        assert len(env_vars) == 1
        assert env_vars[0]["key"] == "NOVITA_API_KEY"
        assert "novita.ai" in env_vars[0]["url"]

    def test_capabilities_text_only(self):
        from plugins.image_gen.novita import NovitaImageGenProvider

        caps = NovitaImageGenProvider().capabilities()
        assert caps["modalities"] == ["text"]
        assert caps["max_reference_images"] == 0


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


class TestModelResolution:
    def test_env_override_qwen(self, monkeypatch):
        monkeypatch.setenv("NOVITA_IMAGE_MODEL", "qwen-image-txt2img")
        from plugins.image_gen.novita import _resolve_model

        model_id, meta = _resolve_model()
        assert model_id == "qwen-image-txt2img"
        assert meta["path"] == "qwen-image-txt2img"

    def test_default_when_no_override(self, monkeypatch):
        monkeypatch.delenv("NOVITA_IMAGE_MODEL", raising=False)
        from plugins.image_gen.novita import _resolve_model

        model_id, _meta = _resolve_model()
        assert model_id == "z-image-turbo"


# ---------------------------------------------------------------------------
# Generate — main flow
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("NOVITA_API_KEY", raising=False)
        from plugins.image_gen.novita import NovitaImageGenProvider

        result = NovitaImageGenProvider().generate(prompt="test")
        assert result["success"] is False
        assert "NOVITA_API_KEY" in result["error"]
        assert result["error_type"] == "auth_required"

    def test_empty_prompt(self):
        from plugins.image_gen.novita import NovitaImageGenProvider

        result = NovitaImageGenProvider().generate(prompt="   ")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_image_url_unsupported(self):
        from plugins.image_gen.novita import NovitaImageGenProvider

        result = NovitaImageGenProvider().generate(
            prompt="test", image_url="https://example.com/a.png"
        )
        assert result["success"] is False
        assert result["error_type"] == "modality_unsupported"

    def test_successful_generation(self):
        """Happy path: submit → one poll → succeeded → URL downloaded."""
        from plugins.image_gen.novita import NovitaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_succeeded_task("https://novita.cdn/result.jpeg"))

        with patch("plugins.image_gen.novita.requests.post", return_value=submit) as mock_post, \
             patch("plugins.image_gen.novita.requests.get", return_value=poll) as mock_get, \
             patch(
                 "plugins.image_gen.novita.save_url_image",
                 return_value=Path("/tmp/novita_z-image-turbo_test.jpeg"),
             ) as mock_save, \
             patch("plugins.image_gen.novita.time.sleep"):
            result = NovitaImageGenProvider().generate(prompt="A cinematic lamp")

        assert result["success"] is True
        assert result["image"] == "/tmp/novita_z-image-turbo_test.jpeg"
        assert result["provider"] == "novita"
        assert result["model"] == "z-image-turbo"
        assert result["aspect_ratio"] == "landscape"
        assert result["task_id"] == "00000000-0000-0000-0000-000000000abc"
        # Submit hit the z-image-turbo endpoint
        post_url = mock_post.call_args[0][0]
        assert post_url.endswith("/async/z-image-turbo")
        # Poll hit /async/task-result
        poll_url = mock_get.call_args[0][0]
        assert poll_url.endswith("/async/task-result")
        mock_save.assert_called_once()

    def test_qwen_model_routes_to_qwen_endpoint(self, monkeypatch):
        monkeypatch.setenv("NOVITA_IMAGE_MODEL", "qwen-image-txt2img")
        from plugins.image_gen.novita import NovitaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_succeeded_task())

        with patch("plugins.image_gen.novita.requests.post", return_value=submit) as mock_post, \
             patch("plugins.image_gen.novita.requests.get", return_value=poll), \
             patch(
                 "plugins.image_gen.novita.save_url_image",
                 return_value=Path("/tmp/x.jpeg"),
             ), \
             patch("plugins.image_gen.novita.time.sleep"):
            NovitaImageGenProvider().generate(prompt="test")

        post_url = mock_post.call_args[0][0]
        assert post_url.endswith("/async/qwen-image-txt2img")

    def test_failed_task_returns_error(self):
        from plugins.image_gen.novita import NovitaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_failed_task("nsfw content detected"))

        with patch("plugins.image_gen.novita.requests.post", return_value=submit), \
             patch("plugins.image_gen.novita.requests.get", return_value=poll), \
             patch("plugins.image_gen.novita.time.sleep"):
            result = NovitaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "nsfw content detected" in result["error"]

    def test_submit_missing_task_id(self):
        from plugins.image_gen.novita import NovitaImageGenProvider

        submit = _submit_response()
        submit.json.return_value = {}

        with patch("plugins.image_gen.novita.requests.post", return_value=submit), \
             patch("plugins.image_gen.novita.time.sleep"):
            result = NovitaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "invalid_response"

    def test_no_image_url_in_success(self):
        from plugins.image_gen.novita import NovitaImageGenProvider

        submit = _submit_response()
        succeeded = _succeeded_task()
        succeeded["images"] = []
        poll = _poll_response(succeeded)

        with patch("plugins.image_gen.novita.requests.post", return_value=submit), \
             patch("plugins.image_gen.novita.requests.get", return_value=poll), \
             patch("plugins.image_gen.novita.time.sleep"):
            result = NovitaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_calls_ctx(self):
        from plugins.image_gen.novita import NovitaImageGenProvider, register

        ctx = MagicMock()
        register(ctx)
        ctx.register_image_gen_provider.assert_called_once()
        (provider_arg,) = ctx.register_image_gen_provider.call_args[0]
        assert isinstance(provider_arg, NovitaImageGenProvider)
