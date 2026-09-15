"""Tests for the native Google AI Studio image-generation plugin."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import plugins.image_gen.gemini as gemini_plugin


class _Response:
    status_code = 200
    reason = "OK"

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_IMAGE_MODEL", raising=False)
    yield tmp_path


def _image_response(data: bytes = b"image-bytes"):
    return _Response({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": base64.b64encode(data).decode("ascii"),
                            }
                        }
                    ]
                }
            }
        ]
    })


def test_metadata_and_setup_schema():
    provider = gemini_plugin.GeminiImageGenProvider()

    assert provider.name == "gemini"
    assert provider.display_name == "Google AI Studio"
    models = provider.list_models()
    assert provider.default_model() == gemini_plugin.DEFAULT_MODEL
    assert gemini_plugin.DEFAULT_MODEL in {item["id"] for item in models}
    assert all(
        {"id", "display", "speed", "strengths", "price"} <= set(item) for item in models
    )
    assert provider.capabilities() == {
        "modalities": ["text", "image"],
        "max_reference_images": 14,
    }
    assert [item["key"] for item in provider.get_setup_schema()["env_vars"]] == [
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ]


def test_model_precedence(monkeypatch):
    monkeypatch.setenv("GEMINI_IMAGE_MODEL", "from-env")
    monkeypatch.setattr(
        gemini_plugin,
        "_load_image_gen_config",
        lambda: {"gemini": {"model": "from-config"}, "model": "from-top-level"},
    )

    assert gemini_plugin._resolve_model("from-call") == "from-call"
    assert gemini_plugin._resolve_model() == "from-env"

    monkeypatch.delenv("GEMINI_IMAGE_MODEL")
    assert gemini_plugin._resolve_model() == "from-config"

    monkeypatch.setattr(
        gemini_plugin, "_load_image_gen_config", lambda: {"model": "from-top-level"}
    )
    assert gemini_plugin._resolve_model() == "from-top-level"


def test_generate_posts_native_gemini_payload_and_caches_inline_image(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
    response = _image_response()
    post = MagicMock(return_value=response)

    with patch("requests.post", post):
        result = gemini_plugin.GeminiImageGenProvider().generate(
            "a red kite",
            aspect_ratio="portrait",
            model="gemini-2.5-flash-image",
        )

    assert result["success"] is True
    assert result["provider"] == "gemini"
    assert result["model"] == "gemini-2.5-flash-image"
    assert result["aspect_ratio"] == "portrait"
    assert result["modality"] == "text"
    assert Path(result["image"]).read_bytes() == b"image-bytes"

    post.assert_called_once()
    (endpoint,) = post.call_args.args
    assert endpoint.endswith("/models/gemini-2.5-flash-image:generateContent")
    assert "params" not in post.call_args.kwargs
    assert post.call_args.kwargs["headers"]["x-goog-api-key"] == "google-test-key"
    assert post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
    payload = post.call_args.kwargs["json"]
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "a red kite"}]}]
    assert payload["generationConfig"] == {
        "responseModalities": ["TEXT", "IMAGE"],
        "imageConfig": {"aspectRatio": "9:16"},
    }


def test_generate_inlines_reference_image(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    post = MagicMock(return_value=_image_response())

    with patch("requests.post", post):
        result = gemini_plugin.GeminiImageGenProvider().generate(
            "make it snowy",
            image_url=str(reference),
        )

    assert result["success"] is True
    assert result["modality"] == "image"
    parts = post.call_args.kwargs["json"]["contents"][0]["parts"]
    assert parts[0] == {"text": "make it snowy"}
    assert parts[1]["inlineData"]["mimeType"] == "image/png"
    assert base64.b64decode(parts[1]["inlineData"]["data"]) == b"reference"


def test_generate_respects_legacy_model_reference_limit(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
    reference = "data:image/png;base64," + base64.b64encode(b"reference").decode(
        "ascii"
    )
    post = MagicMock(return_value=_image_response())

    with patch("requests.post", post):
        result = gemini_plugin.GeminiImageGenProvider().generate(
            "combine these references",
            model="gemini-2.5-flash-image",
            reference_image_urls=[reference] * 5,
        )

    assert result["success"] is True
    parts = post.call_args.kwargs["json"]["contents"][0]["parts"]
    assert len(parts) == 4


def test_empty_response_includes_model_text(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
    response = _Response({
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "The request was blocked by safety filters."}]
                },
                "finishReason": "SAFETY",
            }
        ]
    })

    with patch("requests.post", MagicMock(return_value=response)):
        result = gemini_plugin.GeminiImageGenProvider().generate("a disallowed image")

    assert result["success"] is False
    assert result["error_type"] == "empty_response"
    assert "safety filters" in result["error"]


def test_invalid_json_response_returns_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("not json")

    with patch("requests.post", MagicMock(return_value=response)):
        result = gemini_plugin.GeminiImageGenProvider().generate("a cat")

    assert result["success"] is False
    assert result["error_type"] == "invalid_response"


def test_missing_key_returns_auth_error(monkeypatch):
    result = gemini_plugin.GeminiImageGenProvider().generate("a cat")

    assert result["success"] is False
    assert result["error_type"] == "auth_required"
