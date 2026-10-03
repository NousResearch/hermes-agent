"""Tests for the bundled Gemini (Google AI Studio) image_gen plugin.

Native ``generativelanguage.googleapis.com`` backend using
``GOOGLE_API_KEY``/``GEMINI_API_KEY`` directly — no FAL/OpenRouter proxy.
HTTP is mocked at ``post_json``; no network, no key needed.
"""

from __future__ import annotations

import base64

import pytest

import plugins.image_gen.gemini as gemini_plugin


# 1x1 transparent PNG — valid bytes for save_b64_image()
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


def _ok_body(b64: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": b64}}]}}]}


@pytest.fixture(autouse=True)
def _isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_IMAGE_MODEL", raising=False)
    yield


def test_is_available_requires_google_or_gemini_key(monkeypatch):
    provider = gemini_plugin.GeminiImageGenProvider()
    assert provider.is_available() is False
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert provider.is_available() is True
    monkeypatch.delenv("GOOGLE_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert provider.is_available() is True


def test_generate_requires_key():
    result = gemini_plugin.GeminiImageGenProvider().generate(prompt="a cat", aspect_ratio="square")
    assert result["success"] is False
    assert result["error_type"] == "auth_required"
    assert result["provider"] == "gemini"


def test_generate_requires_prompt(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    result = gemini_plugin.GeminiImageGenProvider().generate(prompt="   ", aspect_ratio="square")
    assert result["success"] is False
    assert result["error_type"] == "invalid_argument"


def test_generate_posts_generate_content_and_saves_b64(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    captured: dict = {}

    def _fake_post_json(url, *, headers, payload, timeout, **kwargs):
        captured["url"] = url
        captured["payload"] = payload
        return _ok_body(_b64_png()), None

    monkeypatch.setattr(gemini_plugin, "post_json", _fake_post_json)
    result = gemini_plugin.GeminiImageGenProvider().generate(prompt="a cat", aspect_ratio="square")

    assert result["success"] is True
    assert result["provider"] == "gemini"
    assert result["aspect_ratio"] == "square"
    assert result["image"] and str(result["image"]).endswith(".png")
    assert "generativelanguage.googleapis.com" in captured["url"]
    assert ":generateContent" in captured["url"]
    assert captured["payload"]["generationConfig"]["responseModalities"] == ["TEXT", "IMAGE"]
    text_parts = [
        p.get("text") for p in captured["payload"]["contents"][0]["parts"] if "text" in p
    ]
    assert text_parts == ["a cat"]


def test_generate_maps_aspect_ratio(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    captured: dict = {}
    monkeypatch.setattr(
        gemini_plugin, "post_json",
        lambda url, **kw: (captured.update(payload=kw["payload"]) or (_ok_body(_b64_png()), None)),
    )
    gemini_plugin.GeminiImageGenProvider().generate(prompt="a cat", aspect_ratio="portrait")
    assert captured["payload"]["generationConfig"]["imageConfig"] == {"aspectRatio": "9:16"}


def test_generate_model_precedence(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    captured: dict = {}
    monkeypatch.setattr(
        gemini_plugin, "post_json",
        lambda url, **kw: (captured.update(url=url) or (_ok_body(_b64_png()), None)),
    )
    provider = gemini_plugin.GeminiImageGenProvider()
    monkeypatch.setenv("GEMINI_IMAGE_MODEL", gemini_plugin.DEFAULT_MODEL)
    provider.generate(prompt="a cat", model="gemini-2.5-flash-image")
    assert "gemini-2.5-flash-image" in captured["url"]


def test_generate_surfaces_api_failure(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    from plugins.image_gen._common import HttpFailure

    monkeypatch.setattr(
        gemini_plugin, "post_json",
        lambda url, **kw: (None, HttpFailure("http", "Gemini image generation failed (400): bad", "api_error")),
    )
    result = gemini_plugin.GeminiImageGenProvider().generate(prompt="a cat")
    assert result["success"] is False
    assert result["error_type"] == "api_error"


def test_generate_empty_response_is_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(gemini_plugin, "post_json", lambda url, **kw: ({"candidates": []}, None))
    result = gemini_plugin.GeminiImageGenProvider().generate(prompt="a cat")
    assert result["success"] is False
    assert result["error_type"] == "empty_response"


def test_generate_inlines_local_reference_image(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    ref = tmp_path / "ref.png"
    ref.write_bytes(bytes.fromhex(_PNG_HEX))
    captured: dict = {}
    monkeypatch.setattr(
        gemini_plugin, "post_json",
        lambda url, **kw: (captured.update(payload=kw["payload"]) or (_ok_body(_b64_png()), None)),
    )
    result = gemini_plugin.GeminiImageGenProvider().generate(
        prompt="edit this", reference_image_urls=[str(ref)],
    )
    assert result["success"] is True
    inline_parts = [p for p in captured["payload"]["contents"][0]["parts"] if "inlineData" in p]
    assert len(inline_parts) == 1
    assert inline_parts[0]["inlineData"]["data"] == _b64_png()
