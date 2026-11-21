"""Essential coverage for the bundled MiniMax H3-Image plugin."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import plugins.image_gen.minimax as minimax_plugin


def _ok():
    return {"data": [{"url": "https://cdn.example/img.png"}]}, None


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    return minimax_plugin.MiniMaxImageGenProvider()


def test_available_with_key(provider):
    assert provider.is_available()
    assert provider.default_model() == "MiniMax-H3-Image"
    assert "image" in provider.capabilities()["modalities"]


def test_generate_uses_aspect_ratio_and_default_4k(provider):
    captured = {}

    def fake_post(url, *, headers, payload, timeout, label):
        captured.update(url=url, payload=payload, timeout=timeout)
        return _ok()

    with patch("plugins.image_gen.minimax.post_json", side_effect=fake_post), patch(
        "plugins.image_gen._common.save_url_image", return_value=Path("/tmp/minimax_saved.png"),
    ):
        result = provider.generate("a cat", aspect_ratio="landscape")
    assert result["success"] is True
    assert result["area"] == "4K"
    assert captured["payload"]["aspect_ratio"] == "16:9"
    assert captured["payload"]["area"] == "4K"
    assert "size" not in captured["payload"]
    assert captured["url"].endswith("/images/generations")
    assert captured["timeout"] == (30, 610)


def test_edit_posts_image_url_to_edits(provider):
    captured = {}

    def fake_post(url, *, headers, payload, timeout, label):
        captured.update(url=url, payload=payload)
        return _ok()

    with patch("plugins.image_gen.minimax.post_json", side_effect=fake_post), patch(
        "plugins.image_gen._common.save_url_image", return_value=Path("/tmp/minimax_saved.png"),
    ):
        result = provider.generate(
            "make it rainy", image_url="https://cdn.example/ref.png", aspect_ratio="square")
    assert result["success"] is True
    assert result["modality"] == "image"
    assert captured["url"].endswith("/images/edits")
    assert captured["payload"]["images"] == [{"image_url": "https://cdn.example/ref.png"}]
    assert captured["payload"]["aspect_ratio"] == "1:1"


def test_base_resp_error(provider):
    def fake_post(url, *, headers, payload, timeout, label):
        return {"base_resp": {"status_code": 2013, "status_msg": "model not enabled"}}, None

    with patch("plugins.image_gen.minimax.post_json", side_effect=fake_post):
        result = provider.generate("a cat")
    assert result["success"] is False
    assert result["error_type"] == "api_error"
    assert "model not enabled" in result["error"]


def test_missing_key(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    provider = minimax_plugin.MiniMaxImageGenProvider()
    assert provider.is_available() is False
    result = provider.generate("a cat")
    assert result["success"] is False
    assert result["error_type"] == "auth_required"
