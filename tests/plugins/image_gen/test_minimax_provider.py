"""Tests for the bundled MiniMax H3-Image image_gen plugin."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import plugins.image_gen.minimax as minimax_plugin
from plugins.image_gen._common import HttpFailure


_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _fake_result(*, url="https://cdn.example/img.png", b64=None):
    item = {}
    if url:
        item["url"] = url
    if b64:
        item["b64_json"] = b64
    return {"data": [item]}


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MINIMAX_IMAGE_AREA", raising=False)
    monkeypatch.delenv("MINIMAX_IMAGE_MODEL", raising=False)
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    return minimax_plugin.MiniMaxImageGenProvider()


class TestMetadata:
    def test_picker_matches_resolvable_catalog(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        assert set(ids) == set(provider.models)
        assert provider.default_model() in ids
        assert provider.default_model() == "MiniMax-H3-Image"

    def test_edit_capabilities(self, provider):
        caps = provider.capabilities()
        assert "image" in caps["modalities"]
        assert caps["max_reference_images"] == 16


class TestAvailability:
    def test_no_api_key_unavailable(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        assert minimax_plugin.MiniMaxImageGenProvider().is_available() is False

    def test_api_key_set_available(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test")
        assert minimax_plugin.MiniMaxImageGenProvider().is_available() is True


class TestEndpointConfig:
    def test_config_base_url_and_key_env_reach_request(self, monkeypatch, tmp_path):
        import hermes_yaml as yaml

        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "cn-token")
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"image_gen": {"minimax": {
            "base_url": "https://api.minimaxi.com/v1/",
            "key_env": "MINIMAX_CN_API_KEY",
            "area": "1K",
        }}}))
        provider = minimax_plugin.MiniMaxImageGenProvider()
        assert provider.is_available() is True
        captured = {}

        def fake_post(url, *, headers, payload, timeout, label):
            captured.update(url=url, headers=headers, payload=payload, timeout=timeout, label=label)
            return _fake_result(), None

        with patch("plugins.image_gen.minimax.post_json", side_effect=fake_post), patch(
            "plugins.image_gen._common.save_url_image",
            return_value=Path("/tmp/minimax_saved.png"),
        ):
            result = provider.generate("a cat", aspect_ratio="square")
        assert result["success"] is True
        assert captured["url"] == "https://api.minimaxi.com/v1/images/generations"
        assert captured["headers"]["Authorization"] == "Bearer cn-token"
        assert captured["payload"]["model"] == "MiniMax-H3-Image"
        assert captured["payload"]["aspect_ratio"] == "1:1"
        assert captured["payload"]["area"] == "1K"
        assert "size" not in captured["payload"]
        assert captured["timeout"][1] > 600


class TestGenerate:
    def test_empty_prompt(self, provider):
        result = provider.generate("  ")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_missing_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        result = minimax_plugin.MiniMaxImageGenProvider().generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    @pytest.mark.parametrize("aspect,expected", [
        ("landscape", "16:9"),
        ("square", "1:1"),
        ("portrait", "9:16"),
    ])
    def test_aspect_and_default_4k_area(self, provider, aspect, expected):
        captured = {}

        def fake_post(url, *, headers, payload, timeout, label):
            captured.update(url=url, payload=payload, timeout=timeout)
            return _fake_result(), None

        with patch("plugins.image_gen.minimax.post_json", side_effect=fake_post), patch(
            "plugins.image_gen._common.save_url_image",
            return_value=Path("/tmp/minimax_saved.png"),
        ):
            result = provider.generate("a cat", aspect_ratio=aspect)
        assert result["success"] is True
        assert result["area"] == "4K"
        assert captured["payload"]["aspect_ratio"] == expected
        assert captured["payload"]["area"] == "4K"
        assert "size" not in captured["payload"]
        assert captured["url"].endswith("/images/generations")
        assert captured["timeout"] == (30, 610)

    def test_edit_sends_data_uri_and_uses_edits_endpoint(self, provider, tmp_path):
        source = tmp_path / "source.png"
        source.write_bytes(bytes.fromhex(_PNG_HEX))
        captured = {}

        def fake_post(url, *, headers, payload, timeout, label):
            captured.update(url=url, payload=payload)
            return _fake_result(), None

        with patch("plugins.image_gen.minimax.post_json", side_effect=fake_post), patch(
            "plugins.image_gen._common.save_url_image",
            return_value=Path("/tmp/minimax_saved.png"),
        ):
            result = provider.generate("make it blue", image_url=str(source), aspect_ratio="square")
        assert result["success"] is True
        assert result["modality"] == "image"
        assert captured["url"].endswith("/images/edits")
        images = captured["payload"]["images"]
        assert len(images) == 1
        assert images[0]["image_url"].startswith("data:image/png;base64,")

    def test_edit_passes_through_https_url(self, provider):
        captured = {}

        def fake_post(url, *, headers, payload, timeout, label):
            captured.update(payload=payload)
            return _fake_result(), None

        with patch("plugins.image_gen.minimax.post_json", side_effect=fake_post), patch(
            "plugins.image_gen._common.save_url_image",
            return_value=Path("/tmp/minimax_saved.png"),
        ):
            provider.generate(
                "make it rainy",
                image_url="https://cdn.example/ref.png",
                aspect_ratio="square",
            )
        assert captured["payload"]["images"] == [{"image_url": "https://cdn.example/ref.png"}]

    def test_base_resp_error(self, provider):
        def fake_post(url, *, headers, payload, timeout, label):
            return {"base_resp": {"status_code": 2013, "status_msg": "model not enabled"}}, None

        with patch("plugins.image_gen.minimax.post_json", side_effect=fake_post):
            result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "model not enabled" in result["error"]

    def test_http_error(self, provider):
        def fake_post(url, *, headers, payload, timeout, label):
            return None, HttpFailure("http", "MiniMax image generation failed (429): rate limited", "api_error")

        with patch("plugins.image_gen.minimax.post_json", side_effect=fake_post):
            result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "api_error"
