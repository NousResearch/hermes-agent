"""Tests for the bundled 9router image_gen plugin."""

from __future__ import annotations

import base64
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

plugin = importlib.import_module("plugins.image_gen.9router")

_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


@pytest.fixture(autouse=True)
def _tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("ROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ROUTER_BASE_URL", "http://router.local/v1")
    monkeypatch.setenv("ROUTER_IMAGE_MODEL", "gpt-image-2-low")
    monkeypatch.setenv("ROUTER_RESPONSES_MODEL", "cx/test")
    return plugin.NineRouterImageGenProvider()


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "9router"
        assert provider.default_model() == "gpt-image-2-high"

    def test_models(self, provider):
        assert [m["id"] for m in provider.list_models()] == [
            "gpt-image-2-low",
            "gpt-image-2-medium",
            "gpt-image-2-high",
        ]


class TestAvailability:
    def test_requires_key(self, monkeypatch):
        monkeypatch.delenv("ROUTER_API_KEY", raising=False)
        monkeypatch.delenv("NINE_ROUTER_API_KEY", raising=False)
        monkeypatch.delenv("NINEROUTER_API_KEY", raising=False)
        monkeypatch.setattr(plugin, "_read_key_file", lambda path: None)
        assert plugin.NineRouterImageGenProvider().is_available() is False

    def test_available_with_env_key(self, monkeypatch):
        monkeypatch.setenv("ROUTER_API_KEY", "test-key")
        assert plugin.NineRouterImageGenProvider().is_available() is True

    def test_available_with_compact_env_key(self, monkeypatch):
        monkeypatch.delenv("ROUTER_API_KEY", raising=False)
        monkeypatch.delenv("NINE_ROUTER_API_KEY", raising=False)
        monkeypatch.setenv("NINEROUTER_API_KEY", "test-key")
        assert plugin.NineRouterImageGenProvider().is_available() is True

    def test_available_with_provider_api_key_env(self, monkeypatch):
        monkeypatch.delenv("ROUTER_API_KEY", raising=False)
        monkeypatch.delenv("NINE_ROUTER_API_KEY", raising=False)
        monkeypatch.delenv("NINEROUTER_API_KEY", raising=False)
        monkeypatch.setenv("CUSTOM_ROUTER_KEY", "test-key")
        monkeypatch.setattr(
            plugin,
            "_load_config",
            lambda: {
                "providers": {
                    "9router": {
                        "base_url": "http://router.local/v1",
                        "api_key_env": "CUSTOM_ROUTER_KEY",
                    }
                }
            },
        )
        assert plugin.NineRouterImageGenProvider().is_available() is True


class TestGenerate:
    def test_sends_reference_image_and_saves_result(self, provider, tmp_path, monkeypatch):
        ref = tmp_path / "ref.png"
        ref.write_bytes(bytes.fromhex(_PNG_HEX))
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"output": [{"type": "image_generation_call", "result": _b64_png()}]}).encode()

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["body"] = json.loads(req.data.decode())
            captured["timeout"] = timeout
            return FakeResponse()

        monkeypatch.setattr(plugin.urllib.request, "urlopen", fake_urlopen)
        result = provider.generate("make it shiny", aspect_ratio="square", reference_images=[str(ref)])

        assert result["success"] is True
        assert result["provider"] == "9router"
        assert result["model"] == "gpt-image-2-low"
        assert Path(result["image"]).exists()
        assert captured["url"] == "http://router.local/v1/responses"
        assert captured["body"]["model"] == "cx/test"
        content = captured["body"]["input"][0]["content"]
        assert content[0] == {"type": "input_text", "text": "make it shiny"}
        assert content[1]["type"] == "input_image"
        assert content[1]["image_url"].startswith("data:image/png;base64,")
        assert captured["body"]["tools"][0]["type"] == "image_generation"
        assert captured["body"]["tools"][0]["quality"] == "low"
