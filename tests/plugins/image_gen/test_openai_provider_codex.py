from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import plugins.image_gen.openai as openai_plugin


_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    import base64

    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


class _FakeStream:
    def __init__(self, events, final_response):
        self._events = list(events)
        self._final = final_response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_response(self):
        return self._final


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return openai_plugin.OpenAIImageGenProvider()


class TestCodexAvailability:
    def test_codex_token_makes_provider_available_without_openai_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(openai_plugin, "_read_codex_access_token", lambda: "codex-token")
        assert openai_plugin.OpenAIImageGenProvider().is_available() is True


class TestCodexGenerate:
    def test_generate_uses_codex_stream_path_without_openai_api_key(self, provider, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(openai_plugin, "_read_codex_access_token", lambda: "codex-token")

        output_item = SimpleNamespace(
            type="image_generation_call",
            status="generating",
            id="ig_test",
            result=_b64_png(),
        )
        done_event = SimpleNamespace(type="response.output_item.done", item=output_item)
        final_response = SimpleNamespace(output=[], status="completed", output_text="")

        fake_client = SimpleNamespace(
            responses=SimpleNamespace(
                stream=lambda **kwargs: _FakeStream([done_event], final_response)
            )
        )
        monkeypatch.setattr(openai_plugin, "_build_codex_client", lambda: fake_client)

        result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "gpt-image-2-medium"
        assert result["provider"] == "openai"
        assert result["quality"] == "medium"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"

    def test_codex_stream_request_shape(self, provider, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(openai_plugin, "_read_codex_access_token", lambda: "codex-token")

        captured = {}

        def _stream(**kwargs):
            captured.update(kwargs)
            output_item = SimpleNamespace(
                type="image_generation_call",
                status="generating",
                id="ig_test",
                result=_b64_png(),
            )
            done_event = SimpleNamespace(type="response.output_item.done", item=output_item)
            final_response = SimpleNamespace(output=[], status="completed", output_text="")
            return _FakeStream([done_event], final_response)

        fake_client = SimpleNamespace(responses=SimpleNamespace(stream=_stream))
        monkeypatch.setattr(openai_plugin, "_build_codex_client", lambda: fake_client)

        result = provider.generate("a cat", aspect_ratio="portrait")

        assert result["success"] is True
        assert captured["model"] == "gpt-5.4"
        assert captured["store"] is False
        assert captured["input"][0]["type"] == "message"
        assert captured["input"][0]["role"] == "user"
        assert captured["input"][0]["content"][0]["type"] == "input_text"
        assert captured["tool_choice"]["type"] == "allowed_tools"
        assert captured["tool_choice"]["mode"] == "required"
        assert captured["tool_choice"]["tools"] == [{"type": "image_generation"}]
        tool = captured["tools"][0]
        assert tool["type"] == "image_generation"
        assert tool["model"] == "gpt-image-2"
        assert tool["quality"] == "medium"
        assert tool["size"] == "1024x1536"
        assert tool["output_format"] == "png"
        assert tool["background"] == "opaque"
        assert tool["partial_images"] == 1
