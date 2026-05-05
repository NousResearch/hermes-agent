"""Tests for the bundled ``openai-codex`` image_gen plugin.

Mirrors ``test_openai_provider.py`` but targets the standalone
Codex/ChatGPT-OAuth-backed provider that uses the Responses
``image_generation`` tool path instead of the ``images.generate`` REST
endpoint.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

# The plugin directory uses a hyphen, which is not a valid Python identifier
# for the dotted-import form. Load it via importlib so tests don't need to
# touch sys.path or rename the directory.
codex_plugin = importlib.import_module("plugins.image_gen.openai-codex")


# 1×1 transparent PNG — valid bytes for save_b64_image()
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
    # Codex plugin is API-key-independent; clear it to make the test honest.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return codex_plugin.OpenAICodexImageGenProvider()


# ── Metadata ────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "openai-codex"

    def test_display_name(self, provider):
        assert provider.display_name == "OpenAI (Codex auth)"

    def test_default_model(self, provider):
        assert provider.default_model() == "gpt-image-2-medium"

    def test_list_models_three_tiers(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        assert ids == ["gpt-image-2-low", "gpt-image-2-medium", "gpt-image-2-high"]

    def test_setup_schema_has_no_required_env_vars(self, provider):
        schema = provider.get_setup_schema()
        assert schema["env_vars"] == []
        assert schema["badge"] == "free"

    def test_supports_edit(self, provider):
        assert provider.supports_edit() is True


# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_unavailable_without_codex_token(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: None)
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is False

    def test_available_with_codex_token(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is True

    def test_openai_api_key_alone_is_not_enough(self, monkeypatch):
        # Codex plugin is intentionally orthogonal to the API-key plugin —
        # the API key alone must NOT make it appear available.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: None)
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is False


# ── Generate ────────────────────────────────────────────────────────────────


class TestGenerate:
    def test_returns_auth_error_without_codex_token(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: None)
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_returns_invalid_argument_for_empty_prompt(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        result = provider.generate("   ")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_generate_uses_codex_stream_path(self, provider, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

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
        monkeypatch.setattr(codex_plugin, "_build_codex_client", lambda: fake_client)

        result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "gpt-image-2-medium"
        assert result["provider"] == "openai-codex"
        assert result["quality"] == "medium"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        # Filename prefix differs from the API-key plugin so cache audits can
        # tell the two backends apart.
        assert saved.name.startswith("openai_codex_")

    def test_codex_stream_request_shape(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

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
        monkeypatch.setattr(codex_plugin, "_build_codex_client", lambda: fake_client)

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

    @pytest.mark.parametrize("aspect,expected_size", [
        ("16:9", "1824x1024"),
        ("9:16", "1024x1824"),
        ("4:3", "1360x1024"),
        ("3:4", "1024x1360"),
    ])
    def test_codex_aspect_ratio_mapping(self, provider, monkeypatch, aspect, expected_size):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        captured = {}

        def _stream(**kwargs):
            captured.update(kwargs)
            output_item = SimpleNamespace(type="image_generation_call", result=_b64_png())
            done_event = SimpleNamespace(type="response.output_item.done", item=output_item)
            return _FakeStream([done_event], SimpleNamespace(output=[]))

        fake_client = SimpleNamespace(responses=SimpleNamespace(stream=_stream))
        monkeypatch.setattr(codex_plugin, "_build_codex_client", lambda: fake_client)

        result = provider.generate("a cat", aspect_ratio=aspect)
        assert result["success"] is True
        assert captured["tools"][0]["size"] == expected_size

    def test_codex_explicit_size_overrides_aspect_ratio(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        captured = {}

        def _stream(**kwargs):
            captured.update(kwargs)
            output_item = SimpleNamespace(type="image_generation_call", result=_b64_png())
            done_event = SimpleNamespace(type="response.output_item.done", item=output_item)
            return _FakeStream([done_event], SimpleNamespace(output=[]))

        fake_client = SimpleNamespace(responses=SimpleNamespace(stream=_stream))
        monkeypatch.setattr(codex_plugin, "_build_codex_client", lambda: fake_client)

        result = provider.generate("a cat", aspect_ratio="square", size="1024x1824")
        assert result["success"] is True
        assert result["size"] == "1024x1824"
        assert captured["tools"][0]["size"] == "1024x1824"

    def test_codex_invalid_explicit_size_returns_error(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        monkeypatch.setattr(codex_plugin, "_build_codex_client", lambda: object())

        result = provider.generate("a cat", size="1000x1000")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_partial_image_event_used_when_done_missing(self, provider, monkeypatch):
        """If the stream never emits output_item.done, fall back to the
        partial_image event so users at least get the latest preview frame."""
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

        partial_event = SimpleNamespace(
            type="response.image_generation_call.partial_image",
            partial_image_b64=_b64_png(),
        )
        final_response = SimpleNamespace(output=[], status="completed", output_text="")

        fake_client = SimpleNamespace(
            responses=SimpleNamespace(
                stream=lambda **kwargs: _FakeStream([partial_event], final_response)
            )
        )
        monkeypatch.setattr(codex_plugin, "_build_codex_client", lambda: fake_client)

        result = provider.generate("a cat")
        assert result["success"] is True
        assert Path(result["image"]).exists()

    def test_final_response_sweep_recovers_image(self, provider, monkeypatch):
        """If no image_generation_call event arrives mid-stream, the
        post-stream final-response sweep should still find the image."""
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

        final_item = SimpleNamespace(
            type="image_generation_call",
            status="completed",
            id="ig_final",
            result=_b64_png(),
        )
        final_response = SimpleNamespace(output=[final_item], status="completed", output_text="")

        fake_client = SimpleNamespace(
            responses=SimpleNamespace(
                stream=lambda **kwargs: _FakeStream([], final_response)
            )
        )
        monkeypatch.setattr(codex_plugin, "_build_codex_client", lambda: fake_client)

        result = provider.generate("a cat")
        assert result["success"] is True
        assert Path(result["image"]).exists()

    def test_empty_response_returns_error(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

        final_response = SimpleNamespace(output=[], status="completed", output_text="")
        fake_client = SimpleNamespace(
            responses=SimpleNamespace(
                stream=lambda **kwargs: _FakeStream([], final_response)
            )
        )
        monkeypatch.setattr(codex_plugin, "_build_codex_client", lambda: fake_client)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_client_init_failure_returns_auth_error(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        monkeypatch.setattr(codex_plugin, "_build_codex_client", lambda: None)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_stream_exception_returns_api_error(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

        def _boom(**kwargs):
            raise RuntimeError("cloudflare 403")

        fake_client = SimpleNamespace(responses=SimpleNamespace(stream=_boom))
        monkeypatch.setattr(codex_plugin, "_build_codex_client", lambda: fake_client)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "cloudflare 403" in result["error"]


# ── Edit ────────────────────────────────────────────────────────────────────


class TestEdit:
    def test_edit_uses_input_image_content(self, provider, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        source = tmp_path / "source.png"
        source.write_bytes(bytes.fromhex(_PNG_HEX))

        captured = {}

        def _stream(**kwargs):
            captured.update(kwargs)
            output_item = SimpleNamespace(
                type="image_generation_call",
                status="generating",
                id="ig_edit",
                result=_b64_png(),
            )
            done_event = SimpleNamespace(type="response.output_item.done", item=output_item)
            final_response = SimpleNamespace(output=[], status="completed", output_text="")
            return _FakeStream([done_event], final_response)

        fake_client = SimpleNamespace(responses=SimpleNamespace(stream=_stream))
        monkeypatch.setattr(codex_plugin, "_build_codex_client", lambda: fake_client)

        result = provider.edit("make the background blue", str(source), aspect_ratio="square")

        assert result["success"] is True
        assert result["provider"] == "openai-codex"
        assert result["source_image"] == str(source)
        assert Path(result["image"]).exists()
        assert Path(result["image"]).name.startswith("openai_codex_edit_")

        content = captured["input"][0]["content"]
        assert content[0] == {"type": "input_text", "text": "make the background blue"}
        assert content[1]["type"] == "input_image"
        assert content[1]["image_url"].startswith("data:image/png;base64,")

        tool = captured["tools"][0]
        assert tool["type"] == "image_generation"
        assert tool["model"] == "gpt-image-2"
        assert tool["action"] == "edit"
        assert tool["size"] == "1024x1024"
        assert tool["quality"] == "medium"

    def test_edit_accepts_http_image_url(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        captured = {}

        def _stream(**kwargs):
            captured.update(kwargs)
            output_item = SimpleNamespace(type="image_generation_call", result=_b64_png())
            done_event = SimpleNamespace(type="response.output_item.done", item=output_item)
            return _FakeStream([done_event], SimpleNamespace(output=[]))

        monkeypatch.setattr(
            codex_plugin,
            "_build_codex_client",
            lambda: SimpleNamespace(responses=SimpleNamespace(stream=_stream)),
        )

        result = provider.edit("add a hat", "https://example.com/cat.png")

        assert result["success"] is True
        assert captured["input"][0]["content"][1] == {
            "type": "input_image",
            "image_url": "https://example.com/cat.png",
        }

    def test_edit_accepts_valid_image_data_url(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        captured = {}

        def _stream(**kwargs):
            captured.update(kwargs)
            output_item = SimpleNamespace(type="image_generation_call", result=_b64_png())
            done_event = SimpleNamespace(type="response.output_item.done", item=output_item)
            return _FakeStream([done_event], SimpleNamespace(output=[]))

        monkeypatch.setattr(
            codex_plugin,
            "_build_codex_client",
            lambda: SimpleNamespace(responses=SimpleNamespace(stream=_stream)),
        )
        data_url = f"data:image/png;base64,{_b64_png()}"

        result = provider.edit("add a hat", data_url)

        assert result["success"] is True
        assert captured["input"][0]["content"][1] == {
            "type": "input_image",
            "image_url": data_url,
        }

    def test_edit_rejects_non_image_local_file(self, provider, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        source = tmp_path / "secrets.txt"
        source.write_text("not an image")
        monkeypatch.setattr(
            codex_plugin,
            "_build_codex_client",
            lambda: SimpleNamespace(responses=SimpleNamespace(stream=lambda **kwargs: None)),
        )

        result = provider.edit("make it blue", str(source))

        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        assert "Reference image must be" in result["error"]

    def test_edit_rejects_text_data_url(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        monkeypatch.setattr(
            codex_plugin,
            "_build_codex_client",
            lambda: SimpleNamespace(responses=SimpleNamespace(stream=lambda **kwargs: None)),
        )

        result = provider.edit("make it blue", "data:text/plain;base64,Zm9v")

        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        assert "Unsupported reference image MIME type" in result["error"]

    def test_edit_rejects_malformed_image_data_url(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        monkeypatch.setattr(
            codex_plugin,
            "_build_codex_client",
            lambda: SimpleNamespace(responses=SimpleNamespace(stream=lambda **kwargs: None)),
        )

        result = provider.edit("make it blue", "data:image/png;base64,not-base64")

        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        assert "invalid base64" in result["error"]

    def test_edit_rejects_image_data_url_with_non_image_bytes(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        monkeypatch.setattr(
            codex_plugin,
            "_build_codex_client",
            lambda: SimpleNamespace(responses=SimpleNamespace(stream=lambda **kwargs: None)),
        )

        result = provider.edit("make it blue", "data:image/png;base64,Zm9v")

        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        assert "must contain PNG" in result["error"]

    def test_edit_rejects_oversized_local_image(self, provider, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        source = tmp_path / "huge.png"
        source.write_bytes(bytes.fromhex(_PNG_HEX))
        monkeypatch.setattr(codex_plugin, "_MAX_REFERENCE_IMAGE_BYTES", 4)
        monkeypatch.setattr(
            codex_plugin,
            "_build_codex_client",
            lambda: SimpleNamespace(responses=SimpleNamespace(stream=lambda **kwargs: None)),
        )

        result = provider.edit("make it blue", str(source))

        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        assert "Reference image is too large" in result["error"]

    def test_edit_model_kwarg_overrides_env_selected_tier(self, provider, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-2-low")
        source = tmp_path / "source.png"
        source.write_bytes(bytes.fromhex(_PNG_HEX))
        captured = {}

        def _stream(**kwargs):
            captured.update(kwargs)
            output_item = SimpleNamespace(type="image_generation_call", result=_b64_png())
            done_event = SimpleNamespace(type="response.output_item.done", item=output_item)
            return _FakeStream([done_event], SimpleNamespace(output=[]))

        monkeypatch.setattr(
            codex_plugin,
            "_build_codex_client",
            lambda: SimpleNamespace(responses=SimpleNamespace(stream=_stream)),
        )

        result = provider.edit(
            "add a hat",
            str(source),
            model="gpt-image-2-high",
        )

        assert result["success"] is True
        assert result["model"] == "gpt-image-2-high"
        assert result["quality"] == "high"
        assert captured["tools"][0]["quality"] == "high"

    def test_edit_quality_tier_kwarg_overrides_env_selected_tier(self, provider, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-2-high")
        source = tmp_path / "source.png"
        source.write_bytes(bytes.fromhex(_PNG_HEX))
        captured = {}

        def _stream(**kwargs):
            captured.update(kwargs)
            output_item = SimpleNamespace(type="image_generation_call", result=_b64_png())
            done_event = SimpleNamespace(type="response.output_item.done", item=output_item)
            return _FakeStream([done_event], SimpleNamespace(output=[]))

        monkeypatch.setattr(
            codex_plugin,
            "_build_codex_client",
            lambda: SimpleNamespace(responses=SimpleNamespace(stream=_stream)),
        )

        result = provider.edit(
            "add a hat",
            str(source),
            quality_tier="low",
        )

        assert result["success"] is True
        assert result["model"] == "gpt-image-2-low"
        assert result["quality"] == "low"
        assert captured["tools"][0]["quality"] == "low"

    def test_edit_missing_reference_image_returns_invalid_argument(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        monkeypatch.setattr(
            codex_plugin,
            "_build_codex_client",
            lambda: SimpleNamespace(responses=SimpleNamespace(stream=lambda **kwargs: None)),
        )

        result = provider.edit("add a hat", "/definitely/missing.png")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        assert "Reference image not found" in result["error"]


# ── Plugin entry point ──────────────────────────────────────────────────────


class TestRegistration:
    def test_register_calls_register_image_gen_provider(self):
        registered = []

        class _Ctx:
            def register_image_gen_provider(self, prov):
                registered.append(prov)

        codex_plugin.register(_Ctx())
        assert len(registered) == 1
        assert registered[0].name == "openai-codex"
