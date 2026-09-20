"""Tests for the bundled ``openai-codex`` image_gen plugin.

Mirrors ``test_openai_provider.py`` but targets the ChatGPT-OAuth-backed provider that posts to
the Codex backend's native ``images/generations`` / ``images/edits`` endpoints (the route the
official Codex client uses) — no chat host model, no hosted-tool SSE stream (#105398, #107076).
"""

from __future__ import annotations

import base64
import importlib
import json
from pathlib import Path

import httpx
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


def _png_bytes() -> bytes:
    return bytes.fromhex(_PNG_HEX)


def _b64_png() -> str:
    return base64.b64encode(_png_bytes()).decode()


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    # Codex plugin is API-key-independent; clear it to make the test honest.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return codex_plugin.OpenAICodexImageGenProvider()


@pytest.fixture
def codex_backend(monkeypatch):
    """Route the plugin's ``httpx.Client`` at a fake Codex images backend; returns the request log
    and lets a test swap the response via ``state["respond"]``."""
    monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
    state = {"requests": [], "respond": None}

    def _default(request):
        return httpx.Response(200, json={
            "created": 1, "data": [{"b64_json": _b64_png(), "generation_id": "gen_1"}],
            "background": "opaque", "output_format": "png", "quality": "low", "size": "1254x1254",
        }, headers={"x-codex-imagegen-request-id": "req_abc"}, request=request)

    def _handler(request):
        state["requests"].append(request)
        return (state["respond"] or _default)(request)

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *args, **kwargs: real_client(
            transport=httpx.MockTransport(_handler), headers=kwargs.get("headers"),
            timeout=kwargs.get("timeout")),
    )
    return state


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
        """#102144: the keyless row must declare the shared Codex OAuth bootstrap hook (otherwise setup
        saves the backend without ever signing in) and its hint must name a command that exists."""
        schema = provider.get_setup_schema()
        assert schema["env_vars"] == []
        assert schema["post_setup"] == "openai_codex"
        assert "hermes auth add openai-codex" in schema["post_setup_hint"]
        assert "hermes auth codex`" not in schema["post_setup_hint"]


# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_unavailable_without_codex_token(self, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: None)
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is False

    def test_available_with_codex_token(self, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "tok")
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is True

    def test_openai_api_key_alone_is_not_enough(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: None)
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is False


# ── Generation ──────────────────────────────────────────────────────────────


class TestGenerate:
    def test_returns_auth_error_without_codex_token(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: None)
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_text_to_image_posts_generations_with_no_host_model(self, provider, codex_backend, tmp_path):
        result = provider.generate("a cat", aspect_ratio="portrait")

        assert result["success"] is True
        assert result["model"] == "gpt-image-2-medium"
        assert result["provider"] == "openai-codex"
        assert result["quality"] == "medium"
        assert result["pixel_size"] == "1x1"
        # Backend-reported values travel separately from what we asked for (#107233).
        assert result["reported_quality"] == "low"
        assert result["reported_size"] == "1254x1254"
        assert result["imagegen_request_id"] == "req_abc"
        saved = Path(result["image"])
        assert saved.exists() and saved.parent == tmp_path / "cache" / "images"
        assert saved.name.startswith("openai_codex_")

        (request,) = codex_backend["requests"]
        assert request.url.path.endswith("/backend-api/codex/images/generations")
        assert request.headers["Authorization"] == "Bearer codex-token"
        assert request.headers["x-codex-image-turn-id"]
        body = json.loads(request.content)
        assert body == {
            "prompt": "a cat", "model": "gpt-image-2", "n": 1, "quality": "medium",
            "size": "1024x1536", "background": "opaque",
        }
        # The whole point of the native route: nothing about a chat model in the request.
        assert not any(key in body for key in ("tools", "input", "instructions"))

    def test_source_images_post_edits_with_inline_data_urls(self, provider, codex_backend, tmp_path):
        local = tmp_path / "ref.png"
        local.write_bytes(_png_bytes())
        data_url = "data:image/png;base64," + _b64_png()

        result = provider.generate("edit these", image_url=str(local), reference_image_urls=[data_url])

        assert result["success"] is True
        assert result["modality"] == "image"
        assert result["input_image_count"] == 2
        (request,) = codex_backend["requests"]
        assert request.url.path.endswith("/backend-api/codex/images/edits")
        body = json.loads(request.content)
        assert [img["image_url"] for img in body["images"]] == [data_url, data_url]

    def test_remote_source_url_is_fetched_and_inlined(self, provider, codex_backend, monkeypatch):
        # The backend's own URL downloader 400s on ordinary public images; we fetch client-side.
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        # codex_backend monkeypatches httpx.Client; build the fetch client from
        # the unpatched class so the ref-image download gets the PNG responder.
        real_client = httpx._client.Client
        monkeypatch.setattr(
            "tools.url_safety.create_ssrf_safe_client",
            lambda **kw: real_client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=_png_bytes(), request=request)),
                **kw))

        result = provider.generate("edit", image_url="https://example.com/ref.png")

        assert result["success"] is True
        body = json.loads(codex_backend["requests"][0].content)
        assert body["images"] == [{"image_url": "data:image/png;base64," + _b64_png()}]

    def test_capabilities_advertise_image_inputs(self, provider):
        caps = provider.capabilities()
        assert caps["modalities"] == ["text", "image"]
        assert caps["max_reference_images"] == 16

    def test_rejects_non_image_local_source(self, provider, codex_backend, tmp_path):
        text_path = tmp_path / "not-image.txt"
        text_path.write_text("hello", encoding="utf-8")

        result = provider.generate("edit this", image_url=str(text_path))

        assert result["success"] is False
        assert result["error_type"] == "invalid_image_input"
        assert "not a supported image" in result["error"]
        assert codex_backend["requests"] == []

    def test_http_error_message_surfaces_verbatim_and_bounded(self, provider, codex_backend):
        body = json.dumps({
            "metadata": "x" * 600,
            "error": {"message": "Missing required parameter: 'prompt'.", "type": "invalid_request_error"},
        })
        codex_backend["respond"] = lambda request: httpx.Response(400, text=body, request=request)

        result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "HTTP 400" in result["error"]
        assert "Missing required parameter: 'prompt'." in result["error"]
        assert len(result["error"]) < len(body)

    def test_missing_image_data_is_empty_response(self, provider, codex_backend):
        codex_backend["respond"] = lambda request: httpx.Response(
            200, json={"created": 1, "data": []}, request=request)

        result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"


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
