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
from types import SimpleNamespace

import httpx
import pytest

# The plugin directory uses a hyphen, which is not a valid Python identifier
# for the dotted-import form. Load it via importlib so tests don't need to
# touch sys.path or rename the directory.
codex_plugin = importlib.import_module("plugins.image_gen.openai-codex")


def _codex_jwt(subject: str) -> str:
    """JWT-shaped test stand-in (signature never verified client-side): the canonical host
    only answers for a JWT, so route fixtures that must be accepted use this shape (#121486)."""
    enc = lambda raw: base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    header = enc(b'{"alg":"RS256"}')
    payload = enc(json.dumps({"sub": subject}).encode())
    return f"{header}.{payload}.test-signature"


# dummy fixture credentials (not real secrets): an auth-store OAuth token is JWT-shaped;
# a gateway pool key is opaque.
_JWT_TOKEN = _codex_jwt("codex-image-account")
_GATEWAY_POOL_KEY = "dummy-gateway-pool-key"


@pytest.fixture(autouse=True)
def _no_ambient_pool(monkeypatch):
    """Keep the route resolver on the singleton branch unless a test stages a pool row
    explicitly (the tmp HERMES_HOME pool is empty anyway — this pins it)."""
    monkeypatch.setattr("agent.auxiliary_client._select_pool_entry", lambda provider: (False, None))
    monkeypatch.setattr("agent.auxiliary_client._codex_base_url_override", lambda: "")


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
    monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: _JWT_TOKEN)
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
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: _JWT_TOKEN)
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
        assert request.headers["Authorization"] == f"Bearer {_JWT_TOKEN}"
        assert request.headers["x-codex-image-turn-id"]
        body = json.loads(request.content)
        assert body == {
            "prompt": "a cat", "model": "gpt-image-2", "n": 1, "quality": "medium",
            "size": "1024x1536", "background": "opaque",
        }
        # The whole point of the native route: nothing about a chat model in the request.
        assert not any(key in body for key in ("tools", "input", "instructions"))

    def test_custom_codex_base_receives_the_image_request(self, provider, codex_backend, tmp_path, monkeypatch):
        """When the routing decision resolves a custom Codex base (the profile-scoped override),
        image requests go to that base instead of the hard-coded chatgpt.com host — the credential
        and its destination travel as one pair (#121486)."""
        monkeypatch.setattr(
            "agent.auxiliary_client._codex_base_url_override",
            lambda: "https://codex-gw.example/backend-api/codex")

        result = provider.generate("a cat")

        assert result["success"] is True
        (request,) = codex_backend["requests"]
        assert request.url.host == "codex-gw.example"
        assert request.url.path.endswith("/backend-api/codex/images/generations")

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


# ── Route authority (#121486) ───────────────────────────────────────────────


class TestImageRouteAuthority:
    """The credential and its destination are carried as one authority object: every test
    asserts that no Authorization-bearing request reaches a host other than the one the
    credential is bound to."""

    def test_pool_credential_travels_with_its_row_base(self, monkeypatch, codex_backend):
        """A pool-selected credential is sent only to its own row's base — never to chatgpt.com."""
        monkeypatch.setattr(
            "agent.auxiliary_client._select_pool_entry",
            lambda provider: (True, SimpleNamespace(provider="openai-codex")))
        monkeypatch.setattr("agent.auxiliary_client._pool_runtime_api_key", lambda entry: _GATEWAY_POOL_KEY)
        monkeypatch.setattr(
            "agent.auxiliary_client._pool_runtime_base_url",
            lambda entry, fallback="": "https://gw.example/backend-api/codex")

        token, base = codex_plugin._codex_image_route()
        assert token == _GATEWAY_POOL_KEY
        assert base == "https://gw.example/backend-api/codex"

        result = codex_plugin.OpenAICodexImageGenProvider().generate("a cat")
        assert result["success"] is True
        (request,) = codex_backend["requests"]
        assert request.url.host == "gw.example"
        assert request.headers["Authorization"] == f"Bearer {_GATEWAY_POOL_KEY}"

    def test_override_wins_for_a_pool_credential(self, monkeypatch):
        """The profile-scoped override outranks the pool row's own base, for every reader of
        the row — same rule as the runtime's ``_pool_entry_mode_and_url``."""
        monkeypatch.setattr(
            "agent.auxiliary_client._select_pool_entry",
            lambda provider: (True, SimpleNamespace(provider="openai-codex")))
        monkeypatch.setattr("agent.auxiliary_client._pool_runtime_api_key", lambda entry: _GATEWAY_POOL_KEY)
        monkeypatch.setattr(
            "agent.auxiliary_client._pool_runtime_base_url",
            lambda entry, fallback="": "https://gw.example/backend-api/codex")
        monkeypatch.setattr(
            "agent.auxiliary_client._codex_base_url_override", lambda: "https://override.example/api")

        token, base = codex_plugin._codex_image_route()
        assert token == _GATEWAY_POOL_KEY and base == "https://override.example/api"

    def test_opaque_pool_key_is_declined_for_the_canonical_host(self, monkeypatch, codex_backend):
        """An opaque pool key paired with the canonical chatgpt.com host is a gateway key that
        chatgpt.com cannot answer for — the route is declined and no request leaves the process."""
        monkeypatch.setattr(
            "agent.auxiliary_client._select_pool_entry",
            lambda provider: (True, SimpleNamespace(provider="openai-codex")))
        monkeypatch.setattr("agent.auxiliary_client._pool_runtime_api_key", lambda entry: _GATEWAY_POOL_KEY)
        monkeypatch.setattr(
            "agent.auxiliary_client._pool_runtime_base_url",
            lambda entry, fallback="": "https://chatgpt.com/backend-api/codex")

        token, base = codex_plugin._codex_image_route()
        assert token is None

        result = codex_plugin.OpenAICodexImageGenProvider().generate("a cat")
        assert result["success"] is False and result["error_type"] == "auth_required"
        assert codex_backend["requests"] == []

    def test_direct_chatgpt_positive_control(self, monkeypatch, codex_backend):
        """Positive control: a singleton OAuth JWT with no override is served by the canonical
        host, exactly as before the authority pairing."""
        token, base = codex_plugin._codex_image_route()
        assert token == _JWT_TOKEN
        assert base == "https://chatgpt.com/backend-api/codex"

        result = codex_plugin.OpenAICodexImageGenProvider().generate("a cat")
        assert result["success"] is True
        (request,) = codex_backend["requests"]
        assert request.url.host == "chatgpt.com"
        assert request.headers["Authorization"] == f"Bearer {_JWT_TOKEN}"


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
