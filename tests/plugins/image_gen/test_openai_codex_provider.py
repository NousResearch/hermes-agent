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
    monkeypatch.delenv("HERMES_CODEX_BASE_URL", raising=False)
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
    # Seed the auth.json token below the credential/base resolver so generate() exercises the real
    # (token, base_url) binding; no pool present.
    from agent import auxiliary_client
    monkeypatch.setattr(auxiliary_client, "_select_pool_entry", lambda provider, model=None: (False, None))
    monkeypatch.setattr(auxiliary_client, "_read_codex_singleton_token", lambda: "codex-token")
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
        monkeypatch.setattr(codex_plugin, "_read_codex_credential", lambda: (None, None))
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is False

    def test_available_with_codex_token(self, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_credential", lambda: ("tok", "https://chatgpt.com/backend-api/codex"))
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is True

    def test_openai_api_key_alone_is_not_enough(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(codex_plugin, "_read_codex_credential", lambda: (None, None))
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is False


# ── Generation ──────────────────────────────────────────────────────────────


class TestGenerate:
    @pytest.mark.parametrize("edit", [False, True])
    def test_custom_codex_route_keeps_image_request_on_configured_host(
        self, provider, codex_backend, monkeypatch, edit,
    ):
        monkeypatch.setenv("HERMES_CODEX_BASE_URL", "https://images.example.test/codex/")
        kwargs = {"image_url": "data:image/png;base64," + _b64_png()} if edit else {}
        result = provider.generate("a cat", **kwargs)
        assert result["success"] is True
        (request,) = codex_backend["requests"]
        path = "images/edits" if edit else "images/generations"
        assert str(request.url) == f"https://images.example.test/codex/{path}"
        assert request.headers["Authorization"] == "Bearer codex-token"
        assert request.headers["originator"] == "codex_cli_rs"

    @pytest.mark.parametrize(
        "model_id",
        [
            "gpt-image-1.5",
            "gpt-image-2",
            "gpt-image-2.5",
            "gpt-image-2.5-flare",
            "gpt-image-2.5-sunburst",
            "gpt-image-2.6",
            "gpt-image-2.6-flare",
            "gpt-image-2.6-sunburst",
        ],
    )
    def test_custom_codex_generation_sends_configured_model_id(
        self,
        provider,
        codex_backend,
        monkeypatch,
        tmp_path,
        model_id,
    ):
        monkeypatch.setenv("HERMES_CODEX_BASE_URL", "https://images.example.test/codex")
        (tmp_path / "config.yaml").write_text(
            f"image_gen:\n  openai-codex:\n    model: {model_id}\n", encoding="utf-8"
        )
        result = provider.generate("a cat")
        assert result["success"] is True
        assert result["model"] == model_id
        assert result["reported_model"] is None
        saved = Path(result["image"])
        assert saved.suffix == ".png"
        assert saved.read_bytes() == _png_bytes()
        (request,) = codex_backend["requests"]
        assert request.url.path == "/codex/images/generations"
        assert request.headers["Authorization"] == "Bearer codex-token"
        body = json.loads(request.content)
        assert body["model"] == model_id
        assert "quality" not in body

    def test_image_generate_uses_scoped_selection_and_text_only_schema(
        self,
        provider,
        codex_backend,
        monkeypatch,
        tmp_path,
    ):
        from tools import image_generation_tool as image_tool

        monkeypatch.setenv("HERMES_CODEX_BASE_URL", "https://images.example.test/codex")
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  provider: openai-codex\n  openai-codex:\n    model: gpt-image-2.6-flare\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(image_tool, "_get_plugin_provider", lambda name: provider)
        fields = image_tool._build_dynamic_image_schema()["parameters"]["properties"]
        assert "image_url" not in fields
        assert "reference_image_urls" not in fields
        result = json.loads(image_tool._handle_image_generate({"prompt": "a cat"}))
        assert result["success"] is True
        (request,) = codex_backend["requests"]
        assert request.url.path == "/codex/images/generations"
        assert json.loads(request.content)["model"] == "gpt-image-2.6-flare"

    @pytest.mark.parametrize("edit", [False, True])
    def test_stock_codex_accepts_literal_gpt_image_2(
        self, provider, codex_backend, tmp_path, edit
    ):
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  openai-codex:\n    model: gpt-image-2\n",
            encoding="utf-8",
        )
        kwargs = {"image_url": "data:image/png;base64," + _b64_png()} if edit else {}
        assert provider.generate("a cat", **kwargs)["model"] == "gpt-image-2"
        (request,) = codex_backend["requests"]
        endpoint = "edits" if edit else "generations"
        assert request.url.path.endswith(f"/images/{endpoint}")
        assert json.loads(request.content)["model"] == "gpt-image-2"

    def test_other_providers_shared_models_do_not_override_a_stock_codex_tier(
        self, provider, codex_backend, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-2.5-flare")
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  model: gpt-image-2.6-sunburst\n  openai-codex:\n    model: gpt-image-2-high\n",
            encoding="utf-8",
        )
        result = provider.generate("a cat")
        assert result["success"] is True
        assert result["model"] == "gpt-image-2-high"
        (request,) = codex_backend["requests"]
        body = json.loads(request.content)
        assert body["model"] == "gpt-image-2"
        assert body["quality"] == "high"

    def test_stock_codex_rejects_unverified_model_without_request(
        self,
        provider,
        codex_backend,
        tmp_path,
    ):
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  openai-codex:\n    model: gpt-image-2.6-sunburst\n",
            encoding="utf-8",
        )
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["model"] == "gpt-image-2.6-sunburst"
        assert result["error_type"] == "unsupported_model"
        assert codex_backend["requests"] == []

    def test_custom_codex_rejects_unverified_edit_before_loading_sources(
        self,
        provider,
        codex_backend,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setenv("HERMES_CODEX_BASE_URL", "https://images.example.test/codex")
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  openai-codex:\n    model: gpt-image-2.5-flare\n",
            encoding="utf-8",
        )
        assert provider.capabilities()["modalities"] == ["text"]
        result = provider.generate("edit", reference_image_urls=["/missing/image.png"])
        assert result["success"] is False
        assert result["error_type"] == "modality_unsupported"
        assert codex_backend["requests"] == []

    def test_custom_codex_model_refusal_preserves_api_message_and_does_not_retry(
        self,
        provider,
        codex_backend,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setenv("HERMES_CODEX_BASE_URL", "https://images.example.test/codex")
        (tmp_path / "config.yaml").write_text(
            "image_gen:\n  openai-codex:\n    model: gpt-image-2.6\n",
            encoding="utf-8",
        )
        codex_backend["respond"] = lambda request: httpx.Response(
            400,
            json={"error": {"message": "Model is unavailable on this endpoint."}},
            request=request,
        )
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert result["model"] == "gpt-image-2.6"
        assert "HTTP 400" in result["error"]
        assert "Model is unavailable on this endpoint." in result["error"]
        (request,) = codex_backend["requests"]
        assert json.loads(request.content)["model"] == "gpt-image-2.6"

    @pytest.mark.parametrize(
        ("image_format", "extension"),
        [("JPEG", ".jpg"), ("WEBP", ".webp"), ("GIF", ".gif")],
    )
    def test_output_extension_comes_from_bytes(
        self,
        provider,
        codex_backend,
        image_format,
        extension,
    ):
        from io import BytesIO

        from PIL import Image

        buffer = BytesIO()
        Image.new("RGB", (1, 1)).save(buffer, format=image_format)
        raw = buffer.getvalue()
        codex_backend["respond"] = lambda request: httpx.Response(
            200,
            json={
                "data": [{"b64_json": base64.b64encode(raw).decode()}],
                "output_format": "png",
                "model": "backend-reported-id",
            },
        )
        result = provider.generate("a cat")
        assert result["success"] is True
        assert result["reported_model"] == "backend-reported-id"
        saved = Path(result["image"])
        assert saved.suffix == extension
        assert saved.read_bytes() == raw

    @pytest.mark.parametrize(
        "b64", ["not-base64!", base64.b64encode(b"not an image").decode()]
    )
    def test_invalid_image_response_does_not_write_a_file(
        self, provider, codex_backend, tmp_path, b64
    ):
        codex_backend["respond"] = lambda request: httpx.Response(
            200, json={"data": [{"b64_json": b64}], "output_format": "png"}
        )
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "invalid_response"
        assert not list((tmp_path / "cache" / "generated" / "images").glob("*"))

    @pytest.mark.parametrize("override", ["", "   "])
    def test_blank_override_preserves_official_route(
        self, provider, codex_backend, monkeypatch, override,
    ):
        monkeypatch.setenv("HERMES_CODEX_BASE_URL", override)
        assert provider.generate("a cat")["success"] is True
        (request,) = codex_backend["requests"]
        assert str(request.url) == "https://chatgpt.com/backend-api/codex/images/generations"
        assert request.headers["originator"] == "hermes-agent"

    def test_profile_scope_routes_each_request_without_borrowing_process_override(
        self, provider, codex_backend, monkeypatch,
    ):
        from agent import secret_scope
        from agent.secret_scope import reset_secret_scope, set_secret_scope

        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
        monkeypatch.setenv("HERMES_CODEX_BASE_URL", "https://other-profile.example.test/codex")
        for scope, expected in [
            ({"HERMES_CODEX_BASE_URL": "https://profile.example.test/images-api/"},
             "https://profile.example.test/images-api/images/generations"),
            ({}, "https://chatgpt.com/backend-api/codex/images/generations"),
        ]:
            token = set_secret_scope(scope)
            try:
                assert provider.generate("a cat")["success"] is True
            finally:
                reset_secret_scope(token)
            assert str(codex_backend["requests"][-1].url) == expected

    @pytest.mark.parametrize("failure", ["http", "empty", "exception"])
    def test_custom_route_failure_never_retries_on_official_host(
        self, provider, codex_backend, monkeypatch, failure,
    ):
        monkeypatch.setenv("HERMES_CODEX_BASE_URL", "https://images.example.test/codex")

        def respond(request):
            if failure == "exception":
                raise httpx.ConnectError("fixture connection failed", request=request)
            if failure == "http":
                return httpx.Response(403, json={"error": {"message": "denied"}})
            return httpx.Response(200, json={"data": []})

        codex_backend["respond"] = respond
        assert provider.generate("a cat")["success"] is False
        (request,) = codex_backend["requests"]
        assert request.url.host == "images.example.test"

    def test_returns_auth_error_without_codex_token(self, provider, monkeypatch):
        from agent import auxiliary_client
        monkeypatch.setattr(auxiliary_client, "_select_pool_entry", lambda provider, model=None: (False, None))
        monkeypatch.setattr(auxiliary_client, "_read_codex_singleton_token", lambda: None)
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
        assert saved.exists() and saved.parent == tmp_path / "cache" / "generated" / "images"
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

    def test_custom_codex_base_receives_the_image_request(self, provider, codex_backend, tmp_path, monkeypatch):
        """With ``HERMES_CODEX_BASE_URL`` set, image requests go to the gateway's base instead of
        the hard-coded chatgpt.com host (#121486) — the text client honours the same override."""
        monkeypatch.setenv("HERMES_CODEX_BASE_URL", "https://codex-gw.example/backend-api/codex")

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
