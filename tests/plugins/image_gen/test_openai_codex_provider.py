"""Tests for the bundled ``openai-codex`` image_gen plugin."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

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


# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_unavailable_without_codex_token(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: None)
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is False

    def test_available_with_codex_token(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            codex_plugin, "_read_codex_access_token", lambda: "codex-token"
        )
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

    def test_generate_uses_codex_images_path(self, provider, monkeypatch, tmp_path):
        monkeypatch.setattr(
            codex_plugin, "_read_codex_access_token", lambda: "codex-token"
        )
        monkeypatch.setattr(
            codex_plugin, "_collect_image_b64", lambda *a, **kw: _b64_png()
        )

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

    def test_codex_generation_request_shape(self, provider, monkeypatch):
        monkeypatch.setattr(
            codex_plugin, "_read_codex_access_token", lambda: "codex-token"
        )

        captured = {}

        def _collect(token, *, prompt, size, quality, input_images=None):
            captured.update(
                codex_plugin._build_images_payload(
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    input_images=input_images,
                )
            )
            return _b64_png()

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _collect)

        result = provider.generate("a cat", aspect_ratio="portrait")
        assert result["success"] is True

        assert captured == {
            "prompt": "a cat",
            "background": "opaque",
            "model": "gpt-image-2",
            "n": 1,
            "quality": "medium",
            "size": "1024x1536",
        }

    def test_capabilities_advertise_image_inputs(self, provider):
        caps = provider.capabilities()
        assert caps["modalities"] == ["text", "image"]
        # A primary image consumes one of Codex's five total edit-image slots.
        assert caps["max_reference_images"] == 4

    def test_rejects_non_image_local_source(self, provider, monkeypatch, tmp_path):
        monkeypatch.setattr(
            codex_plugin, "_read_codex_access_token", lambda: "codex-token"
        )
        text_path = tmp_path / "not-image.txt"
        text_path.write_text("hello")

        result = provider.generate("edit this", image_url=str(text_path))

        assert result["success"] is False
        assert result["error_type"] == "invalid_image_input"
        assert "not a supported image" in result["error"]

    def test_empty_response_returns_error(self, provider, monkeypatch):
        monkeypatch.setattr(
            codex_plugin, "_read_codex_access_token", lambda: "codex-token"
        )
        monkeypatch.setattr(codex_plugin, "_collect_image_b64", lambda *a, **kw: None)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "empty_response"
        assert "no image data" in result["error"]

    def test_images_exception_returns_api_error(self, provider, monkeypatch):
        monkeypatch.setattr(
            codex_plugin, "_read_codex_access_token", lambda: "codex-token"
        )

        def _boom(*args, **kwargs):
            raise RuntimeError("cloudflare 403")

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _boom)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "cloudflare 403" in result["error"]

    def test_images_http_error_surfaces_verbatim(self, provider, monkeypatch):
        import httpx

        monkeypatch.setattr(
            codex_plugin, "_read_codex_access_token", lambda: "codex-token"
        )

        body = json.dumps({
            "error": {
                "message": "Image generation quota exceeded.",
                "type": "invalid_request_error",
                "param": None,
            }
        })

        def _handler(request):
            return httpx.Response(400, text=body, request=request)

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda *args, **kwargs: real_client(
                transport=httpx.MockTransport(_handler),
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout"),
            ),
        )

        result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "HTTP 400" in result["error"]
        assert "quota exceeded" in result["error"]


class TestRequestShape:
    def test_generation_payload_uses_dedicated_images_contract(self):
        payload = codex_plugin._build_images_payload(
            prompt="a red circle",
            size="1024x1024",
            quality="low",
        )
        assert payload["model"] == "gpt-image-2"
        assert payload["prompt"] == "a red circle"
        assert payload["quality"] == "low"
        assert payload["size"] == "1024x1024"
        assert "tools" not in payload

    def test_edit_payload_uses_images_array(self):
        payload = codex_plugin._build_images_payload(
            prompt="make it blue",
            size="1024x1024",
            quality="medium",
            input_images=[
                {"type": "input_image", "image_url": "data:image/png;base64,abc"}
            ],
        )
        assert payload["images"] == [{"image_url": "data:image/png;base64,abc"}]

    def test_edit_payload_preserves_validated_image_order(self):
        payload = codex_plugin._build_images_payload(
            prompt="combine these",
            size="1024x1024",
            quality="medium",
            input_images=[
                {"type": "input_image", "image_url": "https://example.test/one.png"},
                {"type": "input_image", "image_url": "https://example.test/two.png"},
            ],
        )

        assert payload["images"] == [
            {"image_url": "https://example.test/one.png"},
            {"image_url": "https://example.test/two.png"},
        ]

    @pytest.mark.parametrize(
        "malformed_part",
        [
            pytest.param(None, id="null-part"),
            pytest.param("private-image-marker", id="string-part"),
            pytest.param({}, id="missing-fields"),
            pytest.param(
                {"type": "input_image", "image_url": {"data": "private-image-marker"}},
                id="non-string-url",
            ),
            pytest.param(
                {
                    "type": "unexpected-private-image-marker",
                    "image_url": "https://example.test/image.png",
                },
                id="wrong-part-type",
            ),
            pytest.param(
                {"type": "input_image", "image_url": "   "},
                id="blank-url",
            ),
        ],
    )
    def test_edit_payload_rejects_malformed_internal_image_parts(self, malformed_part):
        with pytest.raises(ValueError) as excinfo:
            codex_plugin._build_images_payload(
                prompt="edit",
                size="1024x1024",
                quality="medium",
                input_images=[malformed_part],
            )

        message = str(excinfo.value)
        assert "index 0" in message
        assert "private-image-marker" not in message
        assert len(message) < 200

    def test_collect_routes_generation_to_dedicated_endpoint(self, monkeypatch):
        import httpx

        captured = {}

        def _handler(request):
            captured["path"] = request.url.path
            captured["accept"] = request.headers["Accept"]
            captured["authorization"] = request.headers["Authorization"]
            captured["content_type"] = request.headers["Content-Type"]
            captured["originator"] = request.headers["originator"]
            captured["turn_id"] = request.headers["x-codex-image-turn-id"]
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200, json={"data": [{"b64_json": _b64_png()}]}, request=request
            )

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda *args, **kwargs: real_client(
                transport=httpx.MockTransport(_handler),
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout"),
            ),
        )
        assert (
            codex_plugin._collect_image_b64(
                "codex-token", prompt="a cat", size="1024x1024", quality="low"
            )
            == _b64_png()
        )
        assert captured["path"].endswith("/codex/images/generations")
        assert captured["accept"] == "application/json"
        assert captured["authorization"] == "Bearer codex-token"
        assert captured["content_type"] == "application/json"
        assert captured["originator"] == "codex_cli_rs"
        assert captured["turn_id"]
        assert captured["payload"]["prompt"] == "a cat"

    def test_collect_routes_inputs_to_dedicated_edits_endpoint(self, monkeypatch):
        import httpx

        captured = {}

        def _handler(request):
            captured["path"] = request.url.path
            captured["authorization"] = request.headers["Authorization"]
            captured["content_type"] = request.headers["Content-Type"]
            captured["originator"] = request.headers["originator"]
            captured["turn_id"] = request.headers["x-codex-image-turn-id"]
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200, json={"data": [{"b64_json": _b64_png()}]}, request=request
            )

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda *args, **kwargs: real_client(
                transport=httpx.MockTransport(_handler),
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout"),
            ),
        )
        image = {"type": "input_image", "image_url": "data:image/png;base64,abc"}
        assert (
            codex_plugin._collect_image_b64(
                "codex-token",
                prompt="edit",
                size="1024x1024",
                quality="low",
                input_images=[image],
            )
            == _b64_png()
        )
        assert captured["path"].endswith("/codex/images/edits")
        assert captured["authorization"] == "Bearer codex-token"
        assert captured["content_type"] == "application/json"
        assert captured["originator"] == "codex_cli_rs"
        assert captured["turn_id"]
        assert captured["payload"]["images"] == [{"image_url": image["image_url"]}]

    def test_http_error_body_is_truncated_but_preserved(self, monkeypatch):
        """A large error body is capped at 500 chars and still surfaced."""
        import httpx

        body = json.dumps({
            "metadata": "x" * 600,
            "error": {"message": "Dedicated image request was rejected."},
        })

        def _handler(request):
            return httpx.Response(400, text=body, request=request)

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda *args, **kwargs: real_client(
                transport=httpx.MockTransport(_handler),
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout"),
            ),
        )

        with pytest.raises(RuntimeError, match="HTTP 400") as excinfo:
            codex_plugin._collect_image_b64(
                "codex-token",
                prompt="a cat",
                size="1024x1024",
                quality="low",
            )

        message = str(excinfo.value)
        # Body is capped, but the actionable wire message still reaches the user.
        assert "Dedicated image request was rejected" in message
        assert len(message) < len(body)

    def test_invalid_json_response_is_diagnostic(self, monkeypatch):
        import httpx

        credential = "credential-marker-must-not-leak"
        image_data = _b64_png() * 20
        body = (
            "upstream returned malformed content; "
            f"Authorization: Bearer {credential}; "
            f"image=data:image/png;base64,{image_data}; " + "padding=" + ("x" * 2_000)
        )

        def _handler(request):
            return httpx.Response(200, text=body, request=request)

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda *args, **kwargs: real_client(
                transport=httpx.MockTransport(_handler),
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout"),
            ),
        )
        with pytest.raises(RuntimeError, match="invalid JSON") as excinfo:
            codex_plugin._collect_image_b64(
                "codex-token", prompt="a cat", size="1024x1024", quality="low"
            )

        message = str(excinfo.value)
        assert "upstream returned malformed content" in message
        assert credential not in message
        assert image_data[:40] not in message
        assert len(message) < 700

    @pytest.mark.parametrize(
        "response_payload",
        [
            pytest.param([], id="root-list"),
            pytest.param({"status": "ok"}, id="missing-data"),
            pytest.param({"data": {}}, id="non-list-data"),
            pytest.param({"data": [None]}, id="non-object-item"),
            pytest.param({"data": [{"b64_json": 123}]}, id="non-string-image"),
        ],
    )
    def test_success_with_unexpected_json_shape_is_diagnostic(
        self, monkeypatch, response_payload
    ):
        import httpx

        def _handler(request):
            return httpx.Response(200, json=response_payload, request=request)

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda *args, **kwargs: real_client(
                transport=httpx.MockTransport(_handler),
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout"),
            ),
        )

        with pytest.raises(RuntimeError, match="unexpected response") as excinfo:
            codex_plugin._collect_image_b64(
                "codex-token", prompt="a cat", size="1024x1024", quality="low"
            )

        assert len(str(excinfo.value)) < 700

    def test_unexpected_json_diagnostic_is_bounded_and_sanitized(self, monkeypatch):
        import httpx

        credential = "credential-marker-must-not-leak"
        image_data = _b64_png() * 20
        body = json.dumps({
            "error": {
                "message": "backend schema mismatch",
                "access_token": credential,
            },
            "data": {"b64_json": image_data},
            "padding": "x" * 2_000,
        })

        def _handler(request):
            return httpx.Response(200, text=body, request=request)

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda *args, **kwargs: real_client(
                transport=httpx.MockTransport(_handler),
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout"),
            ),
        )

        with pytest.raises(RuntimeError, match="unexpected response") as excinfo:
            codex_plugin._collect_image_b64(
                "codex-token", prompt="a cat", size="1024x1024", quality="low"
            )

        message = str(excinfo.value)
        assert "backend schema mismatch" in message
        assert credential not in message
        assert image_data[:40] not in message
        assert len(message) < 700

    def test_empty_data_list_remains_an_empty_response(self, monkeypatch):
        import httpx

        def _handler(request):
            return httpx.Response(200, json={"data": []}, request=request)

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda *args, **kwargs: real_client(
                transport=httpx.MockTransport(_handler),
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout"),
            ),
        )

        assert (
            codex_plugin._collect_image_b64(
                "codex-token", prompt="a cat", size="1024x1024", quality="low"
            )
            is None
        )

    def test_five_total_input_images_are_accepted(self, tmp_path):
        paths = []
        for index in range(5):
            path = tmp_path / f"image-{index}.png"
            path.write_bytes(bytes.fromhex(_PNG_HEX))
            paths.append(str(path))

        images = codex_plugin._normalize_input_images(paths[0], paths[1:])

        assert len(images) == 5
        assert all(
            part["image_url"].startswith("data:image/png;base64,") for part in images
        )

    def test_more_than_five_total_input_images_are_rejected(self, tmp_path):
        paths = []
        for index in range(6):
            path = tmp_path / f"image-{index}.png"
            path.write_bytes(bytes.fromhex(_PNG_HEX))
            paths.append(str(path))

        with pytest.raises(ValueError, match="at most 5 total images"):
            codex_plugin._normalize_input_images(paths[0], paths[1:])

    def test_primary_plus_five_references_returns_invalid_input(
        self, provider, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            codex_plugin, "_read_codex_access_token", lambda: "codex-token"
        )
        paths = []
        for index in range(6):
            path = tmp_path / f"image-{index}.png"
            path.write_bytes(bytes.fromhex(_PNG_HEX))
            paths.append(str(path))

        result = provider.generate(
            "edit these",
            image_url=paths[0],
            reference_image_urls=paths[1:],
        )

        assert result["success"] is False
        assert result["error_type"] == "invalid_image_input"
        assert "at most 5 total images" in result["error"]


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
