"""Tests for the bundled ``openai-codex`` image_gen plugin.

Mirrors ``test_openai_provider.py`` but targets the standalone
Codex/ChatGPT-OAuth-backed provider that uses the Responses
``image_generation`` tool path instead of the ``images.generate`` REST
endpoint.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

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
        # GPT Image 2 three-tier catalog remains the leading entries; 2.5 variants follow.
        assert ids[:3] == ["gpt-image-2-low", "gpt-image-2-medium", "gpt-image-2-high"]

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


    def test_generate_uses_codex_stream_path(self, provider, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        monkeypatch.setattr(codex_plugin, "_collect_image_b64", lambda *a, **kw: {"b64": _b64_png(), "source": "final"})

        result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "gpt-image-2-medium"
        assert result["provider"] == "openai-codex"
        assert result["quality"] == "medium"
        assert result.get("image_source") == "final"
        assert result.get("pixel_size") == "1x1"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        # Filename prefix differs from the API-key plugin so cache audits can
        # tell the two backends apart.
        assert saved.name.startswith("openai_codex_")

    def test_codex_stream_request_shape(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

        captured = {}

        def _collect(token, **kwargs):
            payload_kwargs = {
                k: kwargs[k]
                for k in ("prompt", "size", "quality", "input_images", "api_model")
                if k in kwargs
            }
            captured.update(codex_plugin._build_responses_payload(**payload_kwargs))
            return {"b64": _b64_png(), "source": "final"}

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _collect)

        result = provider.generate("a cat", aspect_ratio="portrait")
        assert result["success"] is True

        assert captured["model"] == "gpt-5.5"
        assert captured["store"] is False
        assert captured["input"][0]["type"] == "message"
        assert captured["input"][0]["role"] == "user"
        assert captured["input"][0]["content"][0]["type"] == "input_text"
        # Regression for #19505: the Codex backend 400s on every tool_choice
        # shape we have for the hosted ``image_generation`` tool, so the
        # provider must omit tool_choice entirely and rely on instructions.
        assert "tool_choice" not in captured

        tool = captured["tools"][0]
        assert tool["type"] == "image_generation"
        assert tool["model"] == "gpt-image-2"
        assert tool["quality"] == "medium"
        assert tool["size"] == "1024x1536"
        assert tool["output_format"] == "png"
        assert tool["background"] == "opaque"
        # Progressive previews disabled: partial frames were being saved as
        # finals and presented as smeared/unfinished images.
        assert tool["partial_images"] == 0

    def test_capabilities_advertise_image_inputs(self, provider):
        caps = provider.capabilities()
        assert caps["modalities"] == ["text", "image"]
        assert caps["max_reference_images"] == 16


    def test_rejects_non_image_local_source(self, provider, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        text_path = tmp_path / "not-image.txt"
        text_path.write_text("hello")

        result = provider.generate("edit this", image_url=str(text_path))

        assert result["success"] is False
        assert result["error_type"] == "invalid_image_input"
        assert "not a supported image" in result["error"]


    def test_partial_image_event_used_when_done_missing(self):
        """Extractor may surface partial b64 when no final exists (fallback only)."""
        payload = {
            "type": "response.image_generation_call.partial_image",
            "partial_image_b64": _b64_png(),
        }
        assert codex_plugin._extract_image_b64(payload) == _b64_png()
        result, partial = codex_plugin._extract_image_candidates(payload)
        assert result is None
        assert partial == _b64_png()

    def test_final_result_wins_over_coexisting_partial_in_same_payload(self):
        """Blind spot that shipped the smear bug: both fields in one payload.

        partial_image_b64 must never overwrite image_generation_call.result
        when they coexist in the same event tree.
        """
        final = _b64_png()
        # Distinct non-empty stand-in so equality proves which field won.
        partial = "cGFydGlhbC1vbmx5LW5vdC1hLXJlYWwtZmluYWw="
        payload = {
            "type": "response.output_item.done",
            "item": {
                "type": "image_generation_call",
                "status": "completed",
                "result": final,
                "partial_image_b64": partial,
            },
        }
        assert codex_plugin._extract_image_b64(payload) == final
        result, got_partial = codex_plugin._extract_image_candidates(payload)
        assert result == final
        assert got_partial == partial

    def test_nested_final_wins_over_sibling_partial(self):
        payload = {
            "type": "response.completed",
            "response": {
                "output": [{
                    "type": "image_generation_call",
                    "status": "completed",
                    "result": _b64_png(),
                }],
            },
            "partial_image_b64": "cGFydGlhbC1zaWJsaW5n",
        }
        assert codex_plugin._extract_image_b64(payload) == _b64_png()

    def test_sse_parser_handles_event_and_data_lines(self):
        class _Response:
            def iter_lines(self):
                return iter([
                    "event: response.output_item.done",
                    'data: {"item": {"type": "image_generation_call", "result": "abc"}}',
                    "",
                ])

        events = list(codex_plugin._iter_sse_json(_Response()))
        assert events == [{
            "type": "response.output_item.done",
            "item": {"type": "image_generation_call", "result": "abc"},
        }]

    def test_final_response_sweep_recovers_image(self):
        """Completed response output is found by recursive payload scanning."""
        payload = {
            "type": "response.completed",
            "response": {
                "output": [{
                    "type": "image_generation_call",
                    "status": "completed",
                    "id": "ig_final",
                    "result": _b64_png(),
                }],
            },
        }
        assert codex_plugin._extract_image_b64(payload) == _b64_png()

    def test_partial_only_stream_fails_closed_after_retry(self, provider, monkeypatch):
        """Partial-only streams must not return success:true with a smear frame."""
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        calls = {"n": 0}

        def _partial_only(*args, **kwargs):
            calls["n"] += 1
            return {"b64": _b64_png(), "source": "partial"}

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _partial_only)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "incomplete_image"
        assert "partial" in result["error"].lower()
        # One initial attempt + one content-agnostic retry.
        assert calls["n"] == codex_plugin._NONFINAL_RETRIES + 1

    def test_empty_stream_retries_then_fails(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        calls = {"n": 0}

        def _empty(*args, **kwargs):
            calls["n"] += 1
            return None

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _empty)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "empty_response"
        assert calls["n"] == codex_plugin._NONFINAL_RETRIES + 1

    def test_partial_then_final_on_retry_succeeds(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        calls = {"n": 0}

        def _then_final(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"b64": _b64_png(), "source": "partial"}
            return {"b64": _b64_png(), "source": "final"}

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _then_final)

        result = provider.generate("a cat")
        assert result["success"] is True
        assert result.get("image_source") == "final"
        assert calls["n"] == 2

    def test_empty_then_final_on_retry_succeeds(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        calls = {"n": 0}

        def _then_final(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return None
            return {"b64": _b64_png(), "source": "final"}

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _then_final)

        result = provider.generate("a cat")
        assert result["success"] is True
        assert result.get("image_source") == "final"
        assert calls["n"] == 2

    def test_empty_response_returns_error(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        monkeypatch.setattr(codex_plugin, "_NONFINAL_RETRIES", 0)
        monkeypatch.setattr(codex_plugin, "_collect_image_b64", lambda *a, **kw: None)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_stream_exception_returns_api_error(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

        def _boom(*args, **kwargs):
            raise RuntimeError("cloudflare 403")

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _boom)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "cloudflare 403" in result["error"]

    def test_tool_choice_400_surfaces_verbatim_not_as_capability_error(
        self, provider, monkeypatch
    ):
        """The tool_choice 400 must NOT be reported as an account limitation.

        Regression for #19505 / #49008 / #31335: a previous version classified
        this exact request-shape rejection as "Image generation is not enabled
        for the current Codex account", telling every affected user to abandon
        Codex over a bug in our own payload. The wire error must reach the user
        unedited so it stays diagnosable.

        Drives the REAL httpx boundary (not a mocked ``_collect_image_b64``) so
        the classification path is actually exercised — mocking the collector
        would skip the code under test entirely.
        """
        import httpx

        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

        body = json.dumps({
            "error": {
                "message": "Tool choice 'image_generation' not found in 'tools' parameter.",
                "type": "invalid_request_error",
                "param": "tool_choice",
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
        assert "tools' parameter" in result["error"]
        # The account-entitlement misdiagnosis must not come back.
        assert "not enabled for the current Codex account" not in result["error"]
        assert result["error_type"] != "capability_unsupported"
        # Do not infer account entitlement or image-model rejection from error prose.
        assert "compatibility" not in result["error"].lower()


class TestRequestShape:
    def test_payload_omits_tool_choice(self):
        """Codex rejects every tool_choice shape for hosted image_generation."""
        payload = codex_plugin._build_responses_payload(
            prompt="a red circle",
            size="1024x1024",
            quality="low",
        )
        assert "tool_choice" not in payload
        # The hosted tool itself is still requested, and instructions do the steering.
        assert payload["tools"][0]["type"] == "image_generation"
        assert payload["instructions"]

    def test_http_error_body_is_truncated_but_preserved(self, monkeypatch):
        """A large error body is capped at 500 chars and still surfaced."""
        import httpx

        body = json.dumps({
            "metadata": "x" * 600,
            "error": {
                "message": "Tool choice 'image_generation' not found in 'tools' parameter."
            },
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
        assert "tools' parameter" in message
        assert len(message) < len(body)


# ── GPT Image 2.5 Flare / Sunburst (issue #106708) ──────────────────────────


_GPT_IMAGE_25_MODELS = ("gpt-image-2.5-flare", "gpt-image-2.5-sunburst")
_GPT_IMAGE_25_QUALITIES = ("auto", "low", "medium", "high", "xhigh", "max")
_GPT_IMAGE_25_IDS = tuple(
    model if quality == "auto" else f"{model}-{quality}"
    for model in _GPT_IMAGE_25_MODELS
    for quality in _GPT_IMAGE_25_QUALITIES
)


def _catalog_id(api_model: str, quality: str) -> str:
    return api_model if quality == "auto" else f"{api_model}-{quality}"


def _set_codex_model(tmp_path, model_id: str) -> None:
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"image_gen": {"openai-codex": {"model": model_id}}})
    )


def _capture_payload(captured: dict):
    """Record ``_collect_image_b64`` kwargs and the Responses payload they produce."""

    def _collect(token, **kwargs):
        captured["collect_kwargs"] = kwargs
        payload_kwargs = {
            "prompt": kwargs["prompt"],
            "size": kwargs["size"],
            "quality": kwargs["quality"],
            "input_images": kwargs.get("input_images"),
        }
        import inspect

        params = inspect.signature(codex_plugin._build_responses_payload).parameters
        if "api_model" in params and kwargs.get("api_model"):
            payload_kwargs["api_model"] = kwargs["api_model"]
        captured.update(codex_plugin._build_responses_payload(**payload_kwargs))
        return {"b64": _b64_png(), "source": "final"}

    return _collect


class TestGptImage25Catalog:
    def test_list_models_includes_flare_and_sunburst_quality_variants(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        for model_id in _GPT_IMAGE_25_IDS:
            assert model_id in ids
        # GPT Image 2 three-tier catalog is retained (CONTROL).
        assert "gpt-image-2-low" in ids
        assert "gpt-image-2-medium" in ids
        assert "gpt-image-2-high" in ids

    def test_picker_ids_match_resolvable_catalog(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        assert set(ids) == set(provider.models)
        assert provider.default_model() in ids
        assert provider.default_model() == "gpt-image-2-medium"


class TestGptImage25Payload:
    def test_flare_text_to_image_sends_api_model_not_gpt_image_2(
        self, provider, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        _set_codex_model(tmp_path, "gpt-image-2.5-flare")
        captured = {}
        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _capture_payload(captured))

        result = provider.generate("a cat")
        assert result["success"] is True
        assert result["model"] == "gpt-image-2.5-flare"
        assert captured["tools"][0]["model"] == "gpt-image-2.5-flare"
        assert captured["tools"][0]["quality"] == "auto"
        assert captured["collect_kwargs"].get("api_model") == "gpt-image-2.5-flare"
        # Must not silently rewrite to GPT Image 2.
        assert captured["tools"][0]["model"] != "gpt-image-2"

    def test_sunburst_edit_with_reference_image_sends_api_model(
        self, provider, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        _set_codex_model(tmp_path, "gpt-image-2.5-sunburst")
        source = tmp_path / "ref.png"
        source.write_bytes(bytes.fromhex(_PNG_HEX))
        captured = {}
        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _capture_payload(captured))

        result = provider.generate(
            "edit this", image_url=str(source), reference_image_urls=[str(source)]
        )
        assert result["success"] is True
        assert result["model"] == "gpt-image-2.5-sunburst"
        assert captured["tools"][0]["model"] == "gpt-image-2.5-sunburst"
        assert captured["tools"][0]["quality"] == "auto"
        content = captured["input"][0]["content"]
        image_parts = [part for part in content if part.get("type") == "input_image"]
        assert len(image_parts) >= 1
        assert captured["tools"][0]["model"] != "gpt-image-2"

    def test_gpt_image_2_medium_default_payload_still_gpt_image_2(
        self, provider, monkeypatch
    ):
        """CONTROL: default catalog selection must keep sending gpt-image-2."""
        monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        captured = {}
        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _capture_payload(captured))

        result = provider.generate("a cat")
        assert result["success"] is True
        assert result["model"] == "gpt-image-2-medium"
        assert captured["tools"][0]["model"] == "gpt-image-2"
        assert captured["tools"][0]["quality"] == "medium"

    @pytest.mark.parametrize("api_model,quality", [
        (model, quality)
        for model in _GPT_IMAGE_25_MODELS
        for quality in _GPT_IMAGE_25_QUALITIES
    ])
    def test_quality_variant_reaches_image_generation_tool(
        self, provider, monkeypatch, tmp_path, api_model, quality
    ):
        monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        tier = _catalog_id(api_model, quality)
        _set_codex_model(tmp_path, tier)
        captured = {}
        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _capture_payload(captured))

        result = provider.generate("a cat")
        assert result["success"] is True
        assert result["model"] == tier
        assert result["quality"] == quality
        assert captured["tools"][0]["model"] == api_model
        assert captured["tools"][0]["quality"] == quality

    def test_unknown_model_id_falls_through_to_gpt_image_2_medium(
        self, provider, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        _set_codex_model(tmp_path, "not-a-real-image-model")
        captured = {}
        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _capture_payload(captured))

        result = provider.generate("a cat")
        assert result["success"] is True
        assert result["model"] == "gpt-image-2-medium"
        assert captured["tools"][0]["model"] == "gpt-image-2"

    def test_unsupported_model_http_error_is_preserved_as_api_error_not_fallback(
        self, provider, monkeypatch, tmp_path
    ):
        """Unknown/unsupported model from Codex must surface; never silently switch providers."""
        import httpx

        monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        _set_codex_model(tmp_path, "gpt-image-2.5-flare")

        body = json.dumps({
            "error": {
                "message": "Unknown model: gpt-image-2.5-flare is not supported",
                "type": "invalid_request_error",
            }
        })

        seen = {}

        def _handler(request):
            seen["json"] = json.loads(request.content)
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
        assert seen["json"]["tools"][0]["model"] == "gpt-image-2.5-flare"
        assert result["success"] is False
        assert result["error_type"] == "api_error"
        err = result["error"].lower()
        assert "Unknown model: gpt-image-2.5-flare is not supported" in result["error"]
        assert "gpt-image-2.5-flare" in result["error"]
        assert "unknown model" in err or "not supported" in err
        assert "not enabled for the current Codex account" not in result["error"]
        # Must not silently retarget OPENAI_API_KEY or FAL.
        assert "OPENAI_API_KEY" not in result["error"]
        assert " fal" not in err


def _mock_codex_stream(monkeypatch, responses, requests):
    """Replace only HTTP transport; keep request construction, SSE parsing and saving real."""
    import httpx

    streams = iter(responses)

    def respond(request):
        assert str(request.url) == f"{codex_plugin._CODEX_BASE_URL}/responses"
        requests.append(json.loads(request.content))
        wire = "".join(f"data: {json.dumps(event)}\n\n" for event in next(streams))
        return httpx.Response(200, text=wire, headers={"content-type": "text/event-stream"})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: real_client(
        transport=httpx.MockTransport(respond), **kwargs,
    ))


@pytest.mark.parametrize("model_id", [
    "gpt-image-2-medium", *[f"{model}-low" for model in _GPT_IMAGE_25_MODELS],
])
@pytest.mark.parametrize("with_reference", [False, True], ids=["generate", "edit"])
@pytest.mark.parametrize("report", ["alias", "echo", "missing"])
@pytest.mark.parametrize("metadata_event", ["response.created", "response.completed"])
def test_success_separates_requested_model_from_unverified_server_report(
    provider, monkeypatch, tmp_path, model_id, with_reference, report, metadata_event,
):
    monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
    monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
    _set_codex_model(tmp_path, model_id)
    original_config = (tmp_path / "config.yaml").read_bytes()
    api_model = codex_plugin.MODELS[model_id]["api_model"]
    reported = {"alias": "gpt-image-2-codex", "echo": api_model, "missing": None}[report]
    tools = [{"type": "function", "name": "unrelated", "model": "not-the-image-model"}]
    if reported is not None:
        tools.append({"type": "image_generation", "model": reported})
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {"model": "host-model"}},
        {"type": "response.completed", "response": {
            "model": "host-model", "output": [{
                "type": "image_generation_call", "status": "completed", "result": _b64_png(),
            }],
        }},
    ]
    for event in events:
        if event["type"] == metadata_event:
            event["response"]["tools"] = tools
    requests = []
    _mock_codex_stream(monkeypatch, [events], requests)
    kwargs = {}
    if with_reference:
        source = tmp_path / "reference.png"
        source.write_bytes(bytes.fromhex(_PNG_HEX))
        kwargs["image_url"] = str(source)

    result = provider.generate("A blue circle on white.", **kwargs)

    assert len(requests) == 1  # A model label must not cause retries or a provider switch.
    assert requests[0]["tools"][0]["model"] == api_model
    content = requests[0]["input"][0]["content"]
    assert any(part["type"] == "input_image" for part in content) == with_reference
    assert result["success"] is True
    assert Path(result["image"]).read_bytes() == bytes.fromhex(_PNG_HEX)
    assert result["provider"] == "openai-codex"
    assert result["model"] == model_id  # Preserve the existing catalog-selection field.
    assert result["requested_model"] == api_model
    assert result["reported_model"] == reported
    # Even an echoed tool configuration is not verified image-engine identity.
    assert result["model_selection_verified"] is False
    assert "unverified" in result["model_selection_note"].lower()
    assert not result.get("error")
    assert (tmp_path / "config.yaml").read_bytes() == original_config


@pytest.mark.parametrize("created_tools,completed_tools,expected", [
    ([{"type": "image_generation", "model": "initial-label"}],
     [{"type": "image_generation", "model": "final-label"}], "final-label"),
    ([{"type": "image_generation", "model": "initial-label"}], None, "initial-label"),
    (None, None, None),
    (None, {"type": "image_generation", "model": "not-a-tools-list"}, None),
    (None, [None, {"type": "image_generation", "model": {"unexpected": "object"}}], None),
    (None, [{"type": "image_generation", "model": "   "}], None),
])
def test_reported_model_belongs_to_successful_attempt_not_a_previous_partial(
    provider, monkeypatch, created_tools, completed_tools, expected,
):
    monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
    partial_attempt = [{"type": "response.created", "response": {"tools": [
        {"type": "image_generation", "model": "discarded-attempt-label"},
    ]}}, {"type": "response.image_generation_call.partial_image", "partial_image_b64": _b64_png()}]
    final_attempt = [
        {"type": "response.created", "response": {"tools": created_tools}},
        {"type": "response.completed", "response": {
            "model": "host-model", "tools": completed_tools,
            "output": [{"type": "image_generation_call", "result": _b64_png()}],
        }},
    ]
    requests = []
    _mock_codex_stream(monkeypatch, [partial_attempt, final_attempt], requests)

    result = provider.generate("A blue circle on white.")

    assert len(requests) == 2
    assert result["success"] is True
    assert result["image_source"] == "final"
    assert Path(result["image"]).read_bytes() == bytes.fromhex(_PNG_HEX)
    assert result["reported_model"] == expected
    assert result["model_selection_verified"] is False


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
