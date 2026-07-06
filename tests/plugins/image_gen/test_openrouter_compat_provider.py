#!/usr/bin/env python3
"""Tests for the OpenRouter-compatible image gen provider (OpenRouter + Nous)."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_RUNTIME = "hermes_cli.runtime_provider.resolve_runtime_provider"
_PNG_DATA_URI = "data:image/png;base64,dGVzdC1pbWFnZS1kYXRh"  # "test-image-data"


def _runtime_ok(**over):
    base = {
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "sk-or-test",
        "source": "env",
    }
    base.update(over)
    return base


def _mock_chat_response(images):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "images": [
                        {"type": "image_url", "image_url": {"url": u}} for u in images
                    ],
                }
            }
        ]
    }
    return resp


def _openrouter():
    from plugins.image_gen.openrouter import OpenRouterCompatImageProvider

    return OpenRouterCompatImageProvider(
        provider_name="openrouter",
        display_name="OpenRouter",
        runtime_name="openrouter",
        config_key="openrouter",
        model_env_var="OPENROUTER_IMAGE_MODEL",
        setup_schema={"name": "OpenRouter (image)", "badge": "paid", "env_vars": []},
    )


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class TestProviderClass:
    def test_names(self):
        from plugins.image_gen.openrouter import _build_providers

        names = {p.name for p in _build_providers()}
        assert names == {"openrouter", "nous"}

    def test_display_names(self):
        from plugins.image_gen.openrouter import _build_providers

        by_name = {p.name: p for p in _build_providers()}
        assert by_name["openrouter"].display_name == "OpenRouter"
        assert by_name["nous"].display_name == "Nous Portal"

    def test_capabilities_support_image_input(self):
        caps = _openrouter().capabilities()
        assert "image" in caps["modalities"]
        assert caps["max_reference_images"] >= 1

    def test_is_available_with_key(self):
        with patch(_RUNTIME, return_value=_runtime_ok()):
            assert _openrouter().is_available() is True

    def test_is_available_without_key(self):
        with patch(_RUNTIME, return_value=_runtime_ok(api_key="")):
            assert _openrouter().is_available() is False

    def test_is_available_on_resolution_error(self):
        with patch(_RUNTIME, side_effect=RuntimeError("boom")):
            assert _openrouter().is_available() is False

    def test_default_model(self):
        from plugins.image_gen.openrouter import DEFAULT_MODEL

        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value={}):
            assert _openrouter().default_model() == DEFAULT_MODEL
            # Default must be an image-output model id (provider/model form).
            assert "/" in DEFAULT_MODEL and "image" in DEFAULT_MODEL

    def test_default_chain_prefers_quality_then_fallback(self):
        from plugins.image_gen.openrouter import _FALLBACK_MODEL, _DEFAULT_MODEL_CHAIN

        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value={}):
            chain = _openrouter()._resolve_model_chain()
        assert chain == list(_DEFAULT_MODEL_CHAIN)
        assert chain[0].startswith("openai/")
        assert chain[-1] == _FALLBACK_MODEL

    def test_model_env_override(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_IMAGE_MODEL", "black-forest-labs/flux.2-pro")
        assert _openrouter()._resolve_model() == "black-forest-labs/flux.2-pro"
        assert _openrouter()._resolve_model_chain() == ["black-forest-labs/flux.2-pro"]

    def test_model_config_override(self):
        cfg = {"openrouter": {"model": "google/gemini-3.1-flash-image-preview"}}
        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value=cfg):
            assert _openrouter()._resolve_model() == "google/gemini-3.1-flash-image-preview"

    def test_model_top_level_config_override(self):
        cfg = {"model": "openai/gpt-image-2"}
        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value=cfg):
            assert _openrouter()._resolve_model_chain() == ["openai/gpt-image-2"]

    def test_nous_honors_top_level_model(self):
        from plugins.image_gen.openrouter import _build_providers

        cfg = {"model": "openai/gpt-image-2"}
        nous = {p.name: p for p in _build_providers()}["nous"]
        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value=cfg):
            assert nous._resolve_model_chain() == ["openai/gpt-image-2"]

    def test_explicit_model_kwarg_wins_over_config(self):
        cfg = {"model": "openai/gpt-image-2"}
        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value=cfg):
            assert _openrouter()._resolve_model_chain("google/gemini-3-pro-image") == [
                "google/gemini-3-pro-image"
            ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_to_image_url_part_passthrough_url(self):
        from plugins.image_gen.openrouter import _to_image_url_part

        assert _to_image_url_part("https://x/y.png") == "https://x/y.png"
        assert _to_image_url_part("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"

    def test_to_image_url_part_inlines_local_file(self, tmp_path):
        from plugins.image_gen.openrouter import _to_image_url_part

        f = tmp_path / "base.png"
        f.write_bytes(b"\x89PNG\r\n")
        part = _to_image_url_part(str(f))
        assert part.startswith("data:image/png;base64,")
        decoded = base64.b64decode(part.split(",", 1)[1])
        assert decoded == b"\x89PNG\r\n"

    def test_to_image_url_part_missing_file(self):
        from plugins.image_gen.openrouter import _to_image_url_part

        assert _to_image_url_part("/no/such/file.png") is None

    def test_to_image_url_part_blocks_credential_store(self, tmp_path, monkeypatch):
        from plugins.image_gen.openrouter import _to_image_url_part

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        auth_json = hermes_home / "auth.json"
        auth_json.write_text('{"api_key":"sk-secret"}', encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        with pytest.raises(ValueError, match="credential store"):
            _to_image_url_part(str(auth_json))

    def test_to_image_url_part_never_reads_blocked_credential(self, tmp_path, monkeypatch):
        """The guard must fire BEFORE path.read_bytes() — the credential store
        must never be inlined into a provider request (#57698)."""
        from pathlib import Path as _P

        from plugins.image_gen.openrouter import _to_image_url_part

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        auth_json = hermes_home / "auth.json"
        auth_json.write_text('{"api_key":"sk-secret"}', encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        real_read_bytes = _P.read_bytes
        read: list = []

        def _spy_read_bytes(self, *a, **k):
            read.append(str(self))
            return real_read_bytes(self, *a, **k)

        monkeypatch.setattr(_P, "read_bytes", _spy_read_bytes)
        with pytest.raises(ValueError, match="credential store"):
            _to_image_url_part(str(auth_json))
        assert str(auth_json) not in read, "blocked credential must never be read"

    def test_extract_images(self):
        from plugins.image_gen.openrouter import _extract_images

        payload = {
            "choices": [
                {"message": {"images": [{"image_url": {"url": "data:image/png;base64,AA"}}]}}
            ]
        }
        assert _extract_images(payload) == ["data:image/png;base64,AA"]

    def test_extract_images_empty(self):
        from plugins.image_gen.openrouter import _extract_images

        assert _extract_images({"choices": [{"message": {"content": "no image"}}]}) == []

    def test_access_error_hint_for_gated_openai_model(self):
        from plugins.image_gen.openrouter import _FALLBACK_MODEL, _access_error_hint

        hint = _access_error_hint(
            "OpenRouter", "openai/gpt-5.4-image-2", "OPENROUTER_IMAGE_MODEL", 404, "No endpoints found"
        )
        assert hint is not None
        assert "openai/gpt-5.4-image-2" in hint
        assert "OPENROUTER_IMAGE_MODEL" in hint
        assert _FALLBACK_MODEL in hint
        # Stays a single line under the humanizer's 200-char truncation.
        assert "\n" not in hint and len(hint) <= 200

    def test_access_error_hint_ignores_non_openai_models(self):
        from plugins.image_gen.openrouter import _access_error_hint

        assert _access_error_hint("OpenRouter", "google/gemini-3-pro-image", "X", 404, "boom") is None

    def test_access_error_hint_ignores_unrelated_errors(self):
        from plugins.image_gen.openrouter import _access_error_hint

        # A 200-class transient with an openai model but no access signal → no hint.
        assert _access_error_hint("OpenRouter", "openai/gpt-5.4-image-2", "X", 500, "server error") is None


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_missing_credentials(self):
        with patch(_RUNTIME, return_value=_runtime_ok(api_key="")):
            result = _openrouter().generate(prompt="a pet")
        assert result["success"] is False
        assert result["error_type"] == "missing_api_key"

    def test_success_data_uri(self):
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_chat_response([_PNG_DATA_URI])), \
             patch(
                 "plugins.image_gen.openrouter.save_b64_image",
                 return_value=Path("/tmp/openrouter_gen.png"),
             ) as mock_save:
            result = _openrouter().generate(prompt="a pet")

        assert result["success"] is True
        assert result["image"] == "/tmp/openrouter_gen.png"
        assert result["provider"] == "openrouter"
        mock_save.assert_called_once()

    def test_success_http_url(self):
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_chat_response(["https://cdn/x.png"])), \
             patch(
                 "plugins.image_gen.openrouter.save_url_image",
                 return_value=Path("/tmp/openrouter_gen_url.png"),
             ) as mock_save_url:
            result = _openrouter().generate(prompt="a pet")

        assert result["success"] is True
        assert result["image"] == "/tmp/openrouter_gen_url.png"
        mock_save_url.assert_called_once()

    def test_empty_response(self):
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_chat_response([])):
            result = _openrouter().generate(prompt="a pet")
        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_payload_shape_and_references(self, tmp_path):
        """Wire payload must carry image modalities, aspect_ratio, and the
        reference image inlined as a data URI (this is what makes pet rows
        stay on-model)."""
        ref = tmp_path / "base.png"
        ref.write_bytes(b"\x89PNG\r\n")

        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_chat_response([_PNG_DATA_URI])) as mock_post, \
             patch("plugins.image_gen.openrouter.save_b64_image", return_value=Path("/tmp/x.png")):
            _openrouter().generate(
                prompt="a pet", aspect_ratio="square", reference_images=[str(ref)]
            )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["modalities"] == ["image", "text"]
        assert payload["image_config"]["aspect_ratio"] == "1:1"
        content = payload["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "a pet"}
        image_parts = [c for c in content if c["type"] == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_auth_header(self):
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_chat_response([_PNG_DATA_URI])) as mock_post, \
             patch("plugins.image_gen.openrouter.save_b64_image", return_value=Path("/tmp/x.png")):
            _openrouter().generate(prompt="a pet")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-or-test"

    def test_generate_uses_model_kwarg_from_dispatch(self):
        """image_generate passes image_gen.model as a model kwarg — honor it."""
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_chat_response([_PNG_DATA_URI])) as mock_post, \
             patch("plugins.image_gen.openrouter.save_b64_image", return_value=Path("/tmp/x.png")):
            result = _openrouter().generate(prompt="a pet", model="openai/gpt-image-2")

        assert result["success"] is True
        assert result["model"] == "openai/gpt-image-2"
        assert mock_post.call_args.kwargs["json"]["model"] == "openai/gpt-image-2"

    def test_posts_to_resolved_base_url(self):
        """Nous routes to its own base URL — proves the same code serves both."""
        nous_runtime = _runtime_ok(
            provider="nous", base_url="https://inference.nousresearch.com/v1", api_key="nous-tok"
        )
        with patch(_RUNTIME, return_value=nous_runtime), \
             patch("requests.post", return_value=_mock_chat_response([_PNG_DATA_URI])) as mock_post, \
             patch("plugins.image_gen.openrouter.save_b64_image", return_value=Path("/tmp/x.png")):
            from plugins.image_gen.openrouter import _build_providers

            nous = {p.name: p for p in _build_providers()}["nous"]
            result = nous.generate(prompt="a pet")

        assert result["success"] is True
        assert result["provider"] == "nous"
        url = mock_post.call_args[0][0]
        assert url == "https://inference.nousresearch.com/v1/chat/completions"

    def test_api_error(self):
        import requests as req_lib

        resp = MagicMock()
        resp.status_code = 401
        resp.text = "Unauthorized"
        resp.json.return_value = {"error": {"message": "Invalid API key"}}
        resp.raise_for_status.side_effect = req_lib.HTTPError(response=resp)

        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=resp) as mock_post:
            result = _openrouter().generate(prompt="a pet")
        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert mock_post.call_count == 1

    def test_timeout(self):
        import requests as req_lib

        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", side_effect=req_lib.Timeout()):
            result = _openrouter().generate(prompt="a pet")
        assert result["success"] is False
        assert result["error_type"] == "timeout"

    def test_access_gated_model_surfaces_hint(self, monkeypatch):
        """A 404 on an OpenAI image model yields the actionable access hint (not
        the misleading generic 'check your key' message)."""
        import requests as req_lib

        monkeypatch.setenv("OPENROUTER_IMAGE_MODEL", "openai/gpt-5.4-image-2")
        resp = MagicMock()
        resp.status_code = 404
        resp.text = "No endpoints found for openai/gpt-5.4-image-2"
        resp.json.return_value = {"error": {"message": "No endpoints found"}}
        resp.raise_for_status.side_effect = req_lib.HTTPError(response=resp)

        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=resp) as mock_post:
            result = _openrouter().generate(prompt="a pet")

        assert result["success"] is False
        assert result["error_type"] == "model_access"
        assert "OpenAI image access" in result["error"]
        assert mock_post.call_count == 1  # explicit override: no auto-fallback chain

    def test_access_gated_default_model_falls_back_to_gemini(self):
        import requests as req_lib

        from plugins.image_gen.openrouter import DEFAULT_MODEL, _FALLBACK_MODEL

        gated = MagicMock()
        gated.status_code = 404
        gated.text = f"No endpoints found for {DEFAULT_MODEL}"
        gated.json.return_value = {"error": {"message": "No endpoints found"}}
        gated.raise_for_status.side_effect = req_lib.HTTPError(response=gated)

        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", side_effect=[gated, _mock_chat_response([_PNG_DATA_URI])]) as mock_post, \
             patch(
                 "plugins.image_gen.openrouter.save_b64_image",
                 return_value=Path("/tmp/openrouter_gen_fallback.png"),
             ):
            result = _openrouter().generate(prompt="a pet")

        assert result["success"] is True
        assert result["model"] == _FALLBACK_MODEL
        assert result["image"] == "/tmp/openrouter_gen_fallback.png"
        assert mock_post.call_count == 2
        first_model = mock_post.call_args_list[0].kwargs["json"]["model"]
        second_model = mock_post.call_args_list[1].kwargs["json"]["model"]
        assert first_model == DEFAULT_MODEL
        assert second_model == _FALLBACK_MODEL


# ---------------------------------------------------------------------------
# Registration + pet integration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_both(self):
        from plugins.image_gen.openrouter import register

        ctx = MagicMock()
        register(ctx)
        registered = [c.args[0].name for c in ctx.register_image_gen_provider.call_args_list]
        assert set(registered) == {"openrouter", "nous"}

    def test_both_are_reference_capable_for_pets(self):
        from agent.pet.generate.imagegen import _REF_CAPABLE

        assert "openrouter" in _REF_CAPABLE
        assert "nous" in _REF_CAPABLE
