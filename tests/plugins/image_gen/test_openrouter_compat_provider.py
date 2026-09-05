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


    def test_default_model(self):
        from plugins.image_gen.openrouter import DEFAULT_MODEL

        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value={}):
            assert _openrouter().default_model() == DEFAULT_MODEL
            # Default must be an image-output model id (provider/model form).
            assert "/" in DEFAULT_MODEL and "image" in DEFAULT_MODEL

    def test_default_model_ignores_runtime_overrides(self, monkeypatch):
        """Catalog defaults must not inherit another provider's saved model."""
        from plugins.image_gen.openrouter import DEFAULT_MODEL

        monkeypatch.setenv("OPENROUTER_IMAGE_MODEL", "custom/provider-image-model")
        stale = {"model": "gpt-image-2-medium"}
        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value=stale):
            provider = _openrouter()
            assert provider.default_model() == DEFAULT_MODEL
            assert provider._resolve_model() == "custom/provider-image-model"


    def test_model_env_override(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_IMAGE_MODEL", "black-forest-labs/flux.2-pro")
        assert _openrouter()._resolve_model() == "black-forest-labs/flux.2-pro"
        assert _openrouter()._resolve_model_chain() == ["black-forest-labs/flux.2-pro"]


    def test_nous_honors_top_level_model(self):
        from plugins.image_gen.openrouter import _build_providers

        cfg = {"model": "openai/gpt-image-2"}
        nous = {p.name: p for p in _build_providers()}["nous"]
        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value=cfg):
            assert nous._resolve_model_chain() == ["openai/gpt-image-2"]

    def test_configured_primary_and_fallback_models(self):
        cfg = {
            "model": "google/gemini-3.1-flash-image",
            "fallback_models": ["openai/gpt-5.4-image-2"],
        }
        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value=cfg):
            assert _openrouter()._resolve_model_chain() == [
                "google/gemini-3.1-flash-image",
                "openai/gpt-5.4-image-2",
            ]

    def test_scoped_fallback_models_override_top_level_fallbacks(self):
        cfg = {
            "model": "google/gemini-3.1-flash-image",
            "fallback_models": ["top-level/fallback"],
            "openrouter": {"fallback_models": ["openai/gpt-5.4-image-2"]},
        }
        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value=cfg):
            assert _openrouter()._resolve_model_chain() == [
                "google/gemini-3.1-flash-image",
                "openai/gpt-5.4-image-2",
            ]

    def test_explicit_model_kwarg_still_bypasses_configured_fallback(self):
        cfg = {
            "model": "google/gemini-3.1-flash-image",
            "fallback_models": ["openai/gpt-5.4-image-2"],
        }
        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value=cfg):
            assert _openrouter()._resolve_model_chain("some/explicit-model") == [
                "some/explicit-model"
            ]

    def test_explicit_model_kwarg_wins_over_config(self):
        cfg = {"model": "openai/gpt-image-2"}
        with patch("plugins.image_gen.openrouter._load_image_gen_config", return_value=cfg):
            assert _openrouter()._resolve_model_chain("google/gemini-3-pro-image") == [
                "google/gemini-3-pro-image"
            ]


# ---------------------------------------------------------------------------
# Live model catalog
# ---------------------------------------------------------------------------


def _mock_models_response(entries):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"data": entries}
    return resp


class TestLiveCatalog:
    def test_live_catalog_lists_all_image_output_models(self):
        """Every image-output model on the endpoint is selectable — including
        ones released after this code shipped."""
        entries = [
            {
                "id": "openai/gpt-5.4-image-2",
                "name": "GPT-5.4 Image 2",
                "architecture": {"output_modalities": ["image"], "input_modalities": ["text", "image"]},
            },
            {
                "id": "some-lab/brand-new-image-model",
                "name": "Brand New",
                "architecture": {"output_modalities": ["image", "text"], "input_modalities": ["text"]},
            },
            {
                "id": "openai/gpt-5.4",  # text-only: excluded
                "architecture": {"output_modalities": ["text"], "input_modalities": ["text"]},
            },
            {
                "id": "openrouter/auto",  # router pseudo-model: excluded
                "architecture": {"output_modalities": ["image", "text"], "input_modalities": ["text"]},
            },
        ]
        provider = _openrouter()
        with patch(_RUNTIME, return_value=_runtime_ok()), patch(
            "requests.get", return_value=_mock_models_response(entries)
        ):
            models = provider.list_models()
        ids = [m["id"] for m in models]
        assert "openai/gpt-5.4-image-2" in ids
        assert "some-lab/brand-new-image-model" in ids
        assert "openai/gpt-5.4" not in ids
        assert "openrouter/auto" not in ids
        # Default chain models sort first.
        assert ids[0] == "openai/gpt-5.4-image-2"

    def test_live_failure_falls_back_to_static_chain(self):
        provider = _openrouter()
        with patch(_RUNTIME, side_effect=RuntimeError("no creds")):
            models = provider.list_models()
        from plugins.image_gen.openrouter import DEFAULT_MODEL, _FALLBACK_MODEL

        assert [m["id"] for m in models] == [DEFAULT_MODEL, _FALLBACK_MODEL]

    def test_live_catalog_is_cached(self):
        provider = _openrouter()
        entries = [
            {
                "id": "openai/gpt-5.4-image-2",
                "architecture": {"output_modalities": ["image"], "input_modalities": ["text"]},
            }
        ]
        with patch(_RUNTIME, return_value=_runtime_ok()), patch(
            "requests.get", return_value=_mock_models_response(entries)
        ) as mock_get:
            provider.list_models()
            provider.list_models()
        assert mock_get.call_count == 1

    def test_picker_merges_image_api_and_chat_catalogs(self):
        """OpenRouter picker = union of /images/models and image-output
        /models entries, deduped, defaults first."""
        from plugins.image_gen.openrouter import _build_providers

        orp = {p.name: p for p in _build_providers()}["openrouter"]

        def fake_get(url, **kw):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if url.endswith("/images/models"):
                resp.json.return_value = {"data": [
                    {"id": "bytedance-seed/seedream-4.5", "name": "Seedream 4.5",
                     "architecture": {"input_modalities": ["text", "image"],
                                      "output_modalities": ["image"]}},
                    {"id": "openai/gpt-5.4-image-2", "name": "GPT-5.4 Image 2",
                     "architecture": {"input_modalities": ["text", "image"],
                                      "output_modalities": ["image"]}},
                ]}
            else:
                resp.json.return_value = {"data": [
                    {"id": "openai/gpt-5.4-image-2",
                     "architecture": {"output_modalities": ["image"],
                                      "input_modalities": ["text", "image"]}},
                    {"id": "google/gemini-3-pro-image",
                     "architecture": {"output_modalities": ["image"],
                                      "input_modalities": ["text", "image"]}},
                ]}
            return resp

        with patch(_RUNTIME, return_value=_runtime_ok()), patch("requests.get", side_effect=fake_get):
            ids = [m["id"] for m in orp.list_models()]
        assert ids[0] == "openai/gpt-5.4-image-2"          # default first
        assert "bytedance-seed/seedream-4.5" in ids        # Image-API-only model present
        assert "google/gemini-3-pro-image" in ids          # chat-catalog model present
        assert len(ids) == len(set(ids))                   # deduped

    def test_nous_portal_picker_excludes_image_api_catalog(self):
        """Nous Portal has no /images route; its picker must not offer
        Image-API-only models it cannot serve."""
        from plugins.image_gen.openrouter import _build_providers

        nous = {p.name: p for p in _build_providers()}["nous"]
        with patch(_RUNTIME, side_effect=RuntimeError("no creds")):
            ids = [m["id"] for m in nous.list_models()]
        from plugins.image_gen.openrouter import DEFAULT_MODEL, _FALLBACK_MODEL

        assert ids == [DEFAULT_MODEL, _FALLBACK_MODEL]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_to_image_url_part_passthrough_url(self):
        from plugins.image_gen.openrouter import _to_image_url_part

        assert _to_image_url_part("https://x/y.png") == "https://x/y.png"
        assert _to_image_url_part("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"


    def test_to_image_url_part_blocks_credential_store(self, tmp_path, monkeypatch):
        from plugins.image_gen.openrouter import _to_image_url_part

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        auth_json = hermes_home / "auth.json"
        auth_json.write_text('{"api_key":"sk-secret"}', encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        with pytest.raises(ValueError, match="credential store"):
            _to_image_url_part(str(auth_json))


    def test_extract_images(self):
        from plugins.image_gen.openrouter import _extract_images

        payload = {
            "choices": [
                {"message": {"images": [{"image_url": {"url": "data:image/png;base64,AA"}}]}}
            ]
        }
        assert _extract_images(payload) == ["data:image/png;base64,AA"]


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


# ---------------------------------------------------------------------------
# Registration + pet integration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dedicated Image API surface (POST /images/generations)
# ---------------------------------------------------------------------------


def _openrouter_image_api():
    """The provider as `_build_providers` really configures it (surface on)."""
    from plugins.image_gen.openrouter import _build_providers

    return {p.name: p for p in _build_providers()}["openrouter"]


def _mock_image_api_response(entries=None, usage=None):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    body = {"created": 0, "data": entries if entries is not None else [
        {"b64_json": "dGVzdA==", "media_type": "image/png"}
    ]}
    if usage is not None:
        body["usage"] = usage
    resp.json.return_value = body
    return resp


class TestImageApiSurface:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        """No config bleed, no catalog cache bleed between tests."""
        import plugins.image_gen.openrouter as mod

        mod._CATALOG_CACHE.clear()
        monkeypatch.setattr(mod, "_load_image_gen_config", lambda: {})
        for knob in ("QUALITY", "BACKGROUND", "RESOLUTION", "SEED", "N",
                     "ASPECT_RATIO", "TIMEOUT", "SURFACE"):
            monkeypatch.delenv(f"OPENROUTER_IMAGE_API_{knob}", raising=False)
        monkeypatch.delenv("OPENROUTER_IMAGE_MODEL", raising=False)
        yield
        mod._CATALOG_CACHE.clear()

    # -- routing ---------------------------------------------------------

    def test_curated_model_routes_without_any_probe(self):
        """The static table answers the common case offline."""
        from plugins.image_gen.openrouter import _select_surface

        with patch("requests.get", side_effect=AssertionError("must not probe")):
            assert _select_surface("openai/gpt-image-2", "https://x/api/v1", "k", "openrouter") == "images"

    def test_chat_defaults_stay_on_chat_even_though_the_catalog_lists_them(self):
        """The regression this guards: /images/models is a superset that
        includes DEFAULT_MODEL and _FALLBACK_MODEL. Routing on catalog
        membership would silently move every existing default call."""
        from plugins.image_gen.openrouter import (
            DEFAULT_MODEL,
            _FALLBACK_MODEL,
            _select_surface,
        )

        catalog = MagicMock()
        catalog.raise_for_status = MagicMock()
        catalog.json.return_value = {
            "data": [{"id": DEFAULT_MODEL}, {"id": _FALLBACK_MODEL}]
        }
        with patch("requests.get", return_value=catalog):
            assert _select_surface(DEFAULT_MODEL, "https://x/api/v1", "k", "openrouter") == "chat"
            assert _select_surface(_FALLBACK_MODEL, "https://x/api/v1", "k", "openrouter") == "chat"

    def test_unknown_catalog_model_routes_to_image_api(self):
        """An id past the curated snapshot but in the live catalog is served
        by the dedicated API — a model picked from the live picker must not
        fall onto chat-completions and 404."""
        import plugins.image_gen.openrouter as orp
        from plugins.image_gen.openrouter import _select_surface

        orp._CATALOG_CACHE.clear()
        catalog = MagicMock()
        catalog.raise_for_status = MagicMock()
        catalog.json.return_value = {"data": [{"id": "brandnew/model-9"}]}
        with patch("requests.get", return_value=catalog) as mock_get:
            assert _select_surface("brandnew/model-9", "https://x/api/v1", "k", "openrouter") == "images"
            assert _select_surface("brandnew/model-9", "https://x/api/v1", "k", "openrouter") == "images"
        # Catalog probe is cached — one fetch serves repeat calls.
        assert mock_get.call_count == 1

    def test_unknown_model_not_in_catalog_stays_on_chat(self):
        import plugins.image_gen.openrouter as orp
        from plugins.image_gen.openrouter import _select_surface

        orp._CATALOG_CACHE.clear()
        catalog = MagicMock()
        catalog.raise_for_status = MagicMock()
        catalog.json.return_value = {"data": [{"id": "something/else"}]}
        with patch("requests.get", return_value=catalog):
            assert _select_surface("not-served/model", "https://x/api/v1", "k", "openrouter") == "chat"

    def test_failed_probe_costs_nothing(self):
        import plugins.image_gen.openrouter as orp
        from plugins.image_gen.openrouter import _select_surface

        orp._CATALOG_CACHE.clear()
        with patch("requests.get", side_effect=OSError("network down")):
            assert _select_surface("unknown/model", "https://x/api/v1", "k", "openrouter") == "chat"

    def test_surface_can_be_forced_both_ways(self, monkeypatch):
        from plugins.image_gen.openrouter import DEFAULT_MODEL, _select_surface

        monkeypatch.setenv("OPENROUTER_IMAGE_API_SURFACE", "images")
        with patch("requests.get", side_effect=AssertionError("must not probe")):
            assert _select_surface(DEFAULT_MODEL, "https://x/api/v1", "k", "openrouter") == "images"

        monkeypatch.setenv("OPENROUTER_IMAGE_API_SURFACE", "chat")
        with patch("requests.get", side_effect=AssertionError("must not probe")):
            assert _select_surface("openai/gpt-image-2", "https://x/api/v1", "k", "openrouter") == "chat"

    def test_image_api_model_posts_to_images_generations(self):
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_image_api_response()) as mock_post, \
             patch("plugins.image_gen.openrouter.save_b64_image", return_value=Path("/tmp/i.png")):
            result = _openrouter_image_api().generate(
                prompt="a red square", aspect_ratio="square", model="openai/gpt-image-2"
            )

        assert result["success"] is True
        assert mock_post.call_args[0][0] == "https://openrouter.ai/api/v1/images/generations"
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "openai/gpt-image-2"
        assert payload["prompt"] == "a red square"
        assert payload["aspect_ratio"] == "1:1"
        assert "messages" not in payload and "modalities" not in payload
        assert result["endpoint"] == "images/generations"

    def test_configured_primary_retries_with_fallback(self, monkeypatch):
        import requests as req_lib

        monkeypatch.setattr(
            "plugins.image_gen.openrouter._load_image_gen_config",
            lambda: {
                "model": "google/gemini-3.1-flash-image",
                "fallback_models": ["openai/gpt-5.4-image-2"],
            },
        )
        missing = MagicMock()
        missing.status_code = 404
        missing.json.return_value = {"error": {"message": "model unavailable"}}
        missing.raise_for_status.side_effect = req_lib.HTTPError(response=missing)

        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", side_effect=[missing, _mock_chat_response([_PNG_DATA_URI])]) as mock_post, \
             patch("plugins.image_gen.openrouter.save_b64_image", return_value=Path("/tmp/i.png")):
            result = _openrouter_image_api().generate(prompt="a red square")

        assert result["success"] is True
        assert result["model"] == "openai/gpt-5.4-image-2"
        assert mock_post.call_count == 2
        assert [call.kwargs["json"]["model"] for call in mock_post.call_args_list] == [
            "google/gemini-3.1-flash-image",
            "openai/gpt-5.4-image-2",
        ]

    def test_chat_model_still_uses_chat_completions(self):
        """The new surface must not capture the existing default chain."""
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_chat_response([_PNG_DATA_URI])) as mock_post, \
             patch("plugins.image_gen.openrouter.save_b64_image", return_value=Path("/tmp/x.png")):
            result = _openrouter_image_api().generate(prompt="a pet")

        assert result["success"] is True
        assert mock_post.call_args[0][0].endswith("/chat/completions")

    def test_nous_never_uses_the_image_api(self):
        """Nous Portal proxies chat-completions and has no /images route."""
        from plugins.image_gen.openrouter import _build_providers

        nous_runtime = _runtime_ok(
            provider="nous", base_url="https://inference.nousresearch.com/v1", api_key="nous-tok"
        )
        with patch(_RUNTIME, return_value=nous_runtime), \
             patch("requests.post", return_value=_mock_chat_response([_PNG_DATA_URI])) as mock_post, \
             patch("plugins.image_gen.openrouter.save_b64_image", return_value=Path("/tmp/x.png")):
            nous = {p.name: p for p in _build_providers()}["nous"]
            result = nous.generate(prompt="a pet", model="openai/gpt-image-2")

        assert result["success"] is True
        assert mock_post.call_args[0][0] == "https://inference.nousresearch.com/v1/chat/completions"

    # -- per-model parameter filtering ------------------------------------

    def test_aspect_ratio_is_mapped_per_model(self):
        from plugins.image_gen.openrouter import _build_image_api_payload

        gemini, _ = _build_image_api_payload(
            model_id="google/gemini-3.1-flash-lite-image", prompt="p",
            semantic_aspect="landscape", references=[], config_key="openrouter", kwargs={},
        )
        mini, _ = _build_image_api_payload(
            model_id="openai/gpt-image-1-mini", prompt="p",
            semantic_aspect="landscape", references=[], config_key="openrouter", kwargs={},
        )
        # gpt-image-1-mini has no 16:9 at all, so landscape degrades to 3:2.
        assert gemini["aspect_ratio"] == "16:9"
        assert mini["aspect_ratio"] == "3:2"

    def test_unsupported_parameter_is_dropped_and_explained(self):
        """The endpoint silently ignores unknown fields, so we must filter."""
        from plugins.image_gen.openrouter import _build_image_api_payload

        payload, notes = _build_image_api_payload(
            model_id="openai/gpt-image-2", prompt="p", semantic_aspect="square",
            references=[], config_key="openrouter", kwargs={"background": "transparent"},
        )
        assert "background" not in payload
        assert any("background" in n for n in notes)

        payload, notes = _build_image_api_payload(
            model_id="openai/gpt-image-1-mini", prompt="p", semantic_aspect="square",
            references=[], config_key="openrouter", kwargs={"background": "transparent"},
        )
        assert payload["background"] == "transparent"

    def test_n_is_clamped_to_the_model_cap(self):
        from plugins.image_gen.openrouter import _build_image_api_payload

        payload, notes = _build_image_api_payload(
            model_id="qwen/qwen-image-3-pro", prompt="p", semantic_aspect="square",
            references=[], config_key="openrouter", kwargs={"n": 20},
        )
        assert payload["n"] == 6
        assert any("cap of 6" in n for n in notes)

    def test_unknown_model_omits_the_aspect_ratio(self):
        """An out-of-enum ratio is a hard 400, so never guess one."""
        from plugins.image_gen.openrouter import _build_image_api_payload

        payload, notes = _build_image_api_payload(
            model_id="brandnew/model-9", prompt="p", semantic_aspect="landscape",
            references=[], config_key="openrouter", kwargs={},
        )
        assert "aspect_ratio" not in payload
        assert any("catalog" in n for n in notes)

    def test_env_knob_applies(self, monkeypatch):
        from plugins.image_gen.openrouter import _build_image_api_payload

        monkeypatch.setenv("OPENROUTER_IMAGE_API_QUALITY", "high")
        payload, _ = _build_image_api_payload(
            model_id="openai/gpt-image-2", prompt="p", semantic_aspect="square",
            references=[], config_key="openrouter", kwargs={},
        )
        assert payload["quality"] == "high"

    # -- references --------------------------------------------------------

    def test_references_use_the_per_model_cap(self, tmp_path):
        """Image API models take far more references than chat's 3."""
        refs = []
        for i in range(5):
            p = tmp_path / f"r{i}.png"
            p.write_bytes(b"\x89PNG\r\n")
            refs.append(str(p))

        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_image_api_response()) as mock_post, \
             patch("plugins.image_gen.openrouter.save_b64_image", return_value=Path("/tmp/i.png")):
            result = _openrouter_image_api().generate(
                prompt="edit", model="openai/gpt-image-2", reference_image_urls=refs
            )

        payload = mock_post.call_args.kwargs["json"]
        assert len(payload["input_references"]) == 5      # chat would have clamped to 3
        assert payload["input_references"][0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert result["modality"] == "image"

    def test_unreadable_sole_reference_fails_instead_of_degrading(self):
        """Degrading an edit to text-to-image bills an unrelated picture."""
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post") as mock_post:
            result = _openrouter_image_api().generate(
                prompt="edit this", model="openai/gpt-image-2",
                image_url="/nonexistent/definitely-missing.png",
            )

        assert result["success"] is False
        assert result["error_type"] == "io_error"
        mock_post.assert_not_called()

    # -- response handling -------------------------------------------------

    def test_cost_and_extras_are_surfaced(self):
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_image_api_response(
                 usage={"cost": 0.0336, "total_tokens": 1128})), \
             patch("plugins.image_gen.openrouter.save_b64_image", return_value=Path("/tmp/i.png")):
            result = _openrouter_image_api().generate(
                prompt="p", aspect_ratio="portrait", model="krea/krea-2-medium"
            )

        assert result["cost_usd"] == 0.0336
        assert result["total_tokens"] == 1128
        assert result["exact_aspect_ratio"] == "9:16"
        assert result["image"] == "/tmp/i.png"

    def test_multiple_images_land_in_additional_images(self):
        entries = [
            {"b64_json": "AA==", "media_type": "image/png"},
            {"b64_json": "BB==", "media_type": "image/png"},
        ]
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_image_api_response(entries)), \
             patch("plugins.image_gen.openrouter.save_b64_image",
                   side_effect=[Path("/tmp/a.png"), Path("/tmp/b.png")]):
            result = _openrouter_image_api().generate(prompt="p", model="openai/gpt-image-2")

        assert result["image"] == "/tmp/a.png"
        assert result["additional_images"] == ["/tmp/b.png"]

    def test_empty_data_is_typed(self):
        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=_mock_image_api_response([])):
            result = _openrouter_image_api().generate(prompt="p", model="openai/gpt-image-2")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_zod_validation_error_is_flattened(self):
        from plugins.image_gen.openrouter import _extract_image_api_error

        resp = MagicMock()
        resp.json.return_value = {
            "success": False,
            "error": {
                "name": "ZodError",
                "message": '[{"path":["aspect_ratio"],"message":"Invalid option"}]',
            },
        }
        assert _extract_image_api_error(resp, "fb").startswith("aspect_ratio: Invalid option")

    def test_auth_error_is_not_retried_as_api_error(self):
        import requests as req_lib

        resp = MagicMock()
        resp.status_code = 401
        resp.text = "Unauthorized"
        resp.json.return_value = {"error": {"message": "Invalid API key"}}
        resp.raise_for_status.side_effect = req_lib.HTTPError(response=resp)

        with patch(_RUNTIME, return_value=_runtime_ok()), \
             patch("requests.post", return_value=resp):
            result = _openrouter_image_api().generate(prompt="p", model="openai/gpt-image-2")

        assert result["success"] is False
        assert result["error_type"] == "auth_error"
        assert "_retryable" not in result

    def test_catalog_models_are_offered_only_by_openrouter(self):
        from plugins.image_gen.openrouter import _IMAGE_API_MODELS, _build_providers

        by_name = {p.name: p for p in _build_providers()}
        openrouter_ids = {m["id"] for m in by_name["openrouter"].list_models()}
        nous_ids = {m["id"] for m in by_name["nous"].list_models()}
        assert "openai/gpt-image-2" in openrouter_ids
        assert set(_IMAGE_API_MODELS) <= openrouter_ids
        assert not (set(_IMAGE_API_MODELS) & nous_ids)

    def test_default_model_is_unchanged_by_the_new_surface(self):
        from plugins.image_gen.openrouter import DEFAULT_MODEL

        assert _openrouter_image_api().default_model() == DEFAULT_MODEL


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
