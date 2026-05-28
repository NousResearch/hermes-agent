#!/usr/bin/env python3
"""Tests for the FAL.ai image generation plugin.

The plugin is a thin registration adapter — actual FAL pipeline logic
lives in ``tools.image_generation_tool`` and is exercised by
``tests/tools/test_image_generation.py``. These tests focus on:

* the ``ImageGenProvider`` ABC surface (name, models, schema)
* call-time indirection (``_it`` resolution at ``generate()`` time so
  ``monkeypatch.setattr(image_tool, ...)`` keeps working)
* response shape stamping (provider/prompt/aspect_ratio/model)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock



# ---------------------------------------------------------------------------
# Provider surface
# ---------------------------------------------------------------------------


class TestFalImageGenProviderSurface:
    def test_name(self):
        from plugins.image_gen.fal import FalImageGenProvider

        assert FalImageGenProvider().name == "fal"

    def test_display_name(self):
        from plugins.image_gen.fal import FalImageGenProvider

        assert FalImageGenProvider().display_name == "FAL.ai"

    def test_default_model_matches_legacy(self):
        from plugins.image_gen.fal import FalImageGenProvider
        from tools.image_generation_tool import DEFAULT_MODEL

        assert FalImageGenProvider().default_model() == DEFAULT_MODEL

    def test_list_models_uses_legacy_catalog(self):
        from plugins.image_gen.fal import FalImageGenProvider
        from tools.image_generation_tool import FAL_MODELS

        provider = FalImageGenProvider()
        models = provider.list_models()
        ids = {m["id"] for m in models}
        # Whatever FAL_MODELS ships, the provider mirrors verbatim.
        assert ids == set(FAL_MODELS.keys())
        # Spot-check the expected first-class fields are present.
        for entry in models:
            for field in ("id", "display", "speed", "strengths", "price"):
                assert field in entry

    def test_setup_schema_advertises_fal_key(self):
        from plugins.image_gen.fal import FalImageGenProvider

        schema = FalImageGenProvider().get_setup_schema()
        assert schema["name"] == "FAL.ai"
        assert schema["badge"] == "paid"
        env_keys = {entry["key"] for entry in schema.get("env_vars", [])}
        assert "FAL_KEY" in env_keys


class TestFalImageGenProviderAvailability:
    def test_is_available_when_legacy_check_passes(self, monkeypatch):
        import tools.image_generation_tool as image_tool
        from plugins.image_gen.fal import FalImageGenProvider

        monkeypatch.setattr(image_tool, "check_fal_api_key", lambda: True)
        assert FalImageGenProvider().is_available() is True

    def test_is_available_false_when_legacy_check_fails(self, monkeypatch):
        import tools.image_generation_tool as image_tool
        from plugins.image_gen.fal import FalImageGenProvider

        monkeypatch.setattr(image_tool, "check_fal_api_key", lambda: False)
        assert FalImageGenProvider().is_available() is False

    def test_is_available_handles_legacy_exception(self, monkeypatch):
        import tools.image_generation_tool as image_tool
        from plugins.image_gen.fal import FalImageGenProvider

        def _boom():
            raise RuntimeError("config broke")

        monkeypatch.setattr(image_tool, "check_fal_api_key", _boom)
        # Picker must not propagate exceptions — show as "not available".
        assert FalImageGenProvider().is_available() is False


# ---------------------------------------------------------------------------
# generate() — call-time indirection
# ---------------------------------------------------------------------------


class TestFalImageGenProviderGenerate:
    def test_generate_delegates_to_legacy_image_generate_tool(self, monkeypatch):
        """Plugin must look up ``image_generate_tool`` at call time so
        ``monkeypatch.setattr(image_tool, "image_generate_tool", ...)``
        takes effect."""
        import tools.image_generation_tool as image_tool
        from plugins.image_gen.fal import FalImageGenProvider

        captured = {}

        def fake_image_generate_tool(prompt, aspect_ratio, **kwargs):
            captured["prompt"] = prompt
            captured["aspect_ratio"] = aspect_ratio
            captured["kwargs"] = kwargs
            return json.dumps({"success": True, "image": "https://fake/image.png"})

        monkeypatch.setattr(image_tool, "image_generate_tool", fake_image_generate_tool)
        monkeypatch.setattr(image_tool, "_resolve_fal_model",
                            lambda: ("fal-ai/flux-2/klein/9b", {}))

        result = FalImageGenProvider().generate(
            "a serene mountain landscape",
            aspect_ratio="square",
            seed=42,
        )

        assert captured["prompt"] == "a serene mountain landscape"
        assert captured["aspect_ratio"] == "square"
        assert captured["kwargs"] == {"seed": 42}
        assert result["success"] is True
        assert result["image"] == "https://fake/image.png"
        # Stamped fields for the unified response shape
        assert result["provider"] == "fal"
        assert result["prompt"] == "a serene mountain landscape"
        assert result["aspect_ratio"] == "square"
        assert result["model"] == "fal-ai/flux-2/klein/9b"

    def test_generate_invalid_aspect_ratio_is_coerced(self, monkeypatch):
        import tools.image_generation_tool as image_tool
        from plugins.image_gen.fal import FalImageGenProvider

        seen_aspect = {}

        def fake(prompt, aspect_ratio, **kwargs):
            seen_aspect["v"] = aspect_ratio
            return json.dumps({"success": True, "image": "x"})

        monkeypatch.setattr(image_tool, "image_generate_tool", fake)
        monkeypatch.setattr(image_tool, "_resolve_fal_model",
                            lambda: ("fal-ai/flux-2/klein/9b", {}))

        FalImageGenProvider().generate("p", aspect_ratio="not-a-real-ratio")
        # ``resolve_aspect_ratio`` clamps to landscape.
        assert seen_aspect["v"] == "landscape"

    def test_generate_passthrough_drops_none_kwargs(self, monkeypatch):
        import tools.image_generation_tool as image_tool
        from plugins.image_gen.fal import FalImageGenProvider

        seen = {}

        def fake(prompt, aspect_ratio, **kwargs):
            seen.update(kwargs)
            return json.dumps({"success": True, "image": "x"})

        monkeypatch.setattr(image_tool, "image_generate_tool", fake)
        monkeypatch.setattr(image_tool, "_resolve_fal_model",
                            lambda: ("fal-ai/flux-2/klein/9b", {}))

        FalImageGenProvider().generate(
            "p",
            aspect_ratio="landscape",
            seed=None,
            num_images=2,
            guidance_scale=None,
        )

        # ``None`` values must not be forwarded — they'd override the
        # model's defaults inside the legacy payload builder.
        assert "seed" not in seen
        assert "guidance_scale" not in seen
        assert seen.get("num_images") == 2

    def test_generate_catches_exception_from_legacy(self, monkeypatch):
        import tools.image_generation_tool as image_tool
        from plugins.image_gen.fal import FalImageGenProvider

        def boom(*args, **kwargs):
            raise RuntimeError("FAL endpoint exploded")

        monkeypatch.setattr(image_tool, "image_generate_tool", boom)

        result = FalImageGenProvider().generate("p")
        assert result["success"] is False
        assert "FAL image generation failed" in result["error"]
        assert result["error_type"] == "RuntimeError"
        assert result["provider"] == "fal"

    def test_generate_invalid_json_response(self, monkeypatch):
        import tools.image_generation_tool as image_tool
        from plugins.image_gen.fal import FalImageGenProvider

        monkeypatch.setattr(image_tool, "image_generate_tool", lambda **kw: "not-json")
        monkeypatch.setattr(image_tool, "_resolve_fal_model",
                            lambda: ("fal-ai/flux-2/klein/9b", {}))

        result = FalImageGenProvider().generate("p")
        assert result["success"] is False
        assert "Invalid JSON" in result["error"]
        assert result["provider"] == "fal"


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestFalImageGenPluginRegistration:
    def test_register_wires_provider_into_registry(self):
        from plugins.image_gen.fal import FalImageGenProvider, register

        ctx = MagicMock()
        register(ctx)

        ctx.register_image_gen_provider.assert_called_once()
        (registered,), _ = ctx.register_image_gen_provider.call_args
        assert isinstance(registered, FalImageGenProvider)
