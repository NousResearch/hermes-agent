"""Tests for agent/image_routing.py — the per-turn image input mode decision."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.image_routing import (
    _coerce_capability_bool,
    _coerce_mode,
    _explicit_aux_vision_override,
    _lookup_supports_vision,
    _supports_vision_override,
    _looks_like_vision_model,
    build_native_content_parts,
    decide_image_input_mode,
)


# ─── _coerce_mode ────────────────────────────────────────────────────────────


class TestCoerceMode:
    def test_valid_modes_pass_through(self):
        assert _coerce_mode("auto") == "auto"
        assert _coerce_mode("native") == "native"
        assert _coerce_mode("text") == "text"

    def test_case_insensitive(self):
        assert _coerce_mode("NATIVE") == "native"
        assert _coerce_mode("Auto") == "auto"

    def test_invalid_falls_back_to_auto(self):
        assert _coerce_mode("nonsense") == "auto"
        assert _coerce_mode("") == "auto"
        assert _coerce_mode(None) == "auto"
        assert _coerce_mode(42) == "auto"

    def test_strips_whitespace(self):
        assert _coerce_mode("  native  ") == "native"


# ─── _explicit_aux_vision_override ───────────────────────────────────────────


class TestExplicitAuxVisionOverride:
    def test_none_config(self):
        assert _explicit_aux_vision_override(None) is False

    def test_empty_config(self):
        assert _explicit_aux_vision_override({}) is False

    def test_default_auto_is_not_explicit(self):
        cfg = {"auxiliary": {"vision": {"provider": "auto", "model": "", "base_url": ""}}}
        assert _explicit_aux_vision_override(cfg) is False

    def test_provider_set_is_explicit(self):
        cfg = {"auxiliary": {"vision": {"provider": "openrouter", "model": ""}}}
        assert _explicit_aux_vision_override(cfg) is True

    def test_model_set_is_explicit(self):
        cfg = {"auxiliary": {"vision": {"provider": "auto", "model": "google/gemini-2.5-flash"}}}
        assert _explicit_aux_vision_override(cfg) is True

    def test_base_url_set_is_explicit(self):
        cfg = {"auxiliary": {"vision": {"provider": "auto", "base_url": "http://localhost:11434"}}}
        assert _explicit_aux_vision_override(cfg) is True


class TestLooksLikeVisionModel:
    @pytest.mark.parametrize("model", [
        "mimo-v2-omni",
        "qwen2.5-vl-72b-instruct",
        "claude-3-vision-preview",
        "vendor/model-visual",
    ])
    def test_recognises_vision_slugs(self, model):
        assert _looks_like_vision_model(model) is True

    @pytest.mark.parametrize("model", [
        "gpt-4o-mini",
        "claude-3-haiku",
        "deepseek-chat",
        "",
    ])
    def test_rejects_non_vision_slugs(self, model):
        assert _looks_like_vision_model(model) is False


# ─── decide_image_input_mode ─────────────────────────────────────────────────


class TestDecideImageInputMode:
    def test_explicit_native_overrides_everything(self):
        cfg = {"agent": {"image_input_mode": "native"}}
        # Non-vision model, aux-vision explicitly configured: native still wins.
        cfg["auxiliary"] = {"vision": {"provider": "openrouter", "model": "foo"}}
        with patch("agent.image_routing._lookup_supports_vision", return_value=False):
            assert decide_image_input_mode("openrouter", "some-non-vision-model", cfg) == "native"

    def test_explicit_text_overrides_everything(self):
        cfg = {"agent": {"image_input_mode": "text"}}
        with patch("agent.image_routing._lookup_supports_vision", return_value=True):
            assert decide_image_input_mode("anthropic", "claude-sonnet-4", cfg) == "text"

    def test_auto_with_vision_capable_model(self):
        with patch("agent.image_routing._lookup_supports_vision", return_value=True):
            assert decide_image_input_mode("anthropic", "claude-sonnet-4", {}) == "native"

    def test_auto_with_non_vision_model(self):
        with patch("agent.image_routing._lookup_supports_vision", return_value=False):
            assert decide_image_input_mode("openrouter", "qwen/qwen3-235b", {}) == "text"

    def test_auto_with_unknown_model(self):
        with patch("agent.image_routing._lookup_supports_vision", return_value=None):
            assert decide_image_input_mode("openrouter", "brand-new-slug", {}) == "text"

    def test_auto_respects_aux_vision_override_even_for_vision_model(self):
        """If the user configured a dedicated vision backend, don't bypass it."""
        cfg = {"auxiliary": {"vision": {"provider": "openrouter", "model": "google/gemini-2.5-flash"}}}
        with patch("agent.image_routing._lookup_supports_vision", return_value=True):
            assert decide_image_input_mode("anthropic", "claude-sonnet-4", cfg) == "text"

    def test_none_config_is_auto(self):
        with patch("agent.image_routing._lookup_supports_vision", return_value=True):
            assert decide_image_input_mode("anthropic", "claude-sonnet-4", None) == "native"

    def test_invalid_mode_coerces_to_auto(self):
        cfg = {"agent": {"image_input_mode": "weird-value"}}
        with patch("agent.image_routing._lookup_supports_vision", return_value=True):
            assert decide_image_input_mode("anthropic", "claude-sonnet-4", cfg) == "native"

    def test_auto_uses_text_for_text_only_modalities_even_with_attachment_flag(self):
        registry = {
            "xiaomi": {
                "models": {
                    "mimo-v2.5-pro": {
                        "attachment": True,
                        "modalities": {"input": ["text"]},
                        "tool_call": True,
                    },
                },
            },
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=registry):
            assert decide_image_input_mode("xiaomi", "mimo-v2.5-pro", {}) == "text"


# ─── _coerce_capability_bool ─────────────────────────────────────────────────


class TestCoerceCapabilityBool:
    def test_real_bool_passes_through(self):
        assert _coerce_capability_bool(True) is True
        assert _coerce_capability_bool(False) is False

    def test_int_0_and_1(self):
        assert _coerce_capability_bool(1) is True
        assert _coerce_capability_bool(0) is False

    def test_other_ints_return_none(self):
        assert _coerce_capability_bool(2) is None
        assert _coerce_capability_bool(-1) is None

    def test_yaml_true_tokens(self):
        for s in ("true", "TRUE", "True", "yes", "on", "1", "  true  "):
            assert _coerce_capability_bool(s) is True

    def test_yaml_false_tokens(self):
        for s in ("false", "FALSE", "False", "no", "off", "0", "  false  "):
            assert _coerce_capability_bool(s) is False

    def test_quoted_false_does_not_silently_become_true(self):
        # Regression: bool("false") is True in Python. A user writing
        # supports_vision: "false" must NOT enable native vision routing.
        assert _coerce_capability_bool("false") is False

    def test_unrecognised_strings_return_none(self):
        # None == fall through to models.dev, not a silent truthy.
        assert _coerce_capability_bool("maybe") is None
        assert _coerce_capability_bool("") is None
        assert _coerce_capability_bool("definitely") is None

    def test_other_types_return_none(self):
        assert _coerce_capability_bool(None) is None
        assert _coerce_capability_bool([]) is None
        assert _coerce_capability_bool({}) is None
        assert _coerce_capability_bool(1.5) is None


# ─── _supports_vision_override ───────────────────────────────────────────────


class TestSupportsVisionOverride:
    def test_no_cfg_returns_none(self):
        assert _supports_vision_override(None, "custom", "my-llava") is None
        assert _supports_vision_override({}, "custom", "my-llava") is None

    def test_top_level_shortcut_wins(self):
        cfg = {"model": {"supports_vision": True}}
        assert _supports_vision_override(cfg, "custom", "my-llava") is True

    def test_top_level_false_propagates(self):
        cfg = {"model": {"supports_vision": False}}
        assert _supports_vision_override(cfg, "custom", "my-llava") is False

    def test_per_provider_per_model_via_runtime_name(self):
        cfg = {
            "providers": {
                "custom": {"models": {"my-llava": {"supports_vision": True}}},
            },
        }
        assert _supports_vision_override(cfg, "custom", "my-llava") is True

    def test_per_provider_per_model_via_config_name(self):
        # Named custom provider — runtime self.provider == "custom", config
        # holds the original name under model.provider.
        cfg = {
            "model": {"provider": "my-vllm"},
            "providers": {
                "my-vllm": {"models": {"my-llava": {"supports_vision": True}}},
            },
        }
        assert _supports_vision_override(cfg, "custom", "my-llava") is True

    def test_quoted_false_string_in_yaml_does_not_enable(self):
        # Real-world: user writes supports_vision: "false" (quoted).
        cfg = {"model": {"supports_vision": "false"}}
        assert _supports_vision_override(cfg, "custom", "my-llava") is False

    def test_unrecognised_value_falls_through(self):
        cfg = {"model": {"supports_vision": "maybe"}}
        assert _supports_vision_override(cfg, "custom", "my-llava") is None

    def test_no_override_returns_none(self):
        cfg = {"model": {"default": "my-llava"}}
        assert _supports_vision_override(cfg, "custom", "my-llava") is None

    def test_malformed_sections_are_ignored(self):
        # User accidentally wrote a string where a section was expected —
        # don't blow up, just fall through.
        cfg = {"model": "some-string", "providers": ["not-a-dict"]}
        assert _supports_vision_override(cfg, "custom", "my-llava") is None


# ─── _lookup_supports_vision (override-aware) ────────────────────────────────


class TestLookupSupportsVisionOverride:
    def test_config_override_short_circuits_models_dev(self):
        # Config says True, models.dev says None — config wins.
        cfg = {"model": {"supports_vision": True}}
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert _lookup_supports_vision("custom", "my-llava", cfg) is True

    def test_config_override_false_beats_vision_capable_models_dev(self):
        # User explicitly disables vision on a models.dev-vision-capable model.
        fake_caps = type("Caps", (), {"supports_vision": True})()
        cfg = {"model": {"supports_vision": False}}
        with patch("agent.models_dev.get_model_capabilities", return_value=fake_caps):
            assert _lookup_supports_vision("anthropic", "claude-sonnet-4", cfg) is False

    def test_no_override_falls_back_to_models_dev(self):
        fake_caps = type("Caps", (), {"supports_vision": True})()
        with patch("agent.models_dev.get_model_capabilities", return_value=fake_caps):
            assert _lookup_supports_vision("anthropic", "claude-sonnet-4", {}) is True

    def test_no_override_no_models_dev_entry_returns_none(self):
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert _lookup_supports_vision("custom", "my-llava", {}) is None

    def test_cfg_none_falls_back_to_models_dev(self):
        # Caller didn't pass cfg at all — old call sites must still work.
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert _lookup_supports_vision("openrouter", "x", None) is None


# ─── decide_image_input_mode with auto + override ────────────────────────────


class TestAutoModeRespectsOverride:
    def test_auto_native_for_custom_with_supports_vision_true(self):
        # The motivating bug: Qwen3.6 on local llama.cpp via provider=custom.
        # Without the override, auto falls back to text. With it, auto picks
        # native — no need to also set agent.image_input_mode: native.
        cfg = {"model": {"supports_vision": True}}
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert decide_image_input_mode("custom", "qwen3.6-35b", cfg) == "native"

    def test_auto_text_for_custom_with_supports_vision_false(self):
        cfg = {"model": {"supports_vision": False}}
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert decide_image_input_mode("custom", "some-text-only", cfg) == "text"

    def test_auto_text_for_custom_with_no_override(self):
        # Unchanged baseline: unknown custom model → text.
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert decide_image_input_mode("custom", "unknown", {}) == "text"

    def test_explicit_aux_vision_override_still_wins(self):
        # If the user has configured a dedicated vision aux backend, respect
        # it even when supports_vision: true is also set.
        cfg = {
            "model": {"supports_vision": True},
            "auxiliary": {"vision": {"provider": "openrouter", "model": "gemini-2.5-pro"}},
        }
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert decide_image_input_mode("custom", "qwen3.6-35b", cfg) == "text"


# ─── build_native_content_parts ──────────────────────────────────────────────


def _png_bytes() -> bytes:
    """Return a tiny valid 1x1 transparent PNG."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=="
    )


    @pytest.mark.parametrize("model", [
        "mimo-v2-omni",
        "qwen2.5-vl-72b-instruct",
        "claude-3-vision-preview",
    ])
    def test_auto_with_unknown_vision_shaped_slug_falls_back_to_native(self, model, monkeypatch):
        monkeypatch.setattr("agent.image_routing._lookup_supports_vision", lambda provider, model, cfg=None: None)
        assert decide_image_input_mode("custom", model, {}) == "native"

    def test_auto_known_non_vision_still_text_even_if_slug_looks_visiony(self, monkeypatch):
        monkeypatch.setattr("agent.image_routing._lookup_supports_vision", lambda provider, model, cfg=None: False)
        assert decide_image_input_mode("custom", "fake-vision-model", {}) == "text"


class TestBuildNativeContentParts:
    def test_text_then_image(self, tmp_path: Path):
        img = tmp_path / "cat.png"
        img.write_bytes(_png_bytes())
        parts, skipped = build_native_content_parts("hello", [str(img)])
        assert skipped == []
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        # User caption is preserved and a per-image path hint is appended so
        # the model can use the local path as a string argument for tools
        # that take ``image_url: str`` (issue #18960).
        assert parts[0]["text"] == f"hello\n\n[Image attached at: {img}]"
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_empty_text_inserts_default_prompt(self, tmp_path: Path):
        img = tmp_path / "cat.jpg"
        img.write_bytes(_png_bytes())
        parts, skipped = build_native_content_parts("", [str(img)])
        assert skipped == []
        # Even with empty user text, we insert a neutral prompt so the turn
        # isn't just pixels, and the path hint is appended after.
        assert parts[0]["type"] == "text"
        assert parts[0]["text"] == (
            f"What do you see in this image?\n\n[Image attached at: {img}]"
        )
        assert parts[1]["type"] == "image_url"

    def test_missing_file_is_skipped(self, tmp_path: Path):
        parts, skipped = build_native_content_parts("hi", [str(tmp_path / "missing.png")])
        assert skipped == [str(tmp_path / "missing.png")]
        # Skipped paths are NOT advertised in the path hints — the model
        # would otherwise be told a non-existent file is attached.
        assert parts == [{"type": "text", "text": "hi"}]

    def test_path_hint_appended(self, tmp_path: Path):
        """The local path of each attached image is appended to the user
        text part so MCP/skill tools that take ``image_url: str`` can be
        invoked on the same image (issue #18960). Mirrors text-mode
        behaviour (`Runner._enrich_message_with_vision`).
        """
        img = tmp_path / "scan.png"
        img.write_bytes(_png_bytes())
        parts, _ = build_native_content_parts("attach this", [str(img)])
        text_part = next(p for p in parts if p.get("type") == "text")
        assert "[Image attached at:" in text_part["text"]
        assert str(img) in text_part["text"]
        # User caption is preserved verbatim ahead of the hint.
        assert text_part["text"].startswith("attach this")

    def test_path_hint_one_per_attached_image(self, tmp_path: Path):
        """Each successfully attached image gets its own path hint line;
        skipped images do NOT appear in the hints.
        """
        good = tmp_path / "good.png"
        good.write_bytes(_png_bytes())
        missing = tmp_path / "missing.png"  # never created
        parts, skipped = build_native_content_parts(
            "see attached", [str(good), str(missing)]
        )
        assert skipped == [str(missing)]
        text_part = next(p for p in parts if p.get("type") == "text")
        assert text_part["text"].count("[Image attached at:") == 1
        assert str(good) in text_part["text"]
        assert str(missing) not in text_part["text"]

    def test_multiple_images(self, tmp_path: Path):
        img1 = tmp_path / "a.png"
        img2 = tmp_path / "b.png"
        img1.write_bytes(_png_bytes())
        img2.write_bytes(_png_bytes())
        parts, skipped = build_native_content_parts("compare these", [str(img1), str(img2)])
        assert skipped == []
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == 2
        # Both paths surface in the text part, one per line.
        text_part = next(p for p in parts if p.get("type") == "text")
        assert text_part["text"].count("[Image attached at:") == 2
        assert str(img1) in text_part["text"]
        assert str(img2) in text_part["text"]

    def test_mime_inference_jpg(self, tmp_path: Path):
        # Real JPEG bytes (SOI marker FF D8 FF): sniffing now wins over suffix.
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01" + b"\x00" * 32)
        parts, _ = build_native_content_parts("x", [str(img)])
        url = parts[1]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    def test_mime_inference_webp(self, tmp_path: Path):
        # Real WEBP bytes (RIFF....WEBP): sniffing now wins over suffix.
        img = tmp_path / "pic.webp"
        img.write_bytes(b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 32)
        parts, _ = build_native_content_parts("", [str(img)])
        url = parts[1]["image_url"]["url"]
        assert url.startswith("data:image/webp;base64,")

    def test_mime_sniff_overrides_misleading_extension(self, tmp_path: Path):
        """Discord-style bug: file is named .webp but contains PNG bytes.
        Anthropic rejects on MIME mismatch (HTTP 400) so we MUST sniff.
        Regression guard for the user-reported Discord PNG-as-WEBP failure.
        """
        img = tmp_path / "discord_cached.webp"
        img.write_bytes(_png_bytes())  # bytes are PNG, suffix lies
        parts, _ = build_native_content_parts("", [str(img)])
        url = parts[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,"), (
            f"Expected MIME sniffing to detect PNG bytes regardless of .webp suffix, got: {url[:60]}"
        )


# ─── Oversize handling ───────────────────────────────────────────────────────


class TestLargeImageHandling:
    """Large images attach at native size; shrink is handled reactively at
    retry time in ``run_agent._try_shrink_image_parts_in_messages`` rather
    than proactively here.
    """

    def test_large_image_passes_through_unchanged(self, tmp_path: Path):
        """A multi-MB image is attached as-is — no resize, no skip."""
        from agent import image_routing as _ir

        img = tmp_path / "medium.png"
        # 200 KB of real bytes; not huge but enough to verify no size gate fires.
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"X" * 200_000)
        url = _ir._file_to_data_url(img)
        assert url is not None
        assert url.startswith("data:image/png;base64,")
        # Base64 expansion means output is ~4/3 of input, plus header.
        assert len(url) > 200_000

    def test_missing_file_returns_none(self, tmp_path: Path):
        from agent import image_routing as _ir
        missing = tmp_path / "does_not_exist.png"
        assert _ir._file_to_data_url(missing) is None

    def test_build_native_parts_no_provider_kwarg(self, tmp_path: Path):
        """build_native_content_parts takes text + paths, no provider kwarg."""
        from agent import image_routing as _ir

        img = tmp_path / "cat.png"
        img.write_bytes(_png_bytes())
        parts, skipped = _ir.build_native_content_parts("hi", [str(img)])
        assert skipped == []
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image_url"
