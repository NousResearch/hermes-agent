"""Tests for agent/image_routing.py — the per-turn image input mode decision."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.image_routing import (
    _coerce_mode,
    _explicit_aux_vision_override,
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


# ─── build_native_content_parts ──────────────────────────────────────────────


def _png_bytes() -> bytes:
    """Return a tiny valid 1x1 transparent PNG."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=="
    )


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


# --- Format compatibility (transcode-to-PNG for non-universal formats) ------


class TestFormatCompatibility:
    """Some image formats Discord (and other chat platforms) accept aren't
    accepted by every major vision provider. Anthropic for example returns
    HTTP 400 'Could not process image' for AVIF/HEIC/BMP/TIFF/ICO/SVG.

    We transcode anything outside the universal-safe set (PNG/JPEG/GIF/WEBP)
    to PNG with Pillow before declaring media_type so the provider call
    actually succeeds. Regression coverage for the user-reported Discord
    'Could not process image' HTTP 400.
    """

    def test_avif_sniffed_correctly(self):
        from agent.image_routing import _sniff_mime_from_bytes
        # Minimal ftyp box with major brand 'avif'
        avif_header = b"\x00\x00\x00\x20ftypavif\x00\x00\x00\x00"
        assert _sniff_mime_from_bytes(avif_header) == "image/avif"

    def test_tiff_sniffed_both_endians(self):
        from agent.image_routing import _sniff_mime_from_bytes
        assert _sniff_mime_from_bytes(b"II*\x00" + b"\x00" * 16) == "image/tiff"
        assert _sniff_mime_from_bytes(b"MM\x00*" + b"\x00" * 16) == "image/tiff"

    def test_ico_sniffed_correctly(self):
        from agent.image_routing import _sniff_mime_from_bytes
        assert _sniff_mime_from_bytes(b"\x00\x00\x01\x00" + b"\x00" * 16) == "image/x-icon"

    def test_svg_sniffed_correctly(self):
        from agent.image_routing import _sniff_mime_from_bytes
        assert _sniff_mime_from_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"/>') == "image/svg+xml"
        assert _sniff_mime_from_bytes(b'<?xml version="1.0"?><svg/>') == "image/svg+xml"

    def test_bmp_transcoded_to_png(self, tmp_path: Path):
        """BMP file should land as image/png in the data URL, not image/bmp,
        because not every provider (Anthropic in particular) accepts BMP.
        """
        Image = pytest.importorskip("PIL.Image", reason="Pillow not installed; transcode is best-effort")
        from agent.image_routing import _file_to_data_url

        img_path = tmp_path / "scan.bmp"
        Image.new("RGB", (4, 4), (255, 0, 0)).save(img_path, format="BMP")
        url = _file_to_data_url(img_path)
        assert url is not None
        assert url.startswith("data:image/png;base64,"), (
            f"BMP must be transcoded to PNG for cross-provider compatibility, got: {url[:60]}"
        )

    def test_tiff_transcoded_to_png(self, tmp_path: Path):
        Image = pytest.importorskip("PIL.Image", reason="Pillow not installed; transcode is best-effort")
        from agent.image_routing import _file_to_data_url

        img_path = tmp_path / "scan.tiff"
        Image.new("RGB", (4, 4), (0, 255, 0)).save(img_path, format="TIFF")
        url = _file_to_data_url(img_path)
        assert url is not None
        assert url.startswith("data:image/png;base64,")

    def test_png_passes_through_no_transcode(self, tmp_path: Path):
        """Universal-safe formats must NOT be re-encoded -- preserves bytes."""
        from agent.image_routing import _file_to_data_url

        img_path = tmp_path / "ok.png"
        img_path.write_bytes(_png_bytes())
        url = _file_to_data_url(img_path)
        assert url is not None
        assert url.startswith("data:image/png;base64,")
        # The data segment should be exactly the b64 of the original bytes,
        # i.e. transcoding was correctly skipped.
        import base64
        b64 = url.split(",", 1)[1]
        assert base64.b64decode(b64) == _png_bytes()

    def test_jpeg_passes_through_no_transcode(self, tmp_path: Path):
        from agent.image_routing import _file_to_data_url

        img_path = tmp_path / "ok.jpg"
        # Tiny but real JPEG
        img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
        url = _file_to_data_url(img_path)
        assert url is not None
        assert url.startswith("data:image/jpeg;base64,")

    def test_transcode_failure_is_skipped_not_crashed(self, tmp_path: Path):
        """If Pillow can't decode (corrupted bytes labeled as a rare format),
        return None so the caller adds the path to ``skipped`` rather than
        sending broken data to the provider.
        """
        from agent.image_routing import _file_to_data_url

        # AVIF magic prefix but nonsense payload Pillow can't decode
        img_path = tmp_path / "corrupt.avif"
        img_path.write_bytes(b"\x00\x00\x00\x20ftypavif" + b"\x00" * 32)
        url = _file_to_data_url(img_path)
        assert url is None

    def test_unsupported_format_skipped_in_native_parts(self, tmp_path: Path):
        """End-to-end: a file whose format we can't decode at all should
        show up in `skipped` from `build_native_content_parts`, not crash
        and not silently send broken bytes.
        """
        from agent.image_routing import build_native_content_parts

        # AVIF magic prefix but nonsense payload
        bad = tmp_path / "broken.avif"
        bad.write_bytes(b"\x00\x00\x00\x20ftypavif" + b"\x00" * 32)
        # And one good PNG alongside
        good = tmp_path / "good.png"
        good.write_bytes(_png_bytes())

        parts, skipped = build_native_content_parts("look", [str(bad), str(good)])
        assert str(bad) in skipped
        # Good image still attached; rest of turn proceeds.
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == 1
