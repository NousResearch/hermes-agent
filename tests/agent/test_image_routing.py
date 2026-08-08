"""Tests for agent/image_routing.py — the per-turn image input mode decision."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from pathlib import Path
from unittest.mock import patch


from agent.image_routing import (
    _coerce_capability_bool,
    _coerce_mode,
    _explicit_aux_vision_override,
    _lookup_supports_vision,
    _supports_vision_override,
    build_native_content_parts,
    decide_image_input_mode,
    extract_image_refs,
)


# ─── _coerce_mode ────────────────────────────────────────────────────────────


class TestCoerceMode:

    def test_case_insensitive(self):
        assert _coerce_mode("NATIVE") == "native"
        assert _coerce_mode("Auto") == "auto"

    def test_invalid_falls_back_to_auto(self):
        assert _coerce_mode("nonsense") == "auto"
        assert _coerce_mode("") == "auto"
        assert _coerce_mode(None) == "auto"
        assert _coerce_mode(42) == "auto"



# ─── _explicit_aux_vision_override ───────────────────────────────────────────


class TestExplicitAuxVisionOverride:
    def test_none_config(self):
        assert _explicit_aux_vision_override(None) is False

    def test_empty_config(self):
        assert _explicit_aux_vision_override({}) is False






# ─── decide_image_input_mode ─────────────────────────────────────────────────


class TestDecideImageInputMode:




    def test_auto_with_unknown_model(self):
        with patch("agent.image_routing._lookup_supports_vision", return_value=None):
            assert decide_image_input_mode("openrouter", "brand-new-slug", {}) == "text"

    def test_auto_prefers_native_for_vision_capable_main_model_even_with_aux_configured(self):
        """Regression #29135: vision-capable main model wins over aux fallback.

        Auxiliary.vision is a fallback for text-only main models; it must
        not preempt native vision on a vision-capable main model.
        """
        cfg = {"auxiliary": {"vision": {"provider": "openrouter", "model": "google/gemini-2.5-flash"}}}
        with patch("agent.image_routing._lookup_supports_vision", return_value=True):
            assert decide_image_input_mode("anthropic", "claude-sonnet-4", cfg) == "native"


    def test_none_config_is_auto(self):
        with patch("agent.image_routing._lookup_supports_vision", return_value=True):
            assert decide_image_input_mode("anthropic", "claude-sonnet-4", None) == "native"




# ─── _coerce_capability_bool ─────────────────────────────────────────────────


class TestCoerceCapabilityBool:
    def test_real_bool_passes_through(self):
        assert _coerce_capability_bool(True) is True
        assert _coerce_capability_bool(False) is False

    def test_int_0_and_1(self):
        assert _coerce_capability_bool(1) is True
        assert _coerce_capability_bool(0) is False






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












# ─── _lookup_supports_vision (override-aware) ────────────────────────────────


class TestLookupSupportsVisionOverride:


    def test_no_override_falls_back_to_models_dev(self):
        fake_caps = type("Caps", (), {"supports_vision": True})()
        with patch("agent.models_dev.get_model_capabilities", return_value=fake_caps):
            assert _lookup_supports_vision("anthropic", "claude-sonnet-4", {}) is True


    def test_ollama_probe_when_models_dev_missing(self):
        cfg = {"model": {"base_url": "http://localhost:11434/v1"}}
        with patch("agent.models_dev.get_model_capabilities", return_value=None), \
             patch("agent.image_routing._should_probe_ollama_vision", return_value=True), \
             patch("agent.model_metadata.query_ollama_supports_vision", return_value=True):
            assert _lookup_supports_vision("ollama", "gemma4:e2b", cfg) is True


    def test_cfg_none_falls_back_to_models_dev(self):
        # Caller didn't pass cfg at all — old call sites must still work.
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert _lookup_supports_vision("openrouter", "x", None) is None


# ─── decide_image_input_mode with auto + override ────────────────────────────


class TestAutoModeRespectsOverride:


    def test_auto_text_for_custom_with_supports_vision_false(self):
        cfg = {"model": {"supports_vision": False}}
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert decide_image_input_mode("custom", "some-text-only", cfg) == "text"

    def test_auto_text_for_custom_with_no_override(self):
        # Unchanged baseline: unknown custom model → text.
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert decide_image_input_mode("custom", "unknown", {}) == "text"




# ─── build_native_content_parts ──────────────────────────────────────────────


def _png_bytes() -> bytes:
    """Return a tiny valid 1x1 transparent PNG."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=="
    )


def _jpeg_bytes(size: tuple[int, int] = (4, 3), quality: int = 90) -> bytes:
    """Return a valid synthetic JPEG."""
    from PIL import Image

    image = Image.new("RGB", size, (20, 80, 140))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _noisy_jpeg_bytes(size: tuple[int, int] = (1200, 900)) -> bytes:
    """Return a valid, poorly compressible JPEG for payload-limit tests."""
    from PIL import Image

    image = Image.effect_noise(size, 100).convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class TestBuildNativeContentParts:
    def test_small_png_and_jpeg_preserve_exact_bytes_and_mime(self, tmp_path: Path):
        fixtures = (
            ("small.png", _png_bytes(), "image/png"),
            ("small.jpg", _jpeg_bytes(), "image/jpeg"),
        )

        for filename, expected_bytes, expected_mime in fixtures:
            image_path = tmp_path / filename
            image_path.write_bytes(expected_bytes)
            parts, skipped = build_native_content_parts("inspect", [str(image_path)])

            assert skipped == []
            data_url = parts[1]["image_url"]["url"]
            header, encoded = data_url.split(",", 1)
            assert header == f"data:{expected_mime};base64"
            assert base64.b64decode(encoded) == expected_bytes

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





# ─── Oversize handling ───────────────────────────────────────────────────────


class TestLargeImageHandling:
    """Native attachments are constrained before entering the user message."""

    def test_byte_safe_image_passes_through_unchanged(self, tmp_path: Path):
        """An image below both limits retains its historical pass-through."""
        from agent import image_routing as _ir

        img = tmp_path / "medium.png"
        # 200 KB of real bytes; not huge but enough to verify no size gate fires.
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"X" * 200_000)
        url = _ir._file_to_data_url(img)
        assert url is not None
        assert url.startswith("data:image/png;base64,")
        # Base64 expansion means output is ~4/3 of input, plus header.
        assert len(url) > 200_000

    def test_oversized_jpeg_is_constrained_and_source_is_unchanged(
        self, tmp_path: Path
    ):
        from PIL import Image

        from agent import image_routing as _ir

        source_bytes = _noisy_jpeg_bytes()
        source_checksum = hashlib.sha256(source_bytes).hexdigest()
        image_path = tmp_path / "oversized.jpg"
        image_path.write_bytes(source_bytes)
        config = {
            "agent": {
                "native_image_max_payload_bytes": 220_000,
                "native_image_max_dimension": 600,
            }
        }

        with patch("hermes_cli.config.load_config_readonly", return_value=config):
            data_url = _ir._file_to_data_url(image_path)

        assert data_url is not None
        assert len(data_url.encode("ascii")) <= 220_000
        assert data_url.startswith("data:image/jpeg;base64,")
        output_bytes = base64.b64decode(data_url.split(",", 1)[1])
        with Image.open(BytesIO(output_bytes)) as output:
            assert output.format == "JPEG"
            assert max(output.size) <= 600
            assert abs((output.width / output.height) - (1200 / 900)) < 0.01
        assert image_path.read_bytes() == source_bytes
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == source_checksum

    def test_config_override_changes_preprocessing_limit(self, tmp_path: Path):
        from agent import image_routing as _ir

        source_bytes = _noisy_jpeg_bytes((800, 600))
        image_path = tmp_path / "override.jpg"
        image_path.write_bytes(source_bytes)

        high_limit = {
            "agent": {
                "native_image_max_payload_bytes": 2_000_000,
                "native_image_max_dimension": 2000,
            }
        }
        low_limit = {
            "agent": {
                "native_image_max_payload_bytes": 150_000,
                "native_image_max_dimension": 500,
            }
        }
        with patch("hermes_cli.config.load_config_readonly", return_value=high_limit):
            unchanged_url = _ir._file_to_data_url(image_path)
        with patch("hermes_cli.config.load_config_readonly", return_value=low_limit):
            constrained_url = _ir._file_to_data_url(image_path)

        assert unchanged_url is not None
        assert constrained_url is not None
        assert base64.b64decode(unchanged_url.split(",", 1)[1]) == source_bytes
        assert len(constrained_url.encode("ascii")) <= 150_000
        assert base64.b64decode(constrained_url.split(",", 1)[1]) != source_bytes

    def test_malformed_config_values_fall_back_to_safe_defaults(self):
        from agent.image_payloads import (
            DEFAULT_NATIVE_IMAGE_MAX_DIMENSION,
            DEFAULT_NATIVE_IMAGE_MAX_PAYLOAD_BYTES,
            get_native_image_limits,
        )
        from hermes_cli.config import DEFAULT_CONFIG

        # The loader fallback and the user-visible config defaults must stay in
        # lockstep even if either setting changes later.
        assert DEFAULT_CONFIG["agent"]["native_image_max_payload_bytes"] == (
            DEFAULT_NATIVE_IMAGE_MAX_PAYLOAD_BYTES
        )
        assert DEFAULT_CONFIG["agent"]["native_image_max_dimension"] == (
            DEFAULT_NATIVE_IMAGE_MAX_DIMENSION
        )

        malformed = {
            "agent": {
                "native_image_max_payload_bytes": "many",
                "native_image_max_dimension": -5,
            }
        }
        with patch("hermes_cli.config.load_config_readonly", return_value=malformed):
            assert get_native_image_limits() == (
                DEFAULT_NATIVE_IMAGE_MAX_PAYLOAD_BYTES,
                DEFAULT_NATIVE_IMAGE_MAX_DIMENSION,
            )

        string_override = {
            "agent": {
                "native_image_max_payload_bytes": "8388608",
                "native_image_max_dimension": "9000",
            }
        }
        with patch("hermes_cli.config.load_config_readonly", return_value=string_override):
            assert get_native_image_limits() == (8 * 1024 * 1024, 9000)

    def test_oversized_corrupt_supported_image_is_skipped_without_crashing(
        self, tmp_path: Path, caplog
    ):
        from agent import image_routing as _ir

        image_path = tmp_path / "corrupt.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"not-an-image" * 500)
        config = {
            "agent": {
                "native_image_max_payload_bytes": 1000,
                "native_image_max_dimension": 7900,
            }
        }
        with patch("hermes_cli.config.load_config_readonly", return_value=config):
            assert _ir._file_to_data_url(image_path) is None

        assert "skipping attachment" in caplog.text

    def test_exif_orientation_is_applied_when_preprocessing(self, tmp_path: Path):
        from PIL import Image

        from agent import image_routing as _ir

        image = Image.new("RGB", (80, 40), (200, 100, 20))
        exif = Image.Exif()
        exif[274] = 6  # Rotate 90 degrees clockwise for display.
        image_path = tmp_path / "oriented.jpg"
        image.save(image_path, format="JPEG", quality=95, exif=exif)
        config = {
            "agent": {
                "native_image_max_payload_bytes": 100_000,
                "native_image_max_dimension": 50,
            }
        }

        with patch("hermes_cli.config.load_config_readonly", return_value=config):
            data_url = _ir._file_to_data_url(image_path)

        assert data_url is not None
        with Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as output:
            assert output.size == (25, 50)

    def test_transparent_oversized_image_remains_transparent_png(self, tmp_path: Path):
        from PIL import Image

        from agent import image_routing as _ir

        image = Image.effect_noise((512, 512), 100).convert("RGBA")
        image.putalpha(Image.linear_gradient("L").resize(image.size))
        image_path = tmp_path / "transparent.png"
        image.save(image_path, format="PNG")
        config = {
            "agent": {
                "native_image_max_payload_bytes": 150_000,
                "native_image_max_dimension": 7900,
            }
        }

        with patch("hermes_cli.config.load_config_readonly", return_value=config):
            data_url = _ir._file_to_data_url(image_path)

        assert data_url is not None
        assert data_url.startswith("data:image/png;base64,")
        assert len(data_url.encode("ascii")) <= 150_000
        with Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as output:
            assert "A" in output.getbands()

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


# ─── extract_image_refs ──────────────────────────────────────────────────────


class TestExtractImageRefs:
    """Scan task body / inbound text for image paths and URLs (kanban worker
    enrichment, issue raised May 2026)."""

    def test_empty_or_none_returns_empty(self):
        assert extract_image_refs("") == ([], [])
        assert extract_image_refs(None) == ([], [])  # type: ignore[arg-type]

    def test_finds_absolute_path(self, tmp_path: Path):
        img = tmp_path / "screenshot.png"
        img.write_bytes(_png_bytes())
        body = f"Look at {img} and tell me what's wrong."
        paths, urls = extract_image_refs(body)
        assert paths == [str(img)]
        assert urls == []

    def test_finds_home_relative_path(self, tmp_path: Path, monkeypatch):
        # Simulate ~/foo.png by pointing HOME at tmp_path and creating the file
        monkeypatch.setenv("HOME", str(tmp_path))
        img = tmp_path / "foo.png"
        img.write_bytes(_png_bytes())
        paths, urls = extract_image_refs("see ~/foo.png please")
        assert paths == [str(img)]
        assert urls == []


    def test_finds_http_image_url(self):
        body = "Check out https://example.com/photos/cat.png — cute right?"
        paths, urls = extract_image_refs(body)
        assert paths == []
        assert urls == ["https://example.com/photos/cat.png"]











# ─── build_native_content_parts with URLs ────────────────────────────────────


class TestBuildNativeContentPartsURLs:
    """URL pass-through support added so kanban task bodies (and other
    inbound surfaces) can route remote image URLs straight to the model."""

    def test_url_only_no_local_paths(self):
        parts, skipped = build_native_content_parts(
            "what is this?",
            [],
            image_urls=["https://example.com/diagram.png"],
        )
        assert skipped == []
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert "[Image attached: https://example.com/diagram.png]" in parts[0]["text"]
        assert parts[0]["text"].startswith("what is this?")
        assert parts[1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/diagram.png"},
        }

    def test_mixed_path_and_url(self, tmp_path: Path):
        img = tmp_path / "local.png"
        img.write_bytes(_png_bytes())
        parts, skipped = build_native_content_parts(
            "compare these",
            [str(img)],
            image_urls=["https://example.com/remote.jpg"],
        )
        assert skipped == []
        # 1 text + 2 image parts (local data URL first, then remote URL).
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == 2
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert image_parts[1]["image_url"]["url"] == "https://example.com/remote.jpg"
        text = parts[0]["text"]
        assert "[Image attached at:" in text
        assert "[Image attached: https://example.com/remote.jpg]" in text





# ─── Format compatibility: transcode non-universal formats to PNG ────────────


class TestFormatCompatibility:
    """Some image formats Discord (and other chat platforms) accept aren't
    accepted by every major vision provider. Anthropic for example returns
    HTTP 400 'Could not process image' for AVIF/HEIC/BMP/TIFF/ICO/SVG.

    We transcode anything outside the universal-safe set (PNG/JPEG/GIF/WEBP)
    to PNG with Pillow before declaring media_type so the provider call
    actually succeeds. Regression coverage for the user-reported Discord
    'Could not process image' HTTP 400 (issue #25935).
    """





    def test_svg_sniffed_correctly(self):
        from agent.image_routing import _sniff_mime_from_bytes
        assert _sniff_mime_from_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"/>') == "image/svg+xml"
        assert _sniff_mime_from_bytes(b'<?xml version="1.0"?><svg/>') == "image/svg+xml"

    def test_bmp_transcoded_to_png(self, tmp_path: Path):
        """BMP file should land as image/png in the data URL, not image/bmp,
        because not every provider (Anthropic) accepts BMP."""
        import pytest
        Image = pytest.importorskip("PIL.Image", reason="Pillow not installed; transcode is best-effort")
        from agent.image_routing import _file_to_data_url

        img_path = tmp_path / "scan.bmp"
        Image.new("RGB", (4, 4), (255, 0, 0)).save(img_path, format="BMP")
        url = _file_to_data_url(img_path)
        assert url is not None
        assert url.startswith("data:image/png;base64,"), (
            f"BMP must be transcoded to PNG for cross-provider compatibility, got: {url[:60]}"
        )


    def test_png_passes_through_no_transcode(self, tmp_path: Path):
        """Universal-safe formats must NOT be re-encoded — preserves bytes."""
        from agent.image_routing import _file_to_data_url

        img_path = tmp_path / "ok.png"
        img_path.write_bytes(_png_bytes())
        url = _file_to_data_url(img_path)
        assert url is not None
        assert url.startswith("data:image/png;base64,")
        b64 = url.split(",", 1)[1]
        assert base64.b64decode(b64) == _png_bytes()

    def test_file_to_data_url_blocks_read_denied_image_path(self, tmp_path: Path):
        """Native image routing must honor the shared credential read guard."""
        from agent.image_routing import _file_to_data_url

        img_path = tmp_path / ".env"
        img_path.write_bytes(_png_bytes())

        assert _file_to_data_url(img_path) is None


    def test_native_content_parts_blocks_image_symlink_to_read_denied_file(self, tmp_path: Path):
        from agent.image_routing import build_native_content_parts
        import os
        import pytest

        secret = tmp_path / ".env"
        secret.write_bytes(_png_bytes())
        img_link = tmp_path / "secret.png"
        try:
            os.symlink(secret, img_link)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        parts, skipped = build_native_content_parts("inspect this", [str(img_link)])

        assert skipped == [str(img_link)]
        assert all(part.get("type") != "image_url" for part in parts)





# ─── vision alias for custom providers ──────────────────────────────────────


class TestCustomProviderVisionAlias:
    """`vision: true` should work as an alias for `supports_vision: true`.

    Covers both config shapes that host named custom providers:
      * the ``providers.<name>.models`` dict, and
      * the legacy list-style ``custom_providers`` entries.

    Regression for the review of PR #31912: named custom providers resolve
    to the runtime value ``provider="custom"`` while the config keeps the
    user-declared name under ``model.provider``. The existing candidate-name
    resolver must be *extended* to accept the ``vision`` alias, not replaced.
    """


    def test_providers_dict_vision_alias_false(self):
        cfg = {
            "providers": {
                "my-vllm": {"models": {"llama-3": {"vision": False}}}
            }
        }
        assert _supports_vision_override(cfg, "my-vllm", "llama-3") is False


    def test_named_custom_provider_bare_custom_runtime_vision_alias(self):
        """Teknium's requested regression case.

        A named custom provider (``model.provider: my-vllm``) is rewritten to
        the runtime value ``provider="custom"`` by
        ``hermes_cli/runtime_provider.py``. The resolver must still match the
        ``my-vllm`` entry via the ``model.provider`` candidate and honour the
        ``vision`` alias.
        """
        cfg = {
            "model": {"provider": "my-vllm"},
            "providers": {
                "my-vllm": {"models": {"llava-v1.6": {"vision": True}}}
            },
        }
        # Runtime provider is the bare normalized value "custom".
        assert _supports_vision_override(cfg, "custom", "llava-v1.6") is True
        assert decide_image_input_mode("custom", "llava-v1.6", cfg) == "native"



    def test_vision_alias_none_when_model_absent(self):
        cfg = {
            "custom_providers": [
                {"name": "my-vllm", "models": {"llava": {"vision": True}}}
            ]
        }
        assert _supports_vision_override(cfg, "custom:my-vllm", "other") is None
