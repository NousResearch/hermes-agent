"""Tests for image cache helpers in gateway/platforms/base.py."""
import os
import pytest

from gateway.platforms.base import (
    cache_image_from_bytes,
    _detect_image_ext,
    _looks_like_image,
)


# Minimal valid magic-byte payloads for each supported format.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 16
GIF87_BYTES = b"GIF87a" + b"\x00" * 16
GIF89_BYTES = b"GIF89a" + b"\x00" * 16
WEBP_BYTES = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 16
BMP_BYTES = b"BM" + b"\x00" * 32
HTML_BYTES = b"<!DOCTYPE html><html><body>not an image</body></html>"


class TestDetectImageExt:
    """Magic-byte detection returns canonical extensions."""

    @pytest.mark.parametrize("data,expected", [
        (PNG_BYTES, ".png"),
        (JPEG_BYTES, ".jpg"),
        (GIF87_BYTES, ".gif"),
        (GIF89_BYTES, ".gif"),
        (WEBP_BYTES, ".webp"),
        (BMP_BYTES, ".bmp"),
    ])
    def test_known_format(self, data, expected):
        assert _detect_image_ext(data) == expected

    def test_unknown_format(self):
        assert _detect_image_ext(HTML_BYTES) is None

    def test_too_short(self):
        assert _detect_image_ext(b"\x89") is None
        assert _detect_image_ext(b"") is None


class TestCacheImageFromBytes:
    """Caching writes a file with a corrected extension based on magic bytes."""

    def test_corrects_extension_when_caller_hint_wrong(self, tmp_path, monkeypatch):
        """If caller passes ext='.jpg' but bytes are PNG, file is saved as .png.

        This is the regression case: Discord (and other gateways) sometimes
        report Content-Type 'image/jpeg' for payloads that are actually PNG
        (clipboard-pasted screenshots, browser-converted images). Anthropic's
        API rejects such mismatches with HTTP 400. The cache must trust
        magic bytes.
        """
        monkeypatch.setattr(
            "gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path
        )
        path = cache_image_from_bytes(PNG_BYTES, ext=".jpg")
        assert path.endswith(".png"), f"expected .png, got {path}"
        assert os.path.exists(path)

    def test_corrects_extension_when_caller_hint_jpeg_alias(self, tmp_path, monkeypatch):
        """Caller passes '.jpeg' (alias) but bytes are PNG → file saved as .png."""
        monkeypatch.setattr(
            "gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path
        )
        path = cache_image_from_bytes(PNG_BYTES, ext=".jpeg")
        assert path.endswith(".png")

    def test_keeps_correct_extension(self, tmp_path, monkeypatch):
        """If caller hint matches reality, extension is preserved (canonical form)."""
        monkeypatch.setattr(
            "gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path
        )
        path = cache_image_from_bytes(JPEG_BYTES, ext=".jpg")
        assert path.endswith(".jpg")

    def test_normalizes_jpeg_to_jpg(self, tmp_path, monkeypatch):
        """JPEG payload + '.jpeg' hint → canonical '.jpg' extension on disk."""
        monkeypatch.setattr(
            "gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path
        )
        path = cache_image_from_bytes(JPEG_BYTES, ext=".jpeg")
        assert path.endswith(".jpg")

    def test_rejects_non_image(self, tmp_path, monkeypatch):
        """HTML / error pages must not be cached as images."""
        monkeypatch.setattr(
            "gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path
        )
        with pytest.raises(ValueError, match="Refusing to cache"):
            cache_image_from_bytes(HTML_BYTES, ext=".jpg")

    def test_webp_detected_from_riff(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path
        )
        path = cache_image_from_bytes(WEBP_BYTES, ext=".jpg")
        assert path.endswith(".webp")

    def test_file_contents_match_input(self, tmp_path, monkeypatch):
        """Bytes written to disk must equal bytes passed in."""
        monkeypatch.setattr(
            "gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path
        )
        path = cache_image_from_bytes(PNG_BYTES, ext=".jpg")
        with open(path, "rb") as f:
            assert f.read() == PNG_BYTES


class TestLooksLikeImage:
    """Magic-byte sniffing accepts all supported formats and rejects others."""

    @pytest.mark.parametrize("data", [
        PNG_BYTES, JPEG_BYTES, GIF87_BYTES, GIF89_BYTES, WEBP_BYTES, BMP_BYTES,
    ])
    def test_accepts_known_formats(self, data):
        assert _looks_like_image(data) is True

    def test_rejects_html(self):
        assert _looks_like_image(HTML_BYTES) is False

    def test_rejects_empty(self):
        assert _looks_like_image(b"") is False
        assert _looks_like_image(b"\x00\x00") is False
