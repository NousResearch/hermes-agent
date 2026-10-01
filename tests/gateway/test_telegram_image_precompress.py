"""Tests for Telegram large-image pre-compression (salvage of PR #74893).

Behind slow HTTP proxies, raw PNGs > 1-2MB exceed PTB's media_write_timeout.
``_compress_image_to_jpeg`` converts large raster images to progressive JPEG
(resizing above 1600px) before upload. These tests exercise the real Pillow
path with real file I/O — no mocks on the compression itself.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from gateway.platforms.base import PlatformConfig  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _adapter():
    return TelegramAdapter(PlatformConfig(enabled=True, token="***"))


def _make_png(path, size=(2400, 1800), noisy=True, mode="RGB"):
    """Write a PNG large enough to cross the 1MB compression threshold."""
    if noisy:
        # True random noise defeats PNG compression so the file exceeds 1MB.
        nbytes = size[0] * size[1] * len(mode)
        img = Image.frombytes(mode, size, os.urandom(nbytes))
    else:
        img = Image.new(mode, size)
    img.save(path, "PNG")
    return path


class TestSniffRasterFormat:
    def test_png_magic(self, tmp_path):
        p = tmp_path / "x.png"
        Image.new("RGB", (4, 4)).save(p, "PNG")
        assert TelegramAdapter._sniff_raster_format(str(p)) == "png"

    def test_webp_magic(self, tmp_path):
        p = tmp_path / "x.webp"
        Image.new("RGB", (4, 4)).save(p, "WEBP")
        assert TelegramAdapter._sniff_raster_format(str(p)) == "webp"

    def test_gif_excluded(self, tmp_path):
        p = tmp_path / "x.gif"
        Image.new("P", (4, 4)).save(p, "GIF")
        assert TelegramAdapter._sniff_raster_format(str(p)) is None

    def test_non_image(self, tmp_path):
        p = tmp_path / "x.bin"
        p.write_bytes(b"not an image at all")
        assert TelegramAdapter._sniff_raster_format(str(p)) is None

    def test_missing_file(self, tmp_path):
        assert TelegramAdapter._sniff_raster_format(str(tmp_path / "nope.png")) is None


class TestCompressImageToJpeg:
    def test_large_png_is_compressed_and_resized(self, tmp_path):
        adapter = _adapter()
        src = _make_png(tmp_path / "big.png")
        assert os.path.getsize(src) > adapter._IMG_COMPRESS_THRESHOLD_BYTES

        out = adapter._compress_image_to_jpeg(str(src))
        assert out is not None
        try:
            assert out.endswith(".jpg")
            assert os.path.getsize(out) < os.path.getsize(src)
            with Image.open(out) as jpg:
                assert jpg.format == "JPEG"
                assert max(jpg.size) <= adapter._IMG_MAX_DIMENSION
        finally:
            os.remove(out)

    def test_small_png_is_left_alone(self, tmp_path):
        adapter = _adapter()
        src = tmp_path / "small.png"
        Image.new("RGB", (32, 32)).save(src, "PNG")
        assert adapter._compress_image_to_jpeg(str(src)) is None

    def test_jpeg_is_left_alone(self, tmp_path):
        adapter = _adapter()
        src = tmp_path / "photo.jpg"
        Image.new("RGB", (2000, 2000)).save(src, "JPEG")
        assert adapter._compress_image_to_jpeg(str(src)) is None

    def test_transparent_png_gets_white_background(self, tmp_path):
        """RGBA must composite onto white, not collapse to a black background."""
        adapter = _adapter()
        src = tmp_path / "transparent.png"
        img = Image.new("RGBA", (2000, 2000), (0, 0, 0, 0))  # fully transparent
        img.save(src, "PNG")
        # Fully-transparent flat PNG compresses tiny; force it over threshold
        # by lowering the adapter threshold for this test.
        adapter._IMG_COMPRESS_THRESHOLD_BYTES = 0

        out = adapter._compress_image_to_jpeg(str(src))
        assert out is not None
        try:
            with Image.open(out) as jpg:
                # Transparent areas must render white (255), not black (0).
                assert jpg.getpixel((10, 10))[0] > 200
        finally:
            os.remove(out)

    def test_missing_file_returns_none(self, tmp_path):
        adapter = _adapter()
        assert adapter._compress_image_to_jpeg(str(tmp_path / "gone.png")) is None
