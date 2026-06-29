"""Tests for PNG alpha → JPEG conversion in the Discord adapter.

Ensures that PNG files with an alpha channel are automatically converted
to JPEG before being sent as Discord attachments, preventing the silent
``multimodal_tool_content_unsupported`` delivery failure.
"""
from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("PIL.Image")


def _make_png(path: str, mode: str = "RGBA") -> str:
    """Create a tiny PNG file with the given PIL mode."""
    from PIL import Image
    if mode == "RGBA":
        img = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
    elif mode == "LA":
        img = Image.new("LA", (4, 4), (128, 200))
    elif mode == "RGB":
        img = Image.new("RGB", (4, 4), (0, 255, 0))
    else:
        img = Image.new(mode, (4, 4))
    img.save(path, "PNG")
    return path


class TestEnsureNoAlphaPng:
    """Tests for ``_ensure_no_alpha_png`` helper in the Discord adapter."""

    def _import_helper(self):
        from plugins.platforms.discord.adapter import _ensure_no_alpha_png
        return _ensure_no_alpha_png

    def test_rgba_png_is_converted(self):
        helper = self._import_helper()
        fd, src = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        tmp = None
        try:
            _make_png(src, "RGBA")
            usable, tmp = helper(src)
            assert tmp is not None
            assert tmp != src
            assert os.path.exists(tmp)
            assert tmp.endswith(".jpg")
            # Verify the converted image has no alpha
            from PIL import Image
            with Image.open(tmp) as img:
                assert img.mode == "RGB"
        finally:
            os.unlink(src)
            if tmp:
                os.unlink(tmp)

    def test_la_png_is_converted(self):
        helper = self._import_helper()
        fd, src = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        tmp = None
        try:
            _make_png(src, "LA")
            usable, tmp = helper(src)
            assert tmp is not None
            assert usable != src
        finally:
            os.unlink(src)
            if tmp:
                os.unlink(tmp)

    def test_rgb_png_is_not_converted(self):
        helper = self._import_helper()
        fd, src = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            _make_png(src, "RGB")
            usable, tmp = helper(src)
            assert tmp is None
            assert usable == src
        finally:
            os.unlink(src)

    def test_jpeg_is_not_converted(self):
        helper = self._import_helper()
        fd, src = tempfile.mkstemp(suffix=".jpeg")
        os.close(fd)
        try:
            from PIL import Image
            Image.new("RGB", (4, 4)).save(src, "JPEG")
            usable, tmp = helper(src)
            assert tmp is None
            assert usable == src
        finally:
            os.unlink(src)

    def test_missing_file_returns_original(self):
        helper = self._import_helper()
        usable, tmp = helper("/nonexistent/path/image.png")
        assert tmp is None
        assert usable == "/nonexistent/path/image.png"

    def test_cleanup_required_after_use(self):
        """Verify that the caller can clean up temp files."""
        helper = self._import_helper()
        fd, src = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            _make_png(src, "RGBA")
            usable, tmp = helper(src)
            assert tmp is not None
            assert os.path.exists(tmp)
            os.unlink(tmp)
            assert not os.path.exists(tmp)
        finally:
            os.unlink(src)
