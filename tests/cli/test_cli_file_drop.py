"""Tests for _detect_file_drop — file path detection that prevents
dragged/pasted absolute paths from being mistaken for slash commands."""

import os
import tempfile
from pathlib import Path

import pytest

from cli import _detect_file_drop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_image(tmp_path):
    """Create a temporary .png file and return its path."""
    img = tmp_path / "screenshot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header
    return img


@pytest.fixture()
def tmp_text(tmp_path):
    """Create a temporary .py file and return its path."""
    f = tmp_path / "main.py"
    f.write_text("print('hello')\n")
    return f


@pytest.fixture()
def tmp_image_with_spaces(tmp_path):
    """Create a file whose name contains spaces (like macOS screenshots)."""
    img = tmp_path / "Screenshot 2026-04-01 at 7.25.32 PM.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    return img


# ---------------------------------------------------------------------------
# Tests: returns None for non-file inputs
# ---------------------------------------------------------------------------

class TestNonFileInputs:
    def test_regular_slash_command(self):
        assert _detect_file_drop("/help") is None

    def test_unknown_slash_command(self):
        assert _detect_file_drop("/xyz") is None

    def test_slash_command_with_args(self):
        assert _detect_file_drop("/config set key value") is None

    def test_empty_string(self):
        assert _detect_file_drop("") is None

    def test_non_slash_input(self):
        assert _detect_file_drop("hello world") is None

    def test_non_string_input(self):
        assert _detect_file_drop(42) is None

    def test_nonexistent_path(self):
        assert _detect_file_drop("/nonexistent/path/to/file.png") is None

    def test_directory_not_file(self, tmp_path):
        """A directory path should not be treated as a file drop."""
        assert _detect_file_drop(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Tests: image file detection
# ---------------------------------------------------------------------------

class TestImageFileDrop:
    def test_simple_image_path(self, tmp_image):
        result = _detect_file_drop(str(tmp_image))
        assert result is not None
        assert result["path"] == tmp_image
        assert result["is_image"] is True
        assert result["remainder"] == ""

    def test_image_with_trailing_text(self, tmp_image):
        user_input = f"{tmp_image} analyze this please"
        result = _detect_file_drop(user_input)
        assert result is not None
        assert result["path"] == tmp_image
        assert result["is_image"] is True
        assert result["remainder"] == "analyze this please"

    @pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".gif", ".webp",
                                      ".bmp", ".tiff", ".tif", ".svg", ".ico"])
    def test_all_image_extensions(self, tmp_path, ext):
        img = tmp_path / f"test{ext}"
        img.write_bytes(b"fake")
        result = _detect_file_drop(str(img))
        assert result is not None
        assert result["is_image"] is True

    def test_uppercase_extension(self, tmp_path):
        img = tmp_path / "photo.JPG"
        img.write_bytes(b"fake")
        result = _detect_file_drop(str(img))
        assert result is not None
        assert result["is_image"] is True


# ---------------------------------------------------------------------------
# Tests: non-image file detection
# ---------------------------------------------------------------------------

class TestNonImageFileDrop:
    def test_python_file(self, tmp_text):
        result = _detect_file_drop(str(tmp_text))
        assert result is not None
        assert result["path"] == tmp_text
        assert result["is_image"] is False
        assert result["remainder"] == ""

    def test_non_image_with_trailing_text(self, tmp_text):
        user_input = f"{tmp_text} review this code"
        result = _detect_file_drop(user_input)
        assert result is not None
        assert result["is_image"] is False
        assert result["remainder"] == "review this code"


# ---------------------------------------------------------------------------
# Tests: backslash-escaped spaces (macOS drag-and-drop)
# ---------------------------------------------------------------------------

class TestEscapedSpaces:
    def test_escaped_spaces_in_path(self, tmp_image_with_spaces):
        r"""macOS drags produce paths like /path/to/my\ file.png"""
        escaped = str(tmp_image_with_spaces).replace(' ', '\\ ')
        result = _detect_file_drop(escaped)
        assert result is not None
        assert result["path"] == tmp_image_with_spaces
        assert result["is_image"] is True

    def test_escaped_spaces_with_trailing_text(self, tmp_image_with_spaces):
        escaped = str(tmp_image_with_spaces).replace(' ', '\\ ')
        user_input = f"{escaped} what is this?"
        result = _detect_file_drop(user_input)
        assert result is not None
        assert result["path"] == tmp_image_with_spaces
        assert result["remainder"] == "what is this?"

    def test_unquoted_spaces_in_path(self, tmp_image_with_spaces):
        result = _detect_file_drop(str(tmp_image_with_spaces))
        assert result is not None
        assert result["path"] == tmp_image_with_spaces
        assert result["is_image"] is True
        assert result["remainder"] == ""

    def test_unquoted_spaces_with_trailing_text(self, tmp_image_with_spaces):
        user_input = f"{tmp_image_with_spaces} what is this?"
        result = _detect_file_drop(user_input)
        assert result is not None
        assert result["path"] == tmp_image_with_spaces
        assert result["remainder"] == "what is this?"

    def test_mixed_escaped_and_literal_spaces_in_path(self, tmp_path):
        img = tmp_path / "Screenshot 2026-04-21 at 1.04.43 PM.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        mixed = str(img).replace("Screenshot ", "Screenshot\\ ").replace("2026-04-21 ", "2026-04-21\\ ").replace("at ", "at\\ ")
        result = _detect_file_drop(mixed)
        assert result is not None
        assert result["path"] == img
        assert result["is_image"] is True
        assert result["remainder"] == ""

    def test_file_uri_image_path(self, tmp_image_with_spaces):
        uri = tmp_image_with_spaces.as_uri()
        result = _detect_file_drop(uri)
        assert result is not None
        assert result["path"] == tmp_image_with_spaces
        assert result["is_image"] is True

    def test_tilde_prefixed_path(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        img = home / "storage" / "shared" / "Pictures" / "cat.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        monkeypatch.setenv("HOME", str(home))

        result = _detect_file_drop("~/storage/shared/Pictures/cat.png what is this?")

        assert result is not None
        assert result["path"] == img
        assert result["is_image"] is True
        assert result["remainder"] == "what is this?"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_path_with_no_extension(self, tmp_path):
        f = tmp_path / "Makefile"
        f.write_text("all:\n\techo hi\n")
        result = _detect_file_drop(str(f))
        assert result is not None
        assert result["is_image"] is False

    def test_path_that_looks_like_command_but_is_file(self, tmp_path):
        """A file literally named 'help' inside a directory starting with /."""
        f = tmp_path / "help"
        f.write_text("not a command\n")
        result = _detect_file_drop(str(f))
        assert result is not None
        assert result["is_image"] is False

    def test_symlink_to_file(self, tmp_image, tmp_path):
        link = tmp_path / "link.png"
        link.symlink_to(tmp_image)
        result = _detect_file_drop(str(link))
        assert result is not None
        assert result["is_image"] is True
