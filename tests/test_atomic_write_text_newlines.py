"""Tests for atomic_write_text newline preservation and cross-platform line endings (#109674)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import utils


def test_atomic_write_text_newline_preservation(tmp_path: Path) -> None:
    """Writing text via atomic_write_text must preserve exact newlines without CRLF translation."""
    target = tmp_path / "exact_lf.txt"
    content = "line1\nline2\nline3\n"
    utils.atomic_write_text(target, content)
    with open(target, "rb") as f:
        raw = f.read()
    assert b"\r\n" not in raw
    assert raw == b"line1\nline2\nline3\n"


def test_atomic_write_text_preserves_explicit_crlf(tmp_path: Path) -> None:
    """If content explicitly has CRLF, it should be written as CRLF without mutation."""
    target = tmp_path / "exact_crlf.txt"
    content = "line1\r\nline2\r\nline3\r\n"
    utils.atomic_write_text(target, content)
    with open(target, "rb") as f:
        raw = f.read()
    assert raw == b"line1\r\nline2\r\nline3\r\n"


def test_atomic_write_passes_newline_empty_to_fdopen(tmp_path: Path) -> None:
    """Verify that os.fdopen is called with newline='' to disable platform newline translation."""
    target = tmp_path / "test_fdopen.txt"
    fdopen_kwargs = {}
    orig_fdopen = utils.os.fdopen

    def mock_fdopen(*args, **kwargs):
        fdopen_kwargs.update(kwargs)
        return orig_fdopen(*args, **kwargs)

    with patch("utils.os.fdopen", side_effect=mock_fdopen):
        utils.atomic_write_text(target, "test\n")

    assert fdopen_kwargs.get("newline") == ""
