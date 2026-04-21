"""Tests for _safe_relpath and cross-drive path completion on Windows.

Covers the fix for issue #13261: ``os.path.relpath`` raises ``ValueError``
when the target path is on a different drive letter than the CWD (Windows
only).  The ``_safe_relpath`` helper catches this and falls back to the
absolute path.
"""

import os
from unittest.mock import patch

import pytest

from hermes_cli.commands import _safe_relpath


class TestSafeRelpath:
    """Unit tests for the _safe_relpath helper."""

    def test_normal_relative(self, tmp_path):
        """When relpath works normally, return the relative path."""
        child = tmp_path / "sub" / "file.txt"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.touch()
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = _safe_relpath(str(child))
            assert result == os.path.join("sub", "file.txt")
        finally:
            os.chdir(old)

    def test_normal_relative_with_start(self, tmp_path):
        """When relpath with an explicit start works, return the result."""
        child = tmp_path / "a" / "b.txt"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.touch()
        result = _safe_relpath(str(child), str(tmp_path))
        assert result == os.path.join("a", "b.txt")

    def test_cross_drive_fallback(self):
        """When relpath raises ValueError (cross-drive), return the path as-is."""
        fake_path = "D:\\projects\\file.py"
        with patch("os.path.relpath", side_effect=ValueError("path is on mount 'D:', start on mount 'C:'")):
            result = _safe_relpath(fake_path)
        assert result == fake_path

    def test_cross_drive_fallback_with_start(self):
        """Cross-drive fallback also works when *start* is explicit."""
        fake_path = "E:\\data\\report.csv"
        with patch("os.path.relpath", side_effect=ValueError("path is on mount 'E:', start on mount 'C:'")):
            result = _safe_relpath(fake_path, "C:\\Users\\me")
        assert result == fake_path

    def test_same_path_returns_dot(self, tmp_path):
        """relpath of a dir to itself is '.', and _safe_relpath preserves that."""
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = _safe_relpath(str(tmp_path))
            assert result == "."
        finally:
            os.chdir(old)

    def test_passes_start_to_relpath(self, tmp_path):
        """Ensure the *start* parameter is forwarded to os.path.relpath."""
        parent = tmp_path / "a"
        child = tmp_path / "a" / "b"
        parent.mkdir()
        child.mkdir()
        assert _safe_relpath(str(child), str(parent)) == "b"
