"""Regression test for #70708 — WinError 1920 on `hermes update`.

`Path.is_file()` raises `OSError` on Windows when another process holds the
target file open (e.g. a persistent `pythonw.exe` reading from the managed
Node tree). The `hermes update` flow must treat that as "not a file" and
fall through to PATH-based resolution or heal-and-retry, not crash the
upgrade.

This test simulates the `OSError` by monkey-patching `Path.is_file` to
raise. It is platform-independent (the test does not need to be run on
Windows — it asserts the contract that any `OSError` is swallowed).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_constants import _safe_path_is_file


def test_safe_path_is_file_true_when_pathlib_says_true(tmp_path: Path) -> None:
    p = tmp_path / "real.txt"
    p.write_text("hi")
    assert _safe_path_is_file(p) is True


def test_safe_path_is_file_false_when_missing(tmp_path: Path) -> None:
    assert _safe_path_is_file(tmp_path / "nope.txt") is False


def test_safe_path_is_file_false_when_path_is_directory(tmp_path: Path) -> None:
    assert _safe_path_is_file(tmp_path) is False


def test_safe_path_is_file_swallows_oserror(tmp_path: Path) -> None:
    """The bug: `Path.is_file()` raised OSError on a Windows file held by
    another process. `_safe_path_is_file` must return False instead."""
    p = tmp_path / "locked.txt"
    p.write_text("x")

    real_is_file = Path.is_file

    def raising_is_file(self, *args, **kwargs):
        # Simulate WinError 1920 — Windows would raise PermissionError /
        # OSError directly. The function under test must catch OSError
        # (the base class) so any platform-specific subclass is covered.
        raise OSError(1920, "The file cannot be accessed by the system")

    with patch.object(Path, "is_file", raising_is_file):
        assert _safe_path_is_file(p) is False

    # Sanity: real path still works after restore.
    assert real_is_file(p) is True


def test_safe_path_is_file_accepts_string_path(tmp_path: Path) -> None:
    p = tmp_path / "str.txt"
    p.write_text("x")
    assert _safe_path_is_file(str(p)) is True
    assert _safe_path_is_file(str(tmp_path / "missing.txt")) is False


def test_safe_path_is_file_no_qstring_handling(tmp_path: Path) -> None:
    """Defensive: an empty string should not raise, should return False."""
    assert _safe_path_is_file("") is False
