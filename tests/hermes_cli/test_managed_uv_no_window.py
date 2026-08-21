"""Tests for CREATE_NO_WINDOW on managed_uv subprocess calls (#88945)."""

from unittest.mock import patch

import pytest

from hermes_cli import managed_uv


def _call_arg_end(text: str, start: int) -> int:
    """Index just past the matching close paren of the call at *start*.

    Skips string literals so a ``)`` inside an argument (e.g.
    ``str(python)``) is not mistaken for the call's close paren.
    """
    depth = 0
    in_str = None
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == in_str and text[i - 1] != "\\":
                in_str = None
        elif ch in ("'", '"'):
            in_str = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


class TestNoWindowCreationFlags:
    def test_returns_zero_off_windows(self):
        with patch.object(managed_uv.sys, "platform", "linux"):
            assert managed_uv._no_window_creationflags() == 0

    def test_returns_create_no_window_on_windows(self):
        with patch.object(managed_uv.sys, "platform", "win32"):
            expected = getattr(
                __import__("subprocess"), "CREATE_NO_WINDOW", 0x08000000
            )
            assert managed_uv._no_window_creationflags() == expected

    def test_every_subprocess_call_carries_the_flag(self):
        """Guard: no subprocess.run call in this module may regress to a
        windowed spawn (console flash when Hermes runs console-less)."""
        import re

        src = managed_uv.__file__
        with open(src, encoding="utf-8") as f:
            text = f.read()

        calls = [m.start() for m in re.finditer(r"subprocess\.run\(", text)]
        assert calls, "expected subprocess.run call sites in managed_uv"

        for start in calls:
            end = _call_arg_end(text, start)
            segment = text[start:end]
            assert "creationflags=" in segment, (
                f"subprocess.run call at offset {start} lacks "
                "creationflags=_no_window_creationflags() (#88945)"
            )
