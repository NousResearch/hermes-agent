"""Tests for native clipboard text write (hermes_cli/clipboard.py).

Mirrors the TUI's writeClipboardText fallback chain: pbcopy /
PowerShell Set-Clipboard / wl-copy / xclip / xsel, with OSC 52 left to
the caller when every backend fails.
"""
import base64
import subprocess
from unittest.mock import patch

import pytest

from hermes_cli import clipboard as clip


def _completed(returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode)


def test_darwin_uses_pbcopy():
    with patch.object(clip.sys, "platform", "darwin"), \
         patch.object(clip.subprocess, "run", return_value=_completed()) as run:
        assert clip.write_clipboard_text("hello") is True
    argv = run.call_args[0][0]
    assert argv == ["pbcopy"]
    assert run.call_args[1]["input"] == b"hello"


def test_windows_uses_powershell_base64():
    with patch.object(clip.sys, "platform", "win32"), \
         patch.object(clip.subprocess, "run", return_value=_completed()) as run:
        assert clip.write_clipboard_text("héllo 🎉") is True
    argv = run.call_args[0][0]
    assert argv[0] == "powershell"
    script = argv[-1]
    b64 = base64.b64encode("héllo 🎉".encode("utf-8")).decode("ascii")
    assert b64 in script
    assert "Set-Clipboard" in script


def test_linux_falls_through_backends_until_success():
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv[0])
        # xclip fails, xsel succeeds
        return _completed(returncode=0 if argv[0] == "xsel" else 1)

    with patch.object(clip.sys, "platform", "linux"), \
         patch.object(clip, "_is_wsl", return_value=False), \
         patch.dict(clip.os.environ, {}, clear=False), \
         patch.object(clip.os.environ, "get", lambda k, d=None: None), \
         patch.object(clip.subprocess, "run", side_effect=fake_run):
        assert clip.write_clipboard_text("x") is True
    assert calls == ["xclip", "xsel"]


def test_returns_false_when_all_backends_fail():
    with patch.object(clip.sys, "platform", "linux"), \
         patch.object(clip, "_is_wsl", return_value=False), \
         patch.object(clip.os.environ, "get", lambda k, d=None: None), \
         patch.object(clip.subprocess, "run", side_effect=FileNotFoundError):
        assert clip.write_clipboard_text("x") is False


def test_wayland_prefers_wl_copy():
    with patch.object(clip.sys, "platform", "linux"), \
         patch.object(clip, "_is_wsl", return_value=False), \
         patch.object(clip.os.environ, "get",
                      lambda k, d=None: ":0" if k == "WAYLAND_DISPLAY" else None), \
         patch.object(clip.subprocess, "run", return_value=_completed()) as run:
        assert clip.write_clipboard_text("x") is True
    assert run.call_args[0][0][0] == "wl-copy"
