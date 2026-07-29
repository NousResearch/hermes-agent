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


def test_is_remote_shell_session_detects_ssh_env():
    assert clip.is_remote_shell_session({"SSH_CONNECTION": "1.2.3.4 5 6.7.8.9 22"})
    assert clip.is_remote_shell_session({"SSH_TTY": "/dev/pts/0"})
    assert clip.is_remote_shell_session({"SSH_CLIENT": "1.2.3.4 5 22"})
    assert not clip.is_remote_shell_session({})
    assert not clip.is_remote_shell_session({"TERM": "xterm-256color"})


class TestOsc52MultiplexerWrapping:
    """CLI _write_osc52_clipboard must wrap for tmux/screen passthrough
    (mirrors ui-tui/src/lib/osc52.ts wrapForMultiplexer)."""

    def _capture_seq(self, env):
        import io
        from unittest.mock import patch as _patch
        from cli import HermesCLI

        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj._app = None
        buf = io.StringIO()
        with _patch.dict(clip.os.environ, env, clear=False), \
             _patch("cli.sys.stdout", buf):
            for var in ("TMUX", "STY"):
                if var not in env:
                    clip.os.environ.pop(var, None)
            cli_obj._write_osc52_clipboard("hello")
        return buf.getvalue()

    def test_tmux_wraps_in_dcs_passthrough(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-123/default,1,0")
        monkeypatch.delenv("STY", raising=False)
        seq = self._capture_seq({"TMUX": "/tmp/tmux-123/default,1,0"})
        assert seq.startswith("\x1bPtmux;")
        assert "]52;c;" in seq
        assert seq.endswith("\x1b\\")


