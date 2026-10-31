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


@pytest.mark.platforms("macos")
def test_darwin_uses_pbcopy():
    with patch.object(clip.subprocess, "run", return_value=_completed()) as run:
        assert clip.write_clipboard_text("hello") is True
    argv = run.call_args[0][0]
    assert argv == ["pbcopy"]
    assert run.call_args[1]["input"] == b"hello"


@pytest.mark.platforms("linux")
def test_linux_falls_through_backends_until_success():
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv[0])
        # xclip fails, xsel succeeds
        return _completed(returncode=0 if argv[0] == "xsel" else 1)

    with patch.object(clip, "_is_wsl", return_value=False), \
         patch.dict(clip.os.environ, {}, clear=False), \
         patch.object(clip.os.environ, "get", lambda k, d=None: None), \
         patch.object(clip.subprocess, "run", side_effect=fake_run):
        assert clip.write_clipboard_text("x") is True
    assert calls == ["xclip", "xsel"]


@pytest.mark.platforms("windows")
def test_windows_copy_past_the_command_line_limit_reaches_powershell():
    """The base64 payload rode PowerShell's -Command argument, so copying more than ~24 KB of
    UTF-8 overflowed Windows' 32,767-character command line (WinError 206) and /copy fell back to
    OSC 52, which conhost and many other terminals ignore."""
    text = "长文本 line 🎉\n" * 4000  # ~80 KB of UTF-8: past the limit even before base64
    runs = []

    def fake_run(argv, **kwargs):
        if len(subprocess.list2cmdline(argv)) >= 32767:  # CreateProcess refuses the spawn
            raise OSError(206, "The filename or extension is too long")
        runs.append((argv, kwargs))
        return _completed()

    with patch.object(clip.subprocess, "run", side_effect=fake_run):
        assert clip.write_clipboard_text(text) is True
    [(argv, kwargs)] = runs
    assert "powershell" in argv[0].lower()
    assert base64.b64decode(kwargs["input"]).decode("utf-8") == text


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


