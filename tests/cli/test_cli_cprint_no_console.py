"""Regression tests for `_cprint` on a non-console stdout (issue #65558).

On Windows, `hermes chat -q` with piped/redirected stdout crashed with
`prompt_toolkit.output.win32.NoConsoleScreenBufferError`, because the pt
output backend can't build a Win32 console from a pipe handle. `_cprint` now
detects the non-console case up front and emits plain, ANSI-stripped,
encoding-safe text instead — parity with prompt_toolkit's own POSIX fallback.

Under pytest, `capsys` replaces stdout with a non-tty capture, so
`_stdout_is_console()` naturally returns False — the exact broken path — with
no monkeypatching needed.
"""

from __future__ import annotations

import sys

import cli


class TestStdoutIsConsole:
    def test_false_when_not_a_tty(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", _FakeStream(isatty=False))
        assert cli._stdout_is_console() is False

    def test_true_when_a_tty(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", _FakeStream(isatty=True))
        assert cli._stdout_is_console() is True

    def test_false_when_isatty_raises(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", _FakeStream(isatty_raises=True))
        assert cli._stdout_is_console() is False


class TestPlainOutput:
    def test_strips_ansi_escapes(self, capsys):
        cli._plain_output("\x1b[2;3mInitializing agent...\x1b[0m")
        out = capsys.readouterr().out
        assert out == "Initializing agent...\n"
        assert "\x1b" not in out

    def test_strips_cursor_and_erase_sequences(self, capsys):
        cli._plain_output("\x1b[2K\x1b[1Gline\x1b[0m")
        assert capsys.readouterr().out == "line\n"

    def test_encoding_safe_on_legacy_codepage(self, monkeypatch):
        # A cp1252 stream raises UnicodeEncodeError on box/braille chars; the
        # fallback must re-encode with replacement rather than crash (or drop).
        stream = _Cp1252Stream()
        monkeypatch.setattr(sys, "stdout", stream)
        cli._plain_output("box █ braille ⠰")  # █ ⠰
        written = "".join(stream.written)
        assert "box " in written and "?" in written
        assert "█" not in written  # the un-encodable char was replaced


class TestCprintNoConsole:
    def test_does_not_raise_and_strips_ansi(self, capsys):
        # capsys stdout is non-tty → this is the piped `hermes chat -q` path
        # that used to raise NoConsoleScreenBufferError on Windows.
        cli._cprint("\x1b[2;3mInitializing agent...\x1b[0m")
        out = capsys.readouterr().out
        assert "Initializing agent..." in out
        assert "\x1b" not in out

    def test_console_path_still_uses_prompt_toolkit(self, monkeypatch):
        # When stdout IS a console and no app is running, `_cprint` must still
        # route through the prompt_toolkit renderer (unchanged behavior).
        monkeypatch.setattr(cli, "_stdout_is_console", lambda: True)
        calls = []
        monkeypatch.setattr(cli, "_pt_print", lambda payload: calls.append(payload))
        cli._cprint("hello")
        assert len(calls) == 1


class _FakeStream:
    encoding = "utf-8"

    def __init__(self, *, isatty: bool = False, isatty_raises: bool = False):
        self._isatty = isatty
        self._isatty_raises = isatty_raises

    def isatty(self) -> bool:
        if self._isatty_raises:
            raise OSError("no fileno")
        return self._isatty

    def write(self, s):  # pragma: no cover - not exercised
        return len(s)

    def flush(self):  # pragma: no cover
        pass


class _Cp1252Stream:
    encoding = "cp1252"

    def __init__(self):
        self.written = []

    def isatty(self) -> bool:
        return False

    def write(self, s):
        s.encode("cp1252")  # raises UnicodeEncodeError on un-encodable chars
        self.written.append(s)
        return len(s)

    def flush(self):
        pass
