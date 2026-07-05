"""Tests for ``cli._notify_input_needed`` — the unified BEL + OSC 9 alert.

Covers:
- OSC 9 sequence shape (ESC ] 9 ; <body> BEL) written to /dev/tty
- BEL (\\a) also written alongside OSC 9
- Disable via ``display.input_alert: false``
- No-op when stdout is not a TTY (redirected-log guard)
- Body sanitization: C0 control chars and DEL stripped from OSC payload
- Failure path silently absorbs IO errors (no /dev/tty in sandbox/Windows)
"""
from __future__ import annotations

from unittest.mock import patch, mock_open, MagicMock

import cli as cli_module


def test_notify_writes_osc9_with_bel_terminator(monkeypatch):
    """Helper writes ``ESC ] 9 ; <body> BEL`` to /dev/tty when enabled + TTY."""
    monkeypatch.setitem(cli_module.CLI_CONFIG, "display", {"input_alert": True})

    fake_tty = MagicMock()
    m = mock_open()
    m.return_value.__enter__.return_value = fake_tty

    with patch("cli.sys.stdout.isatty", return_value=True), \
         patch("builtins.open", m):
        cli_module._notify_input_needed("hello world")

    m.assert_called_once_with("/dev/tty", "w", buffering=1, encoding="utf-8")
    # First write is OSC 9
    osc_call = fake_tty.write.call_args_list[0]
    assert osc_call.args[0] == "\x1b]9;hello world\x07"


def test_notify_writes_bel_char(monkeypatch):
    """BEL (\\a) is also written for terminals that don't parse OSC 9."""
    monkeypatch.setitem(cli_module.CLI_CONFIG, "display", {"input_alert": True})

    fake_tty = MagicMock()
    m = mock_open()
    m.return_value.__enter__.return_value = fake_tty

    with patch("cli.sys.stdout.isatty", return_value=True), \
         patch("builtins.open", m):
        cli_module._notify_input_needed("test")

    # Second write is BEL
    bel_call = fake_tty.write.call_args_list[1]
    assert bel_call.args[0] == "\a"


def test_notify_disabled_via_config(monkeypatch):
    """Setting ``display.input_alert: false`` short-circuits before any IO."""
    monkeypatch.setitem(cli_module.CLI_CONFIG, "display", {"input_alert": False})

    m = mock_open()
    with patch("cli.sys.stdout.isatty", return_value=True), \
         patch("builtins.open", m):
        cli_module._notify_input_needed("hello")

    m.assert_not_called()


def test_notify_skipped_when_stdout_not_tty(monkeypatch):
    """Redirected stdout (pipes, file logs) — no notification emitted."""
    monkeypatch.setitem(cli_module.CLI_CONFIG, "display", {"input_alert": True})

    m = mock_open()
    with patch("cli.sys.stdout.isatty", return_value=False), \
         patch("builtins.open", m):
        cli_module._notify_input_needed("hello")

    m.assert_not_called()


def test_notify_swallows_io_errors(monkeypatch):
    """If /dev/tty can't be opened (sandbox, non-POSIX), the helper is silent."""
    monkeypatch.setitem(cli_module.CLI_CONFIG, "display", {"input_alert": True})

    def _boom(*_args, **_kwargs):
        raise OSError("no tty")

    with patch("cli.sys.stdout.isatty", return_value=True), \
         patch("builtins.open", _boom):
        # Must not raise. The whole point of the helper is best-effort.
        cli_module._notify_input_needed("hello")


def test_notify_sanitizes_control_chars(monkeypatch):
    """OSC 9 body must not contain ESC, BEL, or other C0 control chars.

    Without sanitization, a malicious or malformed body string could inject
    terminal escape sequences into the OSC payload.
    """
    monkeypatch.setitem(cli_module.CLI_CONFIG, "display", {"input_alert": True})

    fake_tty = MagicMock()
    m = mock_open()
    m.return_value.__enter__.return_value = fake_tty

    # Body with ESC (0x1b), BEL (0x07), and NUL (0x00) injected
    malicious_body = "hello\x1b[2J\x07world\x00"

    with patch("cli.sys.stdout.isatty", return_value=True), \
         patch("builtins.open", m):
        cli_module._notify_input_needed(malicious_body)

    osc_call = fake_tty.write.call_args_list[0]
    payload = osc_call.args[0]
    # The OSC payload must contain the sanitized body
    assert "hello" in payload
    assert "world" in payload
    # No injected control chars survived into the body portion
    # (the OSC framing chars \x1b]9; and \x07 terminator are expected)
    body_portion = payload.removeprefix("\x1b]9;").removesuffix("\x07")
    assert "\x1b" not in body_portion
    assert "\x07" not in body_portion
    assert "\x00" not in body_portion
