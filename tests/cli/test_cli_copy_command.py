"""Tests for CLI /copy command."""

import base64
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI
from hermes_cli import clipboard as clipboard_mod


def _make_cli() -> HermesCLI:
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = "sess-copy-test"
    cli_obj._pending_input = MagicMock()
    cli_obj._app = None
    return cli_obj


def test_copy_copies_latest_assistant_message():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "latest"},
    ]

    with patch("hermes_cli.clipboard.write_clipboard_text", return_value=True) as mock_copy:
        result = cli_obj.process_command("/copy")

    assert result is True
    assert mock_copy.call_count == 1
    args, kwargs = mock_copy.call_args
    assert args[0] == "latest"


def test_copy_with_index_uses_requested_assistant_message():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [
        {"role": "assistant", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]

    with patch("hermes_cli.clipboard.write_clipboard_text", return_value=True) as mock_copy:
        cli_obj.process_command("/copy 1")

    args, _ = mock_copy.call_args
    assert args[0] == "one"


def test_copy_strips_reasoning_blocks_before_copy():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [
        {
            "role": "assistant",
            "content": "<REASONING_SCRATCHPAD>internal</REASONING_SCRATCHPAD>\nVisible answer",
        }
    ]

    with patch("hermes_cli.clipboard.write_clipboard_text", return_value=True) as mock_copy:
        cli_obj.process_command("/copy")

    args, _ = mock_copy.call_args
    assert args[0] == "Visible answer"


def test_copy_invalid_index_does_not_copy():
    cli_obj = _make_cli()
    cli_obj.conversation_history = [{"role": "assistant", "content": "only"}]

    with patch("hermes_cli.clipboard.write_clipboard_text", return_value=True) as mock_copy, \
         patch("cli._cprint") as mock_print:
        cli_obj.process_command("/copy 99")

    mock_copy.assert_not_called()
    assert any("Invalid response number" in str(call) for call in mock_print.call_args_list)


# ═════════════════════════════════════════════════════════════════════════
# write_clipboard_text — native fallback layer in hermes_cli/clipboard.py
# ═════════════════════════════════════════════════════════════════════════


@pytest.fixture
def fake_run():
    """Patch subprocess.run inside the clipboard module to a mock."""
    with patch.object(clipboard_mod.subprocess, "run") as m:
        m.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        yield m


def _no_ssh(monkeypatch):
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)


def _ssh_session(monkeypatch):
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 1234 5.6.7.8 22")
    monkeypatch.setenv("SSH_TTY", "/dev/pts/0")


def test_write_clipboard_text_macos_uses_pbcopy(monkeypatch, fake_run):
    """On a local macOS terminal, native pbcopy should be invoked."""
    _no_ssh(monkeypatch)
    monkeypatch.setattr(clipboard_mod.sys, "platform", "darwin")
    osc52_calls = []

    ok = clipboard_mod.write_clipboard_text(
        "hello", osc52_writer=lambda seq: osc52_calls.append(seq)
    )

    assert ok is True
    assert fake_run.called
    cmd_args = fake_run.call_args.args[0]
    assert cmd_args[0] == "pbcopy"
    # input should be the UTF-8 bytes of the payload
    assert fake_run.call_args.kwargs.get("input") == "hello".encode("utf-8")
    # Native succeeded — OSC 52 should NOT have been emitted
    assert osc52_calls == []


def test_write_clipboard_text_macos_falls_back_to_osc52_when_pbcopy_missing(monkeypatch, fake_run):
    """If pbcopy is absent, OSC 52 should be emitted as a fallback."""
    _no_ssh(monkeypatch)
    monkeypatch.setattr(clipboard_mod.sys, "platform", "darwin")
    fake_run.side_effect = FileNotFoundError("pbcopy not found")
    osc52_calls = []

    ok = clipboard_mod.write_clipboard_text(
        "hi", osc52_writer=lambda seq: osc52_calls.append(seq)
    )

    assert ok is True
    assert len(osc52_calls) == 1
    payload = base64.b64encode(b"hi").decode("ascii")
    assert osc52_calls[0] == f"\x1b]52;c;{payload}\x07"


def test_write_clipboard_text_pbcopy_nonzero_exit_reports_failure(monkeypatch, fake_run):
    """If pbcopy is installed but exits non-zero, the native path should
    report failure (so any OSC 52 fallback can run; with no fallback,
    write_clipboard_text returns False and the user sees a real error).

    Covers _run_clipboard_writer's ``returncode != 0`` branch — a real
    case (e.g. pbcopy hitting a sandbox restriction or an unwritable
    pasteboard) that previously had no test coverage.
    """
    _no_ssh(monkeypatch)
    monkeypatch.setattr(clipboard_mod.sys, "platform", "darwin")
    # pbcopy is found, but returns a failure exit code with diagnostic stderr.
    fake_run.return_value = subprocess.CompletedProcess(
        args=["pbcopy"], returncode=1, stdout=b"", stderr=b"pbcopy: write failed",
    )

    # Direct unit assertion: with no OSC 52 sink, native failure → write_clipboard_text returns False.
    ok = clipboard_mod.write_clipboard_text("payload", osc52_writer=None)
    assert ok is False
    cmd_args = fake_run.call_args.args[0]
    assert cmd_args[0] == "pbcopy"

    # Integration: _handle_copy_command must surface that failure to the user
    # (no "Copied" message). Use a raising OSC 52 writer so the bool fallback
    # path also fails — mirroring the real-world "no working backend" case.
    cli_obj = _make_cli()
    cli_obj.conversation_history = [{"role": "assistant", "content": "payload"}]

    def _broken_osc52(_seq: str) -> None:
        raise RuntimeError("no terminal sink available")

    with patch.object(cli_obj, "_emit_osc52_to_terminal", _broken_osc52), \
         patch("cli._cprint") as mock_print:
        cli_obj.process_command("/copy")

    printed = " ".join(str(c) for c in mock_print.call_args_list)
    assert "Copied" not in printed
    assert "fail" in printed.lower() or "could not" in printed.lower() or "unavailable" in printed.lower()


def test_write_clipboard_text_macos_returns_false_when_everything_fails(monkeypatch, fake_run):
    """If pbcopy fails AND no OSC 52 writer is provided, return False."""
    _no_ssh(monkeypatch)
    monkeypatch.setattr(clipboard_mod.sys, "platform", "darwin")
    fake_run.side_effect = FileNotFoundError("pbcopy not found")

    ok = clipboard_mod.write_clipboard_text("nope", osc52_writer=None)

    assert ok is False


def test_write_clipboard_text_ssh_prefers_osc52_first(monkeypatch, fake_run):
    """In an SSH session, OSC 52 is preferred over native (which would copy on the wrong host)."""
    _ssh_session(monkeypatch)
    monkeypatch.setattr(clipboard_mod.sys, "platform", "darwin")
    osc52_calls = []

    ok = clipboard_mod.write_clipboard_text(
        "remote", osc52_writer=lambda seq: osc52_calls.append(seq)
    )

    assert ok is True
    assert len(osc52_calls) == 1
    # Native must NOT be called when OSC 52 succeeds in SSH mode
    assert not fake_run.called


def test_write_clipboard_text_linux_wayland(monkeypatch, fake_run):
    """On a local Wayland Linux session, wl-copy should be invoked."""
    _no_ssh(monkeypatch)
    monkeypatch.setattr(clipboard_mod.sys, "platform", "linux")
    monkeypatch.setattr(clipboard_mod, "_is_wsl", lambda: False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("DISPLAY", raising=False)

    ok = clipboard_mod.write_clipboard_text("wld", osc52_writer=None)

    assert ok is True
    cmd_args = fake_run.call_args.args[0]
    assert cmd_args[0] == "wl-copy"


def test_write_clipboard_text_linux_x11_falls_back_to_xsel(monkeypatch, fake_run):
    """If xclip is missing on X11, xsel should be tried."""
    _no_ssh(monkeypatch)
    monkeypatch.setattr(clipboard_mod.sys, "platform", "linux")
    monkeypatch.setattr(clipboard_mod, "_is_wsl", lambda: False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")

    calls: list = []

    def fake_run_impl(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == "xclip":
            raise FileNotFoundError("xclip")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    fake_run.side_effect = fake_run_impl

    ok = clipboard_mod.write_clipboard_text("x11", osc52_writer=None)

    assert ok is True
    assert calls == ["xclip", "xsel"]


def test_write_clipboard_text_wsl_uses_clip_exe(monkeypatch, fake_run):
    """WSL should prefer clip.exe over Linux backends."""
    _no_ssh(monkeypatch)
    monkeypatch.setattr(clipboard_mod.sys, "platform", "linux")
    monkeypatch.setattr(clipboard_mod, "_is_wsl", lambda: True)

    ok = clipboard_mod.write_clipboard_text("wsl", osc52_writer=None)

    assert ok is True
    cmd_args = fake_run.call_args.args[0]
    assert cmd_args[0] == "clip.exe"


def test_write_clipboard_text_windows_uses_clip(monkeypatch, fake_run):
    """Native Windows should invoke clip.exe."""
    _no_ssh(monkeypatch)
    monkeypatch.setattr(clipboard_mod.sys, "platform", "win32")

    ok = clipboard_mod.write_clipboard_text("win", osc52_writer=None)

    assert ok is True
    cmd_args = fake_run.call_args.args[0]
    assert cmd_args[0] in ("clip", "clip.exe")


# ═════════════════════════════════════════════════════════════════════════
# Integration: _handle_copy_command should report failure if no path works
# ═════════════════════════════════════════════════════════════════════════


def test_handle_copy_reports_failure_when_no_clipboard_path_works(monkeypatch):
    """If both native and OSC 52 silently no-op, /copy must NOT claim success.

    This is the original macOS Terminal.app bug: the OSC 52 sequence was
    written to a sink that nothing honored, but the user saw 'Copied'.
    """
    cli_obj = _make_cli()
    cli_obj.conversation_history = [{"role": "assistant", "content": "payload"}]

    # Force the shared writer to report failure (everything silently no-opped).
    monkeypatch.setattr(
        "hermes_cli.clipboard.write_clipboard_text",
        lambda text, *, osc52_writer=None: False,
    )

    with patch("cli._cprint") as mock_print:
        cli_obj.process_command("/copy")

    printed = " ".join(str(c) for c in mock_print.call_args_list)
    # Must not falsely claim success
    assert "Copied" not in printed
    # Must communicate the failure to the user
    assert "fail" in printed.lower() or "could not" in printed.lower() or "unavailable" in printed.lower()


def test_handle_copy_reports_success_when_clipboard_writer_succeeds(monkeypatch):
    """When the clipboard writer returns True, /copy reports success."""
    cli_obj = _make_cli()
    cli_obj.conversation_history = [{"role": "assistant", "content": "payload"}]

    monkeypatch.setattr(
        "hermes_cli.clipboard.write_clipboard_text",
        lambda text, *, osc52_writer=None: True,
    )

    with patch("cli._cprint") as mock_print:
        cli_obj.process_command("/copy")

    printed = " ".join(str(c) for c in mock_print.call_args_list)
    assert "Copied" in printed
