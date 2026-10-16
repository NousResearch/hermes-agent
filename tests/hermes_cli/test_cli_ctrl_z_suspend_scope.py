"""Tests for Ctrl+Z suspend scope (#83006).

The CLI's Ctrl+Z binding used to suspend the *entire process group* via
``os.kill(0, SIGTSTP)``, which also stopped every background job sharing
the group (long-running terminal/OCR tasks spawned from the session) and
turned a stray 0x1A byte in pasted input into an apparent crash. The fix
signals only the current process, matching shell job-control semantics.
"""
import signal
import sys
import types
from unittest.mock import patch

import pytest

from hermes_cli.cli_tui_mixin import CLITuiMixin

# POSIX-only: signal.SIGTSTP and the Ctrl+Z suspend path do not exist on Windows.
pytestmark = pytest.mark.platforms("posix")


def test_suspend_targets_only_current_process():
    """Ctrl+Z must signal the current pid, never the process group (0)."""
    from hermes_cli.cli_suspend import _suspend_cli_process

    with patch("os.kill") as mock_kill, patch("os.getpid", return_value=4242):
        _suspend_cli_process()
    mock_kill.assert_called_once_with(4242, signal.SIGTSTP)
    called_pids = [call.args[0] for call in mock_kill.call_args_list]
    assert 0 not in called_pids, (
        "Ctrl+Z must not signal the whole process group (os.kill(0, ...))"
    )


def test_ctrl_z_binding_routes_through_the_scoped_suspend(monkeypatch):
    """The real c-z handler must signal only the current pid, never group 0."""
    calls = []
    monkeypatch.setattr("os.getpid", lambda: 4242)
    monkeypatch.setattr("os.kill", lambda pid, sig: calls.append((pid, sig)))
    monkeypatch.setattr("os.write", lambda fd, data: None)
    monkeypatch.setattr("prompt_toolkit.application.run_in_terminal", lambda fn: fn())

    class _Skin:
        def get_branding(self, key, default=None):
            return "Hermes Agent"

    from hermes_cli import skin_engine
    monkeypatch.setattr(skin_engine, "get_active_skin", lambda: _Skin())

    # ``from cli import ...`` pulls the whole CLI module; the handler only
    # needs the print helpers, so stand in a light stub for this test
    # instead of importing the CLI module itself.
    fake_cli = types.ModuleType("cli")
    fake_cli._DIM = ""
    fake_cli._RST = ""
    fake_cli._cprint = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    CLITuiMixin()._tui_handle_ctrl_z(None)

    assert calls == [(4242, signal.SIGTSTP)], (
        f"c-z must signal only the current pid, never the group; got {calls}"
    )
