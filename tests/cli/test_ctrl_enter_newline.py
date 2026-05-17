"""Regression tests for issue #22379 — Ctrl+Enter newline over SSH/WSL.

prompt_toolkit treats c-j (LF) as Enter on POSIX so thin PTYs (docker exec,
some BSD ssh) that send LF for plain Enter still work. But Windows Terminal
(native, WSL, and SSH-forwarded sessions) sends Ctrl+Enter as bare LF — same
byte. Without environment-aware gating, binding c-j to submit means
Ctrl+Enter submits instead of inserting a newline.

These tests pin the gating predicate and the resulting binding behavior.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch


def test_native_windows_preserves_newline():
    import cli as cli_mod
    with patch.object(sys, "platform", "win32"):
        assert cli_mod._preserve_ctrl_enter_newline() is True


def test_ssh_session_preserves_newline_on_linux():
    import cli as cli_mod
    with patch.object(sys, "platform", "linux"):
        with patch.dict(os.environ, {"SSH_CONNECTION": "1.2.3.4 5 6.7.8.9 22"}, clear=False):
            assert cli_mod._preserve_ctrl_enter_newline() is True


def test_ssh_tty_alone_preserves_newline():
    import cli as cli_mod
    with patch.object(sys, "platform", "linux"):
        # Strip out anything that might leak truth
        with patch.dict(os.environ, {"SSH_TTY": "/dev/pts/0"}, clear=True):
            assert cli_mod._preserve_ctrl_enter_newline() is True


def test_wsl_distro_name_preserves_newline():
    import cli as cli_mod
    with patch.object(sys, "platform", "linux"):
        with patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu-Microsoft"}, clear=True):
            assert cli_mod._preserve_ctrl_enter_newline() is True


def test_windows_terminal_session_preserves_newline():
    import cli as cli_mod
    with patch.object(sys, "platform", "linux"):
        with patch.dict(os.environ, {"WT_SESSION": "abc-def"}, clear=True):
            assert cli_mod._preserve_ctrl_enter_newline() is True


def test_pure_local_linux_does_not_preserve():
    """A bare local Linux TTY (no SSH/WSL/WT) keeps c-j → submit so docker exec
    style Enter-as-LF stays usable."""
    import cli as cli_mod
    # Stub out /proc reads — those are the WSL fallback signal.
    with patch.object(sys, "platform", "linux"):
        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.open", side_effect=OSError("no /proc")):
                assert cli_mod._preserve_ctrl_enter_newline() is False


def test_proc_version_microsoft_marker_preserves_newline():
    """WSL detection via /proc when env vars are scrubbed (sudo etc.)."""
    import cli as cli_mod
    from io import StringIO
    with patch.object(sys, "platform", "linux"):
        with patch.dict(os.environ, {}, clear=True):
            real_open = open
            def _fake_open(path, *args, **kwargs):
                if "/proc/version" in str(path) or "/proc/sys/kernel/osrelease" in str(path):
                    return StringIO("Linux version 5.15.167.4-microsoft-standard-WSL2")
                return real_open(path, *args, **kwargs)
            with patch("builtins.open", side_effect=_fake_open):
                assert cli_mod._preserve_ctrl_enter_newline() is True


# ---------------------------------------------------------------------------
# install_ctrl_enter_alias() — ANSI sequence mappings for enhanced terminals
# ---------------------------------------------------------------------------


def test_install_ctrl_enter_alias_maps_csi_u_sequences():
    """Kitty / xterm modifyOtherKeys / mintty Ctrl+Enter sequences alias to
    Alt+Enter (Escape, ControlM) so the existing newline handler fires."""
    from hermes_cli.pt_input_extras import install_ctrl_enter_alias
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.keys import Keys

    install_ctrl_enter_alias()
    alt_enter = (Keys.Escape, Keys.ControlM)
    for seq in ("\x1b[13;5u", "\x1b[27;5;13~", "\x1b[27;5;13u"):
        assert ANSI_SEQUENCES.get(seq) == alt_enter, (
            f"Ctrl+Enter sequence {seq!r} not mapped to Alt+Enter tuple"
        )


def test_install_ctrl_enter_alias_idempotent():
    """Running it twice doesn't double-count or break."""
    from hermes_cli.pt_input_extras import install_ctrl_enter_alias
    install_ctrl_enter_alias()
    second = install_ctrl_enter_alias()
    assert second == 0  # no further changes after first install
