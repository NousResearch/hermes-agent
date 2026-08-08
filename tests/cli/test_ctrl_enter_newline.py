"""Regression tests for issue #22379 — Ctrl+Enter newline over SSH/WSL.

prompt_toolkit treats c-j (LF) as Enter on POSIX so thin PTYs (docker exec,
some BSD ssh) that send LF for plain Enter still work. But Windows Terminal
(native, WSL, and SSH-forwarded sessions) sends Ctrl+Enter as bare LF — same
byte. Without environment-aware gating, binding c-j to submit means
Ctrl+Enter submits instead of inserting a newline.

These tests pin the gating predicate and the resulting binding behavior.
"""

from __future__ import annotations

import builtins
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


def _hermes_home_with_config(config_text: str) -> Path:
    """Create a temp HERMES_HOME, write config.yaml, point HERMES_HOME at it.

    Returns the temp config path.
    """
    tmp = Path(tempfile.mkdtemp(prefix="hermes-test-"))
    config_path = tmp / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    os.environ["HERMES_HOME"] = str(tmp)
    return config_path


def _clear_stale_env() -> None:
    """Remove env vars that would otherwise trigger preserve-newline."""
    for k in list(os.environ.keys()):
        if k.startswith(("SSH_", "WT")) or k in (
            "WSL_DISTRO_NAME",
            "GHOSTTY_RESOURCES_DIR",
            "GHOSTTY_BIN_DIR",
            "TERM_PROGRAM",
        ):
            os.environ.pop(k, None)


def _no_proc_open():
    """Patch builtins.open so /proc reads fail (WSL fallback probe)."""
    real_open = builtins.open
    def _fake_open(path, *args, **kwargs):
        if "/proc/version" in str(path) or "/proc/sys/kernel/osrelease" in str(path):
            raise OSError("no /proc")
        return real_open(path, *args, **kwargs)
    return patch("builtins.open", side_effect=_fake_open)


def test_native_windows_preserves_newline():
    import cli as cli_mod
    with patch.object(sys, "platform", "win32"):
        assert cli_mod._preserve_ctrl_enter_newline() is True




def test_ssh_tty_alone_preserves_newline():
    import cli as cli_mod
    with patch.object(sys, "platform", "linux"):
        # Strip out anything that might leak truth
        with patch.dict(os.environ, {"SSH_TTY": "/dev/pts/0"}, clear=True):
            assert cli_mod._preserve_ctrl_enter_newline() is True




def test_windows_terminal_session_preserves_newline():
    import cli as cli_mod
    with patch.object(sys, "platform", "linux"):
        with patch.dict(os.environ, {"WT_SESSION": "abc-def"}, clear=True):
            assert cli_mod._preserve_ctrl_enter_newline() is True


def test_ghostty_tmux_session_preserves_ctrl_j_newline():
    """Ghostty-inherited env survives tmux even when TERM_PROGRAM becomes tmux."""
    import cli as cli_mod
    with patch.object(sys, "platform", "linux"):
        with patch.dict(
            os.environ,
            {"TERM": "tmux-256color", "TERM_PROGRAM": "tmux", "GHOSTTY_RESOURCES_DIR": "/usr/share/ghostty"},
            clear=True,
        ):
            assert cli_mod._preserve_ctrl_enter_newline() is True




def test_proc_version_microsoft_marker_preserves_newline():
    """WSL detection via /proc when env vars are scrubbed (sudo etc.)."""
    import cli as cli_mod
    from io import StringIO

    with patch.object(sys, "platform", "linux"):
        with patch.dict(os.environ, {}, clear=True):
            real_open = builtins.open
            def _fake_open(path, *args, **kwargs):
                if "/proc/version" in str(path) or "/proc/sys/kernel/osrelease" in str(path):
                    return StringIO("Linux version 5.15.167.4-microsoft-standard-WSL2")
                return real_open(path, *args, **kwargs)
            with patch("builtins.open", side_effect=_fake_open):
                assert cli_mod._preserve_ctrl_enter_newline() is True


def test_pure_local_linux_does_not_preserve():
    """A bare local Linux TTY (no SSH/WSL/WT/Ghostty) keeps c-j → submit so docker exec
    style Enter-as-LF stays usable."""
    import cli as cli_mod
    # Stub out /proc reads — those are the WSL fallback signal.
    with patch.object(sys, "platform", "linux"):
        with patch.dict(os.environ, {}, clear=True):
            with _no_proc_open():
                assert cli_mod._preserve_ctrl_enter_newline() is False


def test_config_override_preserves_ctrl_j_newline():
    """display.ctrl_enter_newline: true forces newline behavior on local POSIX."""
    _clear_stale_env()
    _hermes_home_with_config("display:\n  ctrl_enter_newline: true\n")

    with patch.object(sys, "platform", "linux"):
        with _no_proc_open():
            if "cli" in sys.modules:
                del sys.modules["cli"]
            import cli as cli_mod
            assert cli_mod._preserve_ctrl_enter_newline() is True


def test_config_override_respects_ignore_user_config():
    """HERMES_IGNORE_USER_CONFIG=1 must ignore a user config enabling the setting."""
    _clear_stale_env()
    _hermes_home_with_config("display:\n  ctrl_enter_newline: true\n")
    os.environ["HERMES_IGNORE_USER_CONFIG"] = "1"

    with patch.object(sys, "platform", "linux"):
        with _no_proc_open():
            if "cli" in sys.modules:
                del sys.modules["cli"]
            import cli as cli_mod
            assert cli_mod._preserve_ctrl_enter_newline() is False


# ---------------------------------------------------------------------------
# install_ctrl_enter_alias() — ANSI sequence mappings for enhanced terminals
# ---------------------------------------------------------------------------




