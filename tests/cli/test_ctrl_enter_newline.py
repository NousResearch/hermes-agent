"""Regression tests for issue #22379 — Ctrl+Enter newline over SSH/WSL.

prompt_toolkit treats c-j (LF) as Enter on POSIX so thin PTYs (docker exec,
some BSD ssh) that send LF for plain Enter still work. But Windows Terminal
(native, WSL, and SSH-forwarded sessions) sends Ctrl+Enter as bare LF — same
byte. Without environment-aware gating, binding c-j to submit means
Ctrl+Enter submits instead of inserting a newline.

These tests pin the gating predicate and the resulting binding behavior.

``_preserve_ctrl_enter_newline()`` short-circuits to True on native Windows
before it ever looks at the environment, so the env-driven cases below are
POSIX assertions and run on the Linux job. The native-Windows short-circuit
is marked ``windows_only`` and asserted on the real host — patching
``sys.platform`` to ``"win32"`` here would only re-assert the literal in the
``if``, on an interpreter where none of the Windows terminal behaviour it
exists for is present.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

import sys


class FakeKeyBindings:
    def __init__(self):
        self.bound = []

    def add(self, *keys, **_kwargs):
        def _decorator(handler):
            self.bound.append(keys)
            return handler

        return _decorator


def _bind_submit_keys_for_local_linux(cli_mod, *, multiline_shortcuts_enabled):
    with patch.object(sys, "platform", "linux"):
        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.open", side_effect=OSError("no /proc")):
                kb = FakeKeyBindings()
                cli_mod._bind_prompt_submit_keys(
                    kb,
                    lambda _event: None,
                    multiline_shortcuts_enabled=multiline_shortcuts_enabled,
                )
                return kb


@pytest.mark.windows_only
def test_native_windows_preserves_newline():
    import cli as cli_mod

    assert cli_mod._preserve_ctrl_enter_newline() is True


def test_ssh_tty_alone_preserves_newline():
    import cli as cli_mod
    # Strip out anything that might leak truth
    with patch.dict(os.environ, {"SSH_TTY": "/dev/pts/0"}, clear=True):
        assert cli_mod._preserve_ctrl_enter_newline() is True


def test_windows_terminal_session_preserves_newline():
    import cli as cli_mod
    with patch.dict(os.environ, {"WT_SESSION": "abc-def"}, clear=True):
        assert cli_mod._preserve_ctrl_enter_newline() is True


def test_ghostty_tmux_session_preserves_ctrl_j_newline():
    """Ghostty-inherited env survives tmux even when TERM_PROGRAM becomes tmux."""
    import cli as cli_mod
    with patch.dict(
        os.environ,
        {"TERM": "tmux-256color", "TERM_PROGRAM": "tmux", "GHOSTTY_RESOURCES_DIR": "/usr/share/ghostty"},
        clear=True,
    ):
        assert cli_mod._preserve_ctrl_enter_newline() is True


def test_cli_multiline_shortcuts_default_on():
    """Hermes should default to the common harness behavior: Ctrl+J newline.

    Claude Code documents Ctrl+J as a no-setup newline shortcut, OpenCode's
    default input_newline includes ctrl+j, and Codex exposes Ctrl+J/keymap
    newline behavior. Keep Hermes aligned unless the user opts out.
    """
    import cli as cli_mod

    assert cli_mod._cli_multiline_shortcuts_enabled({"display": {}}) is True


def test_cli_multiline_shortcuts_can_be_disabled():
    import cli as cli_mod

    assert cli_mod._cli_multiline_shortcuts_enabled(
        {"display": {"cli_multiline_shortcuts": False}}
    ) is False


def test_ctrl_j_is_not_submit_when_multiline_shortcuts_enabled():
    """With the default setting, c-j is reserved for the newline handler.

    This fixes local terminals like iTerm2 where Ctrl+J reaches prompt_toolkit
    as c-j but the legacy POSIX fallback bound it to submit.
    """
    import cli as cli_mod

    kb = _bind_submit_keys_for_local_linux(
        cli_mod,
        multiline_shortcuts_enabled=True,
    )

    assert ("enter",) in kb.bound
    assert ("c-j",) not in kb.bound


def test_ctrl_j_legacy_submit_when_multiline_shortcuts_disabled():
    """Users can opt out to preserve Enter-as-LF submit fallback on odd PTYs."""
    import cli as cli_mod

    kb = _bind_submit_keys_for_local_linux(
        cli_mod,
        multiline_shortcuts_enabled=False,
    )

    assert ("enter",) in kb.bound
    assert ("c-j",) in kb.bound


def test_backslash_enter_continuation_replaces_marker_with_newline():
    import cli as cli_mod

    assert cli_mod._apply_backslash_line_continuation("first line\\") == "first line\n"
    assert cli_mod._apply_backslash_line_continuation("first line\\   ") == "first line\n"


def test_iterm_is_allowlisted_for_extended_enter_keys():
    """iTerm2 needs the app to request extended keys before Shift+Enter is distinct."""
    import cli as cli_mod

    assert cli_mod._terminal_supports_extended_enter_keys({"TERM_PROGRAM": "iTerm.app"}) is True


def test_unknown_terminal_does_not_enable_extended_enter_keys():
    import cli as cli_mod

    assert cli_mod._terminal_supports_extended_enter_keys({"TERM_PROGRAM": "unknown"}) is False


def test_alacritty_term_program_is_allowlisted_for_extended_enter_keys():
    """Alacritty implements the Kitty keyboard protocol unconditionally
    (`kitty_keyboard: true` in its own config derivation) but Hermes must
    still ask it to enable extended key reporting to receive distinct
    Shift+Enter sequences."""
    import cli as cli_mod

    assert cli_mod._terminal_supports_extended_enter_keys({"TERM_PROGRAM": "alacritty"}) is True


def test_alacritty_term_var_is_allowlisted_for_extended_enter_keys():
    """Alacritty does not set TERM_PROGRAM by default — only TERM=alacritty
    and the ALACRITTY_WINDOW_ID / ALACRITTY_SOCKET env vars it exports."""
    import cli as cli_mod

    assert cli_mod._terminal_supports_extended_enter_keys({"TERM": "alacritty"}) is True
    assert (
        cli_mod._terminal_supports_extended_enter_keys(
            {"TERM": "xterm-256color", "ALACRITTY_WINDOW_ID": "94212259512240"}
        )
        is True
    )
    assert (
        cli_mod._terminal_supports_extended_enter_keys(
            {"TERM": "xterm-256color", "ALACRITTY_SOCKET": "/run/user/1000/Alacritty-wayland-1.sock"}
        )
        is True
    )


def test_alacritty_is_not_treated_as_ghostty():
    """Alacritty must get the full dual-protocol push, not the Ghostty-only
    modifyOtherKeys exception."""
    import cli as cli_mod

    assert cli_mod._is_ghostty_terminal({"TERM": "alacritty", "ALACRITTY_WINDOW_ID": "1"}) is False


# ---------------------------------------------------------------------------
# Ghostty: must push ONLY modifyOtherKeys, not the Kitty keyboard protocol —
# see cli._is_ghostty_terminal for the full rationale (#87630).
# ---------------------------------------------------------------------------
class _FakeOutput:
    """Minimal output object with write_raw + flush for _enable_extended_enter_keys."""
    def __init__(self):
        self.written = b""
    def write_raw(self, data):
        self.written += data.encode() if isinstance(data, str) else data
    def flush(self):
        pass


def test_ghostty_uses_modify_other_keys_only():
    """Ghostty must NOT push the Kitty keyboard protocol (CSI >1u)."""
    import cli as cli_mod

    out = _FakeOutput()
    result = cli_mod._enable_extended_enter_keys(
        output=out,
        env={"TERM_PROGRAM": "ghostty", "TERM": "xterm-ghostty"},
    )
    assert result is True
    # Must contain modifyOtherKeys push ...
    assert b"\x1b[>4;2m" in out.written
    # ... but NOT the Kitty protocol push.
    assert b"\x1b[>1u" not in out.written


def test_ghostty_via_term_var_uses_modify_other_keys_only():
    """xterm-ghostty TERM (without TERM_PROGRAM) also skips Kitty protocol."""
    import cli as cli_mod

    out = _FakeOutput()
    result = cli_mod._enable_extended_enter_keys(
        output=out,
        env={"TERM": "xterm-ghostty"},
    )
    assert result is True
    assert b"\x1b[>4;2m" in out.written
    assert b"\x1b[>1u" not in out.written


def test_non_ghostty_terminals_still_push_kitty_protocol():
    """iTerm2 and others still get the full dual-protocol push."""
    import cli as cli_mod

    out = _FakeOutput()
    result = cli_mod._enable_extended_enter_keys(
        output=out,
        env={"TERM_PROGRAM": "iTerm.app", "TERM": "xterm-256color"},
    )
    assert result is True
    assert b"\x1b[>1u" in out.written
    assert b"\x1b[>4;2m" in out.written


@pytest.mark.linux_only
def test_proc_version_microsoft_marker_preserves_newline():
    """WSL detection via /proc when env vars are scrubbed (sudo etc.).

    ``linux_only``: the fallback reads ``/proc/version`` — a Linux-only
    interface, and the WSL kernels it sniffs for are Linux kernels.
    """
    import cli as cli_mod
    from io import StringIO
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


def test_is_ghostty_terminal_detection_paths():
    """_is_ghostty_terminal matches exactly the two allowlist conditions."""
    import cli as cli_mod

    assert cli_mod._is_ghostty_terminal({"TERM_PROGRAM": "ghostty"}) is True
    assert cli_mod._is_ghostty_terminal({"TERM": "xterm-ghostty"}) is True
    assert cli_mod._is_ghostty_terminal({"TERM": "XTERM-GHOSTTY"}) is True
    assert cli_mod._is_ghostty_terminal({"TERM_PROGRAM": "iTerm.app"}) is False
    assert cli_mod._is_ghostty_terminal({}) is False
