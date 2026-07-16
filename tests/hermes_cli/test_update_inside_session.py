"""Tests for issue #65585 — ``hermes update`` should detect it's running
inside an active Hermes session on Windows and fail fast with a clear
message instead of proceeding to a confusing mid-update failure.

These tests force ``_is_windows`` to return ``True`` via patching so the
Windows-specific code paths can be exercised on any host.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import main as cli_main


# --------------------------------------------------------------------------- #
# _detect_running_inside_hermes_session
# --------------------------------------------------------------------------- #


def _make_ancestor_proc(pid: int, exe: str):
    """Build a duck-typed psutil Process stand-in for an ancestor."""
    proc = MagicMock()
    proc.pid = pid
    proc.exe.return_value = exe
    return proc


@patch.object(cli_main, "_is_windows", return_value=True)
def test_inside_session_detected_when_great_grandparent_is_shim(_winp, tmp_path):
    """Grandparent hermes.exe → update is inside a session.

    Topology: hermes.exe (session, PID S) → python.exe (agent, PID A)
              → hermes.exe (update launcher, PID L) → python.exe (us, PID me)

    The parent (L) is the launcher for *this* update — skip it.
    The grandparent (A) is python.exe — not a shim.
    The great-grandparent (S) is the session hermes.exe — a shim → detected.
    """
    scripts_dir = tmp_path
    shim = scripts_dir / "hermes.exe"
    shim.write_bytes(b"")
    me = os.getpid()
    launcher_pid = me + 100
    agent_pid = me + 200
    session_pid = me + 300

    ancestors = [
        _make_ancestor_proc(launcher_pid, str(shim)),         # parent — skip
        _make_ancestor_proc(agent_pid, r"C:\Python\python.exe"),  # grandparent
        _make_ancestor_proc(session_pid, str(shim)),          # great-grandparent — shim!
    ]

    fake_current = MagicMock()
    fake_current.parents.return_value = ancestors
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid=None: fake_current,
        process_iter=lambda attrs: iter([]),
    )
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_running_inside_hermes_session(scripts_dir)

    assert result is True


@patch.object(cli_main, "_is_windows", return_value=True)
def test_not_inside_session_when_only_launcher_is_shim(_winp, tmp_path):
    """Only the parent launcher is a shim → user invoked from separate terminal.

    Topology: bash.exe (PID B) → hermes.exe (launcher, PID L) → python.exe (us)

    The parent (L) is the launcher — skip it.
    The grandparent (B) is bash — not a shim → not inside a session.
    """
    scripts_dir = tmp_path
    shim = scripts_dir / "hermes.exe"
    shim.write_bytes(b"")
    me = os.getpid()
    launcher_pid = me + 100
    bash_pid = me + 200

    ancestors = [
        _make_ancestor_proc(launcher_pid, str(shim)),         # parent — skip
        _make_ancestor_proc(bash_pid, r"C:\Windows\System32\bash.exe"),  # not a shim
    ]

    fake_current = MagicMock()
    fake_current.parents.return_value = ancestors
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid=None: fake_current,
        process_iter=lambda attrs: iter([]),
    )
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_running_inside_hermes_session(scripts_dir)

    assert result is False


@patch.object(cli_main, "_is_windows", return_value=True)
def test_not_inside_session_when_no_shim_ancestors(_winp, tmp_path):
    """No ancestors are shims at all → separate terminal, not inside session."""
    scripts_dir = tmp_path
    shim = scripts_dir / "hermes.exe"
    shim.write_bytes(b"")
    me = os.getpid()

    ancestors = [
        _make_ancestor_proc(me + 1, r"C:\Windows\System32\cmd.exe"),
        _make_ancestor_proc(me + 2, r"C:\Windows\System32\explorer.exe"),
    ]

    fake_current = MagicMock()
    fake_current.parents.return_value = ancestors
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid=None: fake_current,
        process_iter=lambda attrs: iter([]),
    )
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_running_inside_hermes_session(scripts_dir)

    assert result is False


@patch.object(cli_main, "_is_windows", return_value=False)
def test_inside_session_check_is_noop_off_windows(_winp, tmp_path):
    """Off Windows, the check is a no-op (returns False)."""
    assert cli_main._detect_running_inside_hermes_session(tmp_path) is False


@patch.object(cli_main, "_is_windows", return_value=True)
def test_inside_session_no_psutil_returns_false(_winp, tmp_path):
    """Without psutil, the check returns False (best-effort, never raises)."""
    scripts_dir = tmp_path
    (scripts_dir / "hermes.exe").write_bytes(b"")

    with patch.dict(sys.modules, {"psutil": None}):
        result = cli_main._detect_running_inside_hermes_session(scripts_dir)

    assert result is False


@patch.object(cli_main, "_is_windows", return_value=True)
def test_inside_session_no_shims_returns_false(_winp, tmp_path):
    """With no shim files present, the check returns False."""
    # Don't create any .exe files
    with patch.dict(sys.modules, {"psutil": types.SimpleNamespace()}):
        result = cli_main._detect_running_inside_hermes_session(tmp_path)

    assert result is False


@patch.object(cli_main, "_is_windows", return_value=True)
def test_inside_session_parents_raises_returns_false(_winp, tmp_path):
    """If psutil.Process().parents() raises, the check returns False."""
    scripts_dir = tmp_path
    (scripts_dir / "hermes.exe").write_bytes(b"")

    fake_current = MagicMock()
    fake_current.parents.side_effect = OSError("access denied")
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid=None: fake_current,
    )
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_running_inside_hermes_session(scripts_dir)

    assert result is False


@patch.object(cli_main, "_is_windows", return_value=True)
def test_inside_session_ancestor_exe_unreadable_does_not_crash(_winp, tmp_path):
    """If an ancestor's .exe() raises, that ancestor is skipped (no crash)."""
    scripts_dir = tmp_path
    shim = scripts_dir / "hermes.exe"
    shim.write_bytes(b"")
    me = os.getpid()

    bad_ancestor = MagicMock()
    bad_ancestor.exe.side_effect = OSError("access denied")

    good_ancestor = _make_ancestor_proc(me + 300, str(shim))

    ancestors = [
        _make_ancestor_proc(me + 100, str(shim)),  # parent — skip
        bad_ancestor,                               # exe raises — skip
        good_ancestor,                              # shim — detected
    ]

    fake_current = MagicMock()
    fake_current.parents.return_value = ancestors
    fake_psutil = types.SimpleNamespace(
        Process=lambda pid=None: fake_current,
    )
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_running_inside_hermes_session(scripts_dir)

    assert result is True


# --------------------------------------------------------------------------- #
# _format_inside_hermes_session_message
# --------------------------------------------------------------------------- #


def test_format_message_explains_issue_and_remediation(tmp_path):
    """The message should explain the problem and how to fix it."""
    msg = cli_main._format_inside_hermes_session_message(tmp_path)

    # Explains the problem
    assert "inside" in msg.lower()
    assert "file lock" in msg.lower() or "lock" in msg.lower()

    # Names the shim
    assert str(tmp_path / "hermes.exe") in msg

    # Provides remediation steps
    assert "/quit" in msg or "exit" in msg.lower()
    assert "hermes update" in msg
    assert "separate terminal" in msg.lower()

    # Mentions --force escape hatch
    assert "--force" in msg


def test_format_message_is_user_friendly(tmp_path):
    """The message should not contain raw WinError or technical jargon."""
    msg = cli_main._format_inside_hermes_session_message(tmp_path)

    # No raw error codes
    assert "WinError" not in msg
    assert "os error" not in msg
    # No traceback-style content
    assert "Traceback" not in msg


# --------------------------------------------------------------------------- #
# _cmd_update_impl integration
# --------------------------------------------------------------------------- #


@patch.object(cli_main, "_is_windows", return_value=True)
def test_update_impl_exits_before_mutation_inside_session(
    _winp, tmp_path, capsys, monkeypatch
):
    """The command guard should fail with code 2 before backup or mutation."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(cli_main, "_venv_scripts_dir", lambda: tmp_path)
    monkeypatch.setattr(
        cli_main, "_detect_running_inside_hermes_session", lambda _path: True
    )
    backup = MagicMock(side_effect=AssertionError("backup must not run"))
    monkeypatch.setattr(cli_main, "_run_pre_update_backup", backup)

    with pytest.raises(SystemExit) as exc_info:
        cli_main._cmd_update_impl(types.SimpleNamespace(force=False, yes=False), False)

    assert exc_info.value.code == 2
    assert "inside an active Hermes session" in capsys.readouterr().out
    backup.assert_not_called()


@patch.object(cli_main, "_is_windows", return_value=True)
def test_update_impl_force_bypasses_inside_session_guard(_winp, monkeypatch):
    """--force skips the shim guard and reaches the first mutation boundary."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    detect = MagicMock(side_effect=AssertionError("guard must be bypassed"))
    monkeypatch.setattr(cli_main, "_detect_running_inside_hermes_session", detect)
    monkeypatch.setattr(
        cli_main,
        "_run_pre_update_backup",
        MagicMock(side_effect=RuntimeError("past shim guard")),
    )

    with pytest.raises(RuntimeError, match="past shim guard"):
        cli_main._cmd_update_impl(types.SimpleNamespace(force=True, yes=False), False)

    detect.assert_not_called()