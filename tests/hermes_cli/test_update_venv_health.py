"""Tests for the Windows half-updated-venv hardening (July 2026 incident).

Covers three additions to ``hermes update``:

1. ``_venv_core_imports_healthy`` — the venv health probe that lets an
   "Already up to date" checkout still repair a broken dependency install.
2. ``_detect_venv_python_processes`` — the venv-interpreter process guard
   that refuses to mutate the venv while a desktop backend / stray python
   holds .pyd files mapped.
3. The commit_count == 0 repair branch wiring in ``_cmd_update_impl``.

All Windows-specific paths are exercised via ``_is_windows`` patching so
they run on any host (same approach as test_update_concurrent_quarantine).
"""

from __future__ import annotations

import subprocess
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import main as cli_main


def _fake_ancestor(pid: int, cmdline: list[str], name: str = "python.exe"):
    proc = MagicMock()
    proc.pid = pid
    proc.cmdline.return_value = cmdline
    proc.name.return_value = name
    return proc


@pytest.mark.parametrize(
    "cmdline, expected_kind",
    [
        (
            ["python.exe", "-m", "tui_gateway.slash_worker", "--session-key", "abc"],
            "Desktop slash worker",
        ),
        (
            ["pythonw.exe", "-m", "hermes_cli.main", "serve", "--port", "9119"],
            "Desktop backend",
        ),
        (
            [
                "pythonw.exe",
                "-m",
                "hermes_cli.main",
                "--profile",
                "worker",
                "serve",
                "--port",
                "9119",
            ],
            "Desktop backend",
        ),
        (
            ["hermes.exe", "--profile", "worker", "serve", "--port", "9119"],
            "Desktop backend",
        ),
    ],
)
def test_detect_active_update_ancestor(cmdline, expected_kind):
    me = MagicMock()
    me.parents.return_value = [_fake_ancestor(555, cmdline)]
    fake_psutil = types.SimpleNamespace(Process=lambda: me)

    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_active_update_ancestor()

    assert result == (555, "python.exe", expected_kind)


def test_detect_active_update_ancestor_ignores_unrelated_parent():
    me = MagicMock()
    me.parents.return_value = [
        _fake_ancestor(
            555,
            ["python.exe", "script.py", "--note", "hermes_cli.main serve"],
        ),
        _fake_ancestor(
            556,
            ["hermes.exe", "--profile", "serve", "dashboard", "--no-open"],
        ),
    ]
    fake_psutil = types.SimpleNamespace(Process=lambda: me)

    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        assert cli_main._detect_active_update_ancestor() is None


def test_active_update_ancestor_guard_precedes_backup(capsys):
    args = _update_args(force=False)
    backup = MagicMock()

    with patch.object(
        cli_main,
        "_detect_active_update_ancestor",
        return_value=(555, "python.exe", "Desktop slash worker"),
    ), patch.object(cli_main, "_run_pre_update_backup", backup):
        with pytest.raises(SystemExit, match="2"):
            cli_main._cmd_update_impl(args, gateway_mode=False)

    backup.assert_not_called()
    output = capsys.readouterr().out
    assert "active Desktop session or worker" in output
    assert "hermes update --force" in output
    assert "stopped or interrupted" in output


def test_force_bypasses_active_update_ancestor_guard():
    args = _update_args(force=True)

    class _PastGuard(Exception):
        pass

    with patch.object(
        cli_main,
        "_detect_active_update_ancestor",
        return_value=(555, "python.exe", "Desktop slash worker"),
    ), patch.object(
        cli_main, "_run_pre_update_backup", side_effect=_PastGuard
    ):
        with pytest.raises(_PastGuard):
            cli_main._cmd_update_impl(args, gateway_mode=False)


# ---------------------------------------------------------------------------
# _venv_core_imports_healthy
# ---------------------------------------------------------------------------




def _fake_venv_python(tmp_path, *, windows: bool = False):
    bin_dir = tmp_path / "venv" / ("Scripts" if windows else "bin")
    bin_dir.mkdir(parents=True)
    py = bin_dir / ("python.exe" if windows else "python")
    py.write_bytes(b"")
    return py




# ---------------------------------------------------------------------------
# _detect_venv_python_processes
# ---------------------------------------------------------------------------


def _proc(pid: int, exe: str, name: str, cmdline: list[str] | None = None, cwd: str = ""):
    proc = MagicMock()
    proc.info = {
        "pid": pid,
        "exe": exe,
        "name": name,
        "cmdline": cmdline or [],
        "cwd": cwd,
    }
    return proc




@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_excludes_self_and_ancestors(_winp, tmp_path):
    import os as _os

    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    parent = MagicMock()
    parent.pid = 555
    me = MagicMock()
    me.parents.return_value = [parent]
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(_os.getpid(), venv_py, "python.exe"),
                _proc(555, venv_py, "hermes.exe"),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []




# ---------------------------------------------------------------------------
# --force vs --force-venv gating of the venv-holder guard
# ---------------------------------------------------------------------------


def _update_args(**overrides):
    defaults = dict(
        gateway=False,
        check=False,
        no_backup=True,
        backup=False,
        yes=True,
        branch=None,
        force=False,
        force_venv=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _run_update_until_guard(args):
    """Drive _cmd_update_impl just far enough to hit the venv-holder guard.

    Everything before the guard is stubbed; the guard firing is observed via
    SystemExit(2). The first statement AFTER the guard is
    ``git_dir = PROJECT_ROOT / ".git"`` — a PROJECT_ROOT sentinel whose
    ``__truediv__`` raises marks 'guard passed'."""

    class _PastGuard(Exception):
        pass

    class _RootSentinel:
        def __truediv__(self, _other):
            raise _PastGuard

    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "_detect_active_update_ancestor", return_value=None
    ), patch.object(
        cli_main, "_venv_scripts_dir", return_value=None
    ), patch.object(cli_main, "_run_pre_update_backup"), patch.object(
        cli_main, "_pause_windows_gateways_for_update", return_value=None
    ), patch.object(
        cli_main, "_resume_windows_gateways_after_update"
    ), patch.object(
        cli_main,
        "_detect_venv_python_processes",
        return_value=[(101, "python.exe", "python.exe -m hermes_cli.main serve")],
    ), patch.object(
        cli_main, "PROJECT_ROOT", _RootSentinel()
    ):
        try:
            cli_main._cmd_update_impl(args, gateway_mode=False)
        except _PastGuard:
            return "past_guard"
        except SystemExit as exc:
            return f"exit_{exc.code}"
    return "returned"


@pytest.mark.parametrize(
    "force,force_venv,expected",
    [
        (False, False, "exit_2"),   # guard fires
        (True, False, "exit_2"),    # plain --force does NOT bypass the venv guard
        (False, True, "past_guard"),  # --force-venv is the explicit escape hatch
        (True, True, "past_guard"),
    ],
)
def test_venv_holder_guard_force_semantics(force, force_venv, expected, capsys):
    result = _run_update_until_guard(_update_args(force=force, force_venv=force_venv))
    assert result == expected, capsys.readouterr().out
