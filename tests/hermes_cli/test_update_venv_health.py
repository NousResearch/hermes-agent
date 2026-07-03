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


# ---------------------------------------------------------------------------
# _venv_core_imports_healthy
# ---------------------------------------------------------------------------


def test_venv_health_reports_healthy_when_no_venv(tmp_path):
    """No venv python in a DEV checkout → nothing to probe → healthy."""
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path):
        healthy, detail = cli_main._venv_core_imports_healthy()
    assert healthy is True
    assert detail == ""


def test_venv_health_missing_venv_unhealthy_on_managed_install(tmp_path):
    """On a managed install (bootstrap marker) the venv IS the install —
    its absence must be reported unhealthy so the repair lane runs instead
    of 'Already up to date!'."""
    (tmp_path / ".hermes-bootstrap-complete").write_text("done")
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path):
        healthy, detail = cli_main._venv_core_imports_healthy()
    assert healthy is False
    assert "venv python missing" in detail


def test_venv_health_missing_venv_unhealthy_with_interrupted_marker(tmp_path):
    """An interrupted-update breadcrumb also flips missing-venv to unhealthy."""
    (tmp_path / ".update-incomplete").write_text("started=1\npid=1\n")
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path):
        healthy, detail = cli_main._venv_core_imports_healthy()
    assert healthy is False
    assert "venv python missing" in detail


def _fake_venv_python(tmp_path, *, windows: bool = False):
    bin_dir = tmp_path / "venv" / ("Scripts" if windows else "bin")
    bin_dir.mkdir(parents=True)
    py = bin_dir / ("python.exe" if windows else "python")
    py.write_bytes(b"")
    return py


def test_venv_health_reports_missing_imports(tmp_path):
    """Probe output lines are surfaced as the unhealthy detail."""
    _fake_venv_python(tmp_path)

    fake = SimpleNamespace(
        returncode=0,
        stdout="fastapi: No module named 'annotated_doc'\n",
        stderr="",
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        cli_main.subprocess, "run", return_value=fake
    ):
        healthy, detail = cli_main._venv_core_imports_healthy()

    assert healthy is False
    assert "annotated_doc" in detail


def test_venv_health_healthy_when_probe_clean(tmp_path):
    _fake_venv_python(tmp_path)
    fake = SimpleNamespace(returncode=0, stdout="", stderr="")
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        cli_main.subprocess, "run", return_value=fake
    ):
        healthy, detail = cli_main._venv_core_imports_healthy()
    assert healthy is True


def test_venv_health_broken_interpreter_is_unhealthy(tmp_path):
    """Nonzero exit with no module list = interpreter itself is broken."""
    _fake_venv_python(tmp_path)
    fake = SimpleNamespace(returncode=1, stdout="", stderr="Fatal Python error: init failed\n")
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        cli_main.subprocess, "run", return_value=fake
    ):
        healthy, detail = cli_main._venv_core_imports_healthy()
    assert healthy is False
    assert "Fatal Python error" in detail


def test_venv_health_probe_failure_reports_healthy(tmp_path):
    """A probe that can't run must NOT force needless reinstalls."""
    _fake_venv_python(tmp_path)
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        cli_main.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(cmd="python", timeout=60),
    ):
        healthy, _detail = cli_main._venv_core_imports_healthy()
    assert healthy is True


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


def test_detect_venv_python_off_windows_is_empty():
    with patch.object(cli_main, "_is_windows", return_value=False):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_finds_backend(_winp, tmp_path):
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    other_py = "C:\\Python311\\python.exe"

    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(101, venv_py, "python.exe", ["python.exe", "-m", "hermes_cli.main", "serve"]),
                _proc(102, other_py, "python.exe", ["python.exe", "somescript.py"]),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [m[0] for m in matches] == [101]
    assert "serve" in matches[0][2]


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


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_no_psutil_is_empty(_winp, tmp_path):
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": None}
    ):
        assert cli_main._detect_venv_python_processes() == []


def test_format_venv_holders_message_flags_desktop_backend(tmp_path):
    matches = [
        (101, "python.exe", "python.exe -m hermes_cli.main serve --host 127.0.0.1"),
        (102, "pythonw.exe", "pythonw.exe -m hermes_cli.main gateway run"),
    ]
    msg = cli_main._format_venv_python_holders_message(matches)
    assert "101" in msg
    assert "desktop app" in msg.lower()
    assert "gateway" in msg
    assert "hermes update" in msg
    assert "--force-venv" in msg


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_catches_outside_venv_trampoline(_winp, tmp_path):
    """uv/base-interpreter trampoline: exe OUTSIDE the venv, but the cmdline
    clearly runs Hermes from this install → must still be flagged as a holder
    (it imports from the venv and holds its .pyd files)."""
    base_py = "C:\\Python311\\python.exe"
    venv_path = str(tmp_path / "venv" / "Scripts" / "python.exe")

    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                # cmdline references the venv path directly
                _proc(201, base_py, "python.exe", [base_py, venv_path, "-m", "x"]),
                # `-m hermes_cli.main serve` with the install root as cwd
                _proc(
                    202,
                    base_py,
                    "python.exe",
                    [base_py, "-m", "hermes_cli.main", "serve"],
                    cwd=str(tmp_path),
                ),
                # unrelated base-interpreter python → NOT a holder
                _proc(203, base_py, "python.exe", [base_py, "somescript.py"], cwd="C:\\other"),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert sorted(m[0] for m in matches) == [201, 202]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_hermes_cli_cmdline_outside_install_not_matched(_winp, tmp_path):
    """A hermes_cli.main process belonging to a DIFFERENT install (neither
    install root in cmdline nor cwd under it) must not be flagged."""
    base_py = "C:\\Python311\\python.exe"
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(
                    301,
                    base_py,
                    "python.exe",
                    [base_py, "-m", "hermes_cli.main", "serve"],
                    cwd="C:\\other-install",
                ),
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
