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
    """No venv python → nothing to probe → healthy (never blocks update)."""
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path):
        healthy, detail = cli_main._venv_core_imports_healthy()
    assert healthy is True
    assert detail == ""


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


def _proc(pid: int, exe: str, name: str, cmdline: list[str] | None = None):
    proc = MagicMock()
    proc.info = {"pid": pid, "exe": exe, "name": name, "cmdline": cmdline or []}
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
    assert "--force" in msg
