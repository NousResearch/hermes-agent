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

import os
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


def _proc(
    pid: int,
    exe: str | None,
    name: str,
    cmdline: list[str] | None = None,
    cwd: str = "",
):
    proc = MagicMock()
    proc.info = {
        "pid": pid,
        "exe": exe,
        "name": name,
        "cmdline": cmdline or [],
        "cwd": cwd,
    }
    proc.environ.return_value = {}
    return proc


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_finds_posix_venv_launcher(_winp, tmp_path):
    venv_py = str(tmp_path / "venv" / "bin" / "python")
    venv_py_versioned = str(tmp_path / "venv" / "bin" / "python3.13")
    base_py = "/usr/bin/python3.13"
    me = MagicMock()
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(101, base_py, "python3.13", [venv_py, "-m", "hermes_cli.main", "serve"]),
                _proc(102, base_py, "python3.13", [base_py, "somescript.py"]),
                _proc(103, venv_py_versioned, "python3.13"),
                _proc(
                    104,
                    str(tmp_path / "venv-other" / "bin" / "python3.13"),
                    "python3.13",
                ),
                _proc(
                    105,
                    base_py,
                    "python3.13",
                    ["venv/bin/python", "-m", "worker"],
                    cwd=str(tmp_path),
                ),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [m[0] for m in matches] == [101, 103, 105]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_finds_posix_dot_venv_launcher(_winp, tmp_path):
    dot_venv_python = str(tmp_path / ".venv" / "bin" / "python")
    proc = _proc(
        106,
        "/usr/bin/python3",
        "python3",
        [dot_venv_python, "-m", "hermes_cli.main", "serve"],
    )
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([proc]),
        Process=MagicMock(),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [106]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_ignores_non_python_mentions(_winp, tmp_path):
    venv_py = str(tmp_path / "venv" / "bin" / "python")
    me = MagicMock()
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(201, "/bin/bash", "bash", ["bash", "-c", f"{venv_py} -m worker"]),
                _proc(
                    202,
                    "/bin/sh",
                    "sh",
                    ["sh", "-c", "python -m hermes_cli.main"],
                    cwd=str(tmp_path),
                ),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_relative_argv0_requires_target_cwd(
    _winp, tmp_path
):
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(
                    202,
                    "/usr/bin/python3",
                    "python3",
                    ["venv/bin/python", "-m", "worker"],
                    cwd="",
                )
            ]
        ),
        Process=MagicMock(),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_resolves_bare_argv0_from_target_path(
    _winp, tmp_path
):
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.write_text("#!/bin/sh\n")
    venv_python.chmod(0o755)
    proc = _proc(
        203,
        "/usr/bin/python3",
        "python3",
        ["python", "-m", "hermes_cli.main", "serve"],
        cwd=str(tmp_path),
    )
    proc.environ.return_value = {
        "PATH": os.pathsep.join(
            [str(venv_bin), "/usr/local/bin", "/usr/bin"]
        )
    }
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([proc]),
        Process=MagicMock(),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [203]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_bare_argv0_path_denied_is_ignored(
    _winp, tmp_path
):
    proc = _proc(
        204,
        "/usr/bin/python3",
        "python3",
        ["python", "-m", "hermes_cli.main", "serve"],
        cwd=str(tmp_path),
    )
    proc.environ.side_effect = PermissionError("environment denied")
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([proc]),
        Process=MagicMock(),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_bare_argv0_honors_path_order(_winp, tmp_path):
    outside_bin = tmp_path / "outside-bin"
    venv_bin = tmp_path / "venv" / "bin"
    outside_bin.mkdir()
    venv_bin.mkdir(parents=True)
    for python_path in (outside_bin / "python", venv_bin / "python"):
        python_path.write_text("#!/bin/sh\n")
        python_path.chmod(0o755)
    proc = _proc(
        205,
        "/usr/bin/python3",
        "python3",
        ["python", "-m", "worker"],
        cwd=str(tmp_path),
    )
    proc.environ.return_value = {
        "PATH": os.pathsep.join([str(outside_bin), str(venv_bin)])
    }
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([proc]),
        Process=MagicMock(),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_finds_retitled_hermes_venv_map(
    _winp, tmp_path
):
    proc = _proc(
        206,
        "/usr/bin/python3",
        "hermes",
        ["hermes"],
        cwd="/tmp",
    )
    proc.environ.return_value = {"PATH": "/usr/local/bin:/usr/bin"}
    proc.memory_maps.return_value = [
        SimpleNamespace(
            path=str(
                tmp_path
                / "venv"
                / "lib"
                / "python3.13"
                / "site-packages"
                / "setproctitle.cpython-313.so"
            )
        )
    ]
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([proc]),
        Process=MagicMock(),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [206]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_ignores_unrelated_retitled_hermes_map(
    _winp, tmp_path
):
    proc = _proc(
        207,
        "/usr/bin/python3",
        "hermes",
        ["hermes"],
        cwd="/tmp",
    )
    proc.environ.return_value = {"PATH": "/usr/local/bin:/usr/bin"}
    proc.memory_maps.return_value = [
        SimpleNamespace(path="/opt/other/lib/setproctitle.so")
    ]
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([proc]),
        Process=MagicMock(),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_excludes_only_self(_winp, tmp_path):
    import os as _os

    venv_py = str(tmp_path / "venv" / "bin" / "python")
    parent = MagicMock()
    parent.pid = 555
    me = MagicMock()
    me.parents.return_value = [parent]
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(_os.getpid(), "/usr/bin/python3", "python3", [venv_py, "update"]),
                _proc(555, "/usr/bin/python3", "python3", [venv_py, "gateway"]),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [m[0] for m in matches] == [555]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_process_iteration_error_is_empty(_winp, tmp_path):
    def denied_process_iter(_attrs):
        raise PermissionError("process table denied")
        yield  # pragma: no cover

    fake_psutil = types.SimpleNamespace(process_iter=denied_process_iter)
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
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


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_windows_ignores_missing_exe(_winp, tmp_path):
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [_proc(101, None, "python.exe", [venv_py, "-m", "hermes_cli.main"])]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
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


@patch.object(cli_main, "_is_windows", return_value=False)
def test_format_venv_holders_message_explains_posix_runtime_mixing(_winp):
    msg = cli_main._format_venv_python_holders_message(
        [(101, "python3", "venv/bin/python -m hermes_cli.main serve")]
    )
    assert "already-loaded modules" in msg
    assert "newly-written package files" in msg


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


def _run_update_until_guard(args, *, is_windows=True, detector=None):
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

    with patch.object(cli_main, "_is_windows", return_value=is_windows), patch.object(
        cli_main, "_venv_scripts_dir", return_value=None
    ), patch.object(cli_main, "_run_pre_update_backup"), patch.object(
        cli_main, "_pause_windows_gateways_for_update", return_value=None
    ), patch.object(
        cli_main, "_resume_windows_gateways_after_update"
    ), patch.object(
        cli_main,
        "_detect_venv_python_processes",
        side_effect=detector,
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


def test_venv_holder_guard_runs_on_posix(capsys):
    result = _run_update_until_guard(
        _update_args(force=False, force_venv=False),
        is_windows=False,
    )
    assert result == "exit_2", capsys.readouterr().out


def test_venv_holder_guard_excludes_explicit_supervisor(monkeypatch, capsys):
    monkeypatch.setenv("_HERMES_UPDATE_SUPERVISOR_PID", "555")
    seen = []
    supervisor = SimpleNamespace(pid=555)
    fake_psutil = types.SimpleNamespace(
        Process=lambda: SimpleNamespace(parents=lambda: [supervisor])
    )

    def detect(*, exclude_pids=None):
        seen.append(exclude_pids)
        return []

    with patch.dict(sys.modules, {"psutil": fake_psutil}), patch(
        "hermes_cli.gateway.find_gateway_pids", return_value=[555, 666]
    ):
        result = _run_update_until_guard(
            _update_args(force=False, force_venv=False),
            is_windows=False,
            detector=detect,
        )

    assert result == "past_guard", capsys.readouterr().out
    assert seen == [{555, 666}]


def test_venv_holder_guard_rejects_non_ancestor_supervisor(monkeypatch, capsys):
    monkeypatch.setenv("_HERMES_UPDATE_SUPERVISOR_PID", "777")
    seen = []
    fake_psutil = types.SimpleNamespace(
        Process=lambda: SimpleNamespace(
            parents=lambda: [SimpleNamespace(pid=555)]
        )
    )

    def detect(*, exclude_pids=None):
        seen.append(exclude_pids)
        return []

    with patch.dict(sys.modules, {"psutil": fake_psutil}), patch(
        "hermes_cli.gateway.find_gateway_pids"
    ) as find_gateways:
        result = _run_update_until_guard(
            _update_args(force=False, force_venv=False),
            is_windows=False,
            detector=detect,
        )

    assert result == "past_guard", capsys.readouterr().out
    assert seen == [set()]
    find_gateways.assert_not_called()
