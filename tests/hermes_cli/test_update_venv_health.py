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

import ctypes
import subprocess
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from hermes_cli import main as cli_main
from hermes_cli import update_cmd as update_module


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
    proc.exe.return_value = exe
    proc.cmdline.return_value = cmdline or []
    proc.cwd.return_value = cwd
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
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        update_module, "_windows_process_image_rows", return_value=None
    ), patch.dict(sys.modules, {"psutil": fake_psutil}):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_fallback_preserves_full_process_contract(_winp, tmp_path):
    unrelated = _proc(101, r"C:\Program Files\nodejs\node.exe", "node.exe")
    trampoline = _proc(
        102,
        r"C:\Users\test\AppData\Roaming\uv\python\python.exe",
        "python.exe",
        ["python.exe", "-m", "hermes_cli.main", "serve"],
        str(tmp_path),
    )
    attrs_seen = []
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: attrs_seen.extend(attrs) or iter([unrelated, trampoline]),
        Process=lambda *a, **k: me,
    )

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        update_module, "_windows_process_image_rows", return_value=None
    ), patch.dict(sys.modules, {"psutil": fake_psutil}):
        assert cli_main._detect_venv_python_processes() == [
            (102, "python.exe", "python.exe -m hermes_cli.main serve")
        ]

    assert attrs_seen == ["pid", "exe", "name", "cmdline", "cwd"]
    unrelated.exe.assert_not_called()
    unrelated.cmdline.assert_not_called()
    unrelated.cwd.assert_not_called()
    trampoline.exe.assert_not_called()
    trampoline.cmdline.assert_not_called()
    trampoline.cwd.assert_not_called()


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_native_path_catches_arbitrary_venv_executable(_winp, tmp_path):
    venv_exe = str(tmp_path / "venv" / "Scripts" / "custom-runner.exe")
    holder = _proc(303, venv_exe, "custom-runner.exe", [venv_exe, "worker.py"])
    me = MagicMock()
    me.parents.return_value = []
    constructed_pids = []

    def process(pid=None):
        if pid is None:
            return me
        constructed_pids.append(pid)
        return holder

    fake_psutil = types.SimpleNamespace(
        process_iter=MagicMock(side_effect=AssertionError("fallback must not run")),
        Process=process,
    )

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        update_module,
        "_windows_process_image_rows",
        return_value=[
            (303, "custom-runner.exe", venv_exe),
            (404, "node.exe", r"C:\Program Files\nodejs\node.exe"),
        ],
    ), patch.dict(sys.modules, {"psutil": fake_psutil}):
        assert cli_main._detect_venv_python_processes() == [
            (303, "custom-runner.exe", f"{venv_exe} worker.py")
        ]

    fake_psutil.process_iter.assert_not_called()
    assert constructed_pids == [303]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_fallback_mid_iteration_error_never_raises(_winp, tmp_path):
    holder = _proc(
        707,
        str(tmp_path / "venv" / "Scripts" / "python.exe"),
        "python.exe",
    )
    me = MagicMock()
    me.parents.return_value = []

    def broken_process_iter(_attrs):
        yield holder
        raise OSError("process table changed")

    fake_psutil = types.SimpleNamespace(
        process_iter=broken_process_iter,
        Process=lambda *a, **k: me,
    )

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        update_module, "_windows_process_image_rows", return_value=None
    ), patch.dict(sys.modules, {"psutil": fake_psutil}):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_native_supports_versioned_external_python(_winp, tmp_path):
    external = r"C:\Python311\python3.11.exe"
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    holder = _proc(505, external, "python3.11.exe", [external, venv_py, "worker.py"])
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=MagicMock(side_effect=AssertionError("fallback must not run")),
        Process=lambda pid=None: me if pid is None else holder,
    )

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        update_module,
        "_windows_process_image_rows",
        return_value=[(505, "python3.11.exe", external)],
    ), patch.dict(sys.modules, {"psutil": fake_psutil}):
        assert cli_main._detect_venv_python_processes() == [
            (505, "python3.11.exe", f"{external} {venv_py} worker.py")
        ]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_revalidates_native_pid_identity(_winp, tmp_path):
    snapshot_exe = str(tmp_path / "venv" / "Scripts" / "custom-runner.exe")
    reused = _proc(606, r"C:\Windows\System32\notepad.exe", "notepad.exe")
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=MagicMock(side_effect=AssertionError("fallback must not run")),
        Process=lambda pid=None: me if pid is None else reused,
    )

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        update_module,
        "_windows_process_image_rows",
        return_value=[(606, "custom-runner.exe", snapshot_exe)],
    ), patch.dict(sys.modules, {"psutil": fake_psutil}):
        assert cli_main._detect_venv_python_processes() == []

    reused.exe.assert_called_once_with()


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_reclassifies_reused_pid_from_live_executable(_winp, tmp_path):
    snapshot_exe = str(tmp_path / "venv" / "Scripts" / "custom-runner.exe")
    live_exe = r"C:\Python311\python3.11.exe"
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    reused = _proc(
        616,
        live_exe,
        "python3.11.exe",
        [live_exe, venv_py, "worker.py"],
    )
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=MagicMock(side_effect=AssertionError("fallback must not run")),
        Process=lambda pid=None: me if pid is None else reused,
    )

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        update_module,
        "_windows_process_image_rows",
        return_value=[(616, "custom-runner.exe", snapshot_exe)],
    ), patch.dict(sys.modules, {"psutil": fake_psutil}):
        assert cli_main._detect_venv_python_processes() == [
            (616, "python3.11.exe", f"{live_exe} {venv_py} worker.py")
        ]

    reused.exe.assert_called_once_with()
    reused.cmdline.assert_called_once_with()


def _fake_kernel32(*, next_error: int = 18, invalid_snapshot: bool = False):
    from ctypes import wintypes

    kernel = types.SimpleNamespace(
        CreateToolhelp32Snapshot=MagicMock(),
        Process32FirstW=MagicMock(),
        Process32NextW=MagicMock(),
        OpenProcess=MagicMock(),
        QueryFullProcessImageNameW=MagicMock(),
        CloseHandle=MagicMock(return_value=True),
    )
    kernel.CreateToolhelp32Snapshot.return_value = (
        wintypes.HANDLE(-1).value if invalid_snapshot else 100
    )

    def first(_snapshot, entry_pointer):
        entry = entry_pointer._obj
        entry.th32ProcessID = 707
        entry.szExeFile = "custom-runner.exe"
        return True

    def query(_process, _flags, buffer, _size_pointer):
        buffer.value = r"C:\Hermes\venv\Scripts\custom-runner.exe"
        return True

    kernel.Process32FirstW.side_effect = first
    kernel.Process32NextW.return_value = False
    kernel.OpenProcess.return_value = 200
    kernel.QueryFullProcessImageNameW.side_effect = query
    return kernel, next_error


@patch.object(cli_main, "_is_windows", return_value=True)
def test_windows_process_image_rows_closes_snapshot_and_process_handles(_winp):
    kernel, last_error = _fake_kernel32()
    with (
        patch.object(ctypes, "WinDLL", create=True, return_value=kernel),
        patch.object(ctypes, "get_last_error", return_value=last_error),
    ):
        rows = update_module._windows_process_image_rows()

    assert rows == [
        (707, "custom-runner.exe", r"C:\Hermes\venv\Scripts\custom-runner.exe")
    ]
    assert kernel.CloseHandle.call_args_list == [call(200), call(100)]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_windows_process_image_rows_invalid_snapshot_fails_to_fallback(_winp):
    kernel, last_error = _fake_kernel32(invalid_snapshot=True)
    with (
        patch.object(ctypes, "WinDLL", create=True, return_value=kernel),
        patch.object(ctypes, "get_last_error", return_value=last_error),
    ):
        assert update_module._windows_process_image_rows() is None

    kernel.CloseHandle.assert_not_called()


@patch.object(cli_main, "_is_windows", return_value=True)
def test_windows_process_image_rows_abnormal_next_error_fails_to_fallback(_winp):
    kernel, last_error = _fake_kernel32(next_error=5)
    with (
        patch.object(ctypes, "WinDLL", create=True, return_value=kernel),
        patch.object(ctypes, "get_last_error", return_value=last_error),
    ):
        assert update_module._windows_process_image_rows() is None

    assert kernel.CloseHandle.call_args_list == [call(200), call(100)]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_windows_process_image_rows_zero_next_error_fails_to_fallback(_winp):
    kernel, last_error = _fake_kernel32(next_error=0)
    with (
        patch.object(ctypes, "WinDLL", create=True, return_value=kernel),
        patch.object(ctypes, "get_last_error", return_value=last_error),
    ):
        assert update_module._windows_process_image_rows() is None

    assert kernel.CloseHandle.call_args_list == [call(200), call(100)]




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
        # Pin the orphan classifier: this test exercises --force/--force-venv
        # gating, not orphan detection (covered in
        # test_update_orphan_backend_reap.py). None = "not provably orphaned"
        # → the guard refuses exactly as before the orphan-reap addition.
        cli_main, "_orphaned_desktop_backend_pids", return_value=None
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
