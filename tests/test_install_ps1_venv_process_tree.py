"""Windows installer regression for Hermes children outside the venv.

The venv sweep deliberately selects process roots by executable path so it
does not kill unrelated Python processes. A selected Hermes process can spawn
a managed-runtime child whose executable lives outside the venv, though. The
installer must stop that whole tree before replacing the venv.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import psutil
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
POWERSHELL = next(
    (candidate for candidate in ("powershell", "pwsh") if shutil.which(candidate)),
    None,
)


def _pid_is_running(pid: int) -> bool:
    return psutil.pid_exists(pid)


def _wait_until_stopped(pid: int, timeout: float = 10) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.1)
    return not _pid_is_running(pid)


def _find_child_pid(parent_pid: int, executable: Path, timeout: float = 10) -> int:
    expected = str(executable).replace("'", "''")
    query = (
        f"$expected = '{expected}'; "
        f"Get-CimInstance Win32_Process -Filter 'ParentProcessId = {parent_pid}' | "
        "Where-Object { $_.ExecutablePath -and "
        "[string]::Equals($_.ExecutablePath, $expected, "
        "[System.StringComparison]::OrdinalIgnoreCase) } | "
        "Select-Object -First 1 -ExpandProperty ProcessId"
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            [POWERSHELL, "-NoProfile", "-Command", query],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
        time.sleep(0.1)
    raise AssertionError(f"child process did not start under PID {parent_pid}")


def _stop_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
    )


def _write_cmd(path: Path, text: str) -> None:
    with path.open("w", encoding="ascii", newline="\r\n") as handle:
        handle.write(text)


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(
    os.name != "nt" or POWERSHELL is None,
    reason="needs Windows and PowerShell",
)
def test_venv_sweep_stops_managed_runtime_children_but_not_unrelated_processes(
    tmp_path: Path,
) -> None:
    hermes_home = tmp_path / "hermes-home"
    install_dir = hermes_home / "hermes-agent"
    venv_scripts = install_dir / "venv" / "Scripts"
    runtime_dir = hermes_home / ".hermes-runtime" / "python" / "generation-test"
    unrelated_dir = tmp_path / "unrelated"
    fake_bin = tmp_path / "fake-bin"
    for directory in (venv_scripts, runtime_dir, unrelated_dir, fake_bin):
        directory.mkdir(parents=True)

    system_cmd = Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe"
    venv_parent_exe = venv_scripts / "python.exe"
    runtime_child_exe = runtime_dir / "python.exe"
    unrelated_exe = unrelated_dir / "python.exe"
    for target in (venv_parent_exe, runtime_child_exe, unrelated_exe):
        shutil.copy2(system_cmd, target)

    parent_script = tmp_path / "parent.cmd"
    _write_cmd(
        parent_script,
        f'@"{runtime_child_exe}" /d /c ping -t 127.0.0.1 ^>nul\n',
    )
    unrelated_script = tmp_path / "unrelated.cmd"
    _write_cmd(unrelated_script, "@ping -t 127.0.0.1 >nul\n")

    # Keep the test away from real gateway tasks and real Hermes launchers
    # while still exercising the installer's actual process enumeration and
    # per-PID taskkill behavior.
    _write_cmd(fake_bin / "schtasks.cmd", "@exit /b 0\n")
    _write_cmd(
        fake_bin / "taskkill.cmd",
        "@echo off\n"
        'echo %* | "%SystemRoot%\\System32\\findstr.exe" /I '
        '/C:"/IM hermes.exe" >nul\n'
        "if not errorlevel 1 exit /b 0\n"
        '"%SystemRoot%\\System32\\taskkill.exe" %*\n',
    )
    _write_cmd(
        fake_bin / "uv.cmd",
        "@echo off\n"
        'if /I "%~1 %~2"=="python find" (\n'
        "  echo C:\\Windows\\System32\\cmd.exe\n"
        "  exit /b 0\n"
        ")\n"
        'if /I "%~1"=="venv" (\n'
        '  if not exist "%CD%\\venv\\Scripts" mkdir "%CD%\\venv\\Scripts"\n'
        "  exit /b 0\n"
        ")\n"
        "exit /b 1\n",
    )

    creation_flags = subprocess.CREATE_NO_WINDOW
    parent = subprocess.Popen(
        [str(venv_parent_exe), "/d", "/c", str(parent_script)],
        creationflags=creation_flags,
    )
    unrelated = subprocess.Popen(
        [str(unrelated_exe), "/d", "/c", str(unrelated_script)],
        creationflags=creation_flags,
    )
    child_pid = 0
    try:
        child_pid = _find_child_pid(parent.pid, runtime_child_exe)
        assert _pid_is_running(child_pid)
        assert _pid_is_running(unrelated.pid)

        env = os.environ | {
            "OS": "Windows_NT",
            "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        }
        result = subprocess.run(
            [
                POWERSHELL,
                "-NoProfile",
                "-File",
                str(INSTALL_PS1),
                "-Stage",
                "venv",
                "-NonInteractive",
                "-InstallDir",
                str(install_dir),
                "-HermesHome",
                str(hermes_home),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert _wait_until_stopped(parent.pid)
        assert _wait_until_stopped(child_pid)
        assert _pid_is_running(unrelated.pid)
    finally:
        for pid in (child_pid, parent.pid, unrelated.pid):
            if pid:
                _stop_tree(pid)
