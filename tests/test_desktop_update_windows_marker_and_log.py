from __future__ import annotations

import os
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


def _powershell() -> Path:
    path = (
        Path(os.environ.get("SystemRoot", r"C:\Windows"))
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    if not path.is_file():
        pytest.skip(f"Windows PowerShell not found at {path}")
    return path


@pytest.mark.windows_only
def test_marker_self_test_adopts_only_the_exact_desktop_claim(tmp_path: Path) -> None:
    install_root = tmp_path / "hermes-agent"
    install_root.mkdir()
    marker = tmp_path / ".hermes-update-in-progress"
    started_at = 1_700_000_000
    desktop_pid = os.getpid()
    expected = f"{desktop_pid}\n{started_at}\n"
    marker.write_text(expected, encoding="utf-8", newline="")

    env = {**os.environ, "HERMES_UPDATE_STARTED_AT": str(started_at)}
    accepted = subprocess.run(
        [
            str(_powershell()),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-InstallRoot",
            str(install_root),
            "-DesktopPid",
            str(desktop_pid),
            "-SelfTestMarker",
            "-NoUi",
            "-NoMarkerCleanup",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=30,
    )

    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    adopted = marker.read_text(encoding="utf-8").splitlines()
    assert int(adopted[0]) != desktop_pid
    assert int(adopted[0]) > 0
    assert int(adopted[1]) > started_at

    marker.write_text(expected.replace(str(started_at), str(started_at - 1)), encoding="utf-8", newline="")
    refused = subprocess.run(
        [
            str(_powershell()),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-InstallRoot",
            str(install_root),
            "-DesktopPid",
            str(desktop_pid),
            "-SelfTestMarker",
            "-NoUi",
            "-NoMarkerCleanup",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=30,
    )

    assert refused.returncode == 8, refused.stdout + refused.stderr
    assert marker.read_text(encoding="utf-8") == expected.replace(
        str(started_at), str(started_at - 1)
    )


@pytest.mark.windows_only
def test_log_self_test_survives_shared_reader(tmp_path: Path) -> None:
    env = {**os.environ, "TEMP": str(tmp_path), "TMP": str(tmp_path)}
    result = subprocess.run(
        [
            str(_powershell()),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-SelfTestLog",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "LOG SELF-TEST: PASS" in result.stdout


@pytest.mark.windows_only
def test_dev_relaunch_passes_the_checkout_as_electron_argument(tmp_path: Path) -> None:
    electron = tmp_path / "electron.exe"
    app = tmp_path / "checkout with spaces"
    result = subprocess.run(
        [
            str(_powershell()),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-RelaunchExe",
            str(electron),
            "-RelaunchAppPath",
            str(app),
            "-SelfTestRelaunchCommand",
            "-NoUi",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    invocation = json.loads(result.stdout)
    assert invocation["Executable"] == str(electron)
    assert invocation["Arguments"] == str(app)
    assert invocation["CommandLine"] == f'"{electron}" "{app}"'


@pytest.mark.windows_only
@pytest.mark.parametrize("direct", [False, True])
def test_relaunch_preserves_app_path_and_custom_environment(tmp_path: Path, direct: bool) -> None:
    output = tmp_path / "environment.json"
    probe = tmp_path / "relaunch probe.py"
    probe.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "keys = ['HERMES_HOME', 'HERMES_DESKTOP_USER_DATA_DIR', 'HERMES_DESKTOP_HERMES_ROOT', 'PATH']\n"
        "Path(os.environ['HERMES_RELAUNCH_TEST_OUTPUT']).write_text("
        "json.dumps({key: os.environ.get(key) for key in keys}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "HERMES_HOME": str(tmp_path / "custom home"),
        "HERMES_DESKTOP_USER_DATA_DIR": str(tmp_path / "custom desktop data"),
        "HERMES_DESKTOP_HERMES_ROOT": str(tmp_path / "custom source"),
        "HERMES_RELAUNCH_TEST_OUTPUT": str(output),
    }
    result = subprocess.run(
        [str(_powershell()), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
         "-SelfTestRelaunchEnvironment", "-RelaunchExe", sys.executable,
         "-RelaunchAppPath", str(probe), "-NoUi", *(["-SelfTestDirectRelaunch"] if direct else [])],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["ReturnValue"] == 0
    deadline = time.monotonic() + 10
    while not output.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    observed = json.loads(output.read_text(encoding="utf-8"))
    for key in ("HERMES_HOME", "HERMES_DESKTOP_USER_DATA_DIR", "HERMES_DESKTOP_HERMES_ROOT"):
        assert observed[key] == env[key]
    assert observed["PATH"] == env.get("PATH", env.get("Path"))


@pytest.mark.windows_only
def test_direct_packaged_relaunch_omits_empty_argument_list() -> None:
    # With no arguments, rundll32 exits without loading a DLL or opening UI.
    executable = _powershell().parents[2] / "rundll32.exe"
    result = subprocess.run(
        [str(_powershell()), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
         "-SelfTestRelaunchEnvironment", "-SelfTestDirectRelaunch", "-RelaunchExe", str(executable), "-NoUi"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["ProcessId"] > 0
