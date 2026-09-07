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


NONCE = "b3" * 24


def _run_marker_claim(install_root: Path, desktop_pid: int, started_at: int,
                      nonce: str = NONCE) -> subprocess.CompletedProcess:
    """Drive step 0 through the UNMODIFIED production claim path.

    -SelfTestMarker only stops the run after step 0; it no longer selects a
    different marker body or a create-only CAS, so what this exercises is
    exactly what a real hand-off executes.
    """
    return subprocess.run(
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
            "-HandoffNonce",
            nonce,
            "-SelfTestMarker",
            "-NoUi",
            "-NoMarkerCleanup",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "HERMES_UPDATE_STARTED_AT": str(started_at)},
        cwd=REPO_ROOT,
        timeout=30,
    )


@pytest.mark.windows_only
def test_marker_self_test_adopts_only_the_exact_desktop_claim(tmp_path: Path) -> None:
    install_root = tmp_path / "hermes-agent"
    install_root.mkdir()
    marker = tmp_path / ".hermes-update-in-progress"
    started_at = 1_700_000_000
    desktop_pid = os.getpid()
    expected = f"{desktop_pid}\n{started_at}\n"
    marker.write_text(expected, encoding="utf-8", newline="")

    accepted = _run_marker_claim(install_root, desktop_pid, started_at)

    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    adopted = marker.read_text(encoding="utf-8").splitlines()
    assert int(adopted[0]) != desktop_pid
    assert int(adopted[0]) > 0
    assert int(adopted[1]) > started_at
    assert len(adopted) == 2, "the two-line wire format must not grow"

    marker.write_text(expected.replace(str(started_at), str(started_at - 1)), encoding="utf-8", newline="")
    refused = _run_marker_claim(install_root, desktop_pid, started_at)

    assert refused.returncode == 8, refused.stdout + refused.stderr
    assert marker.read_text(encoding="utf-8") == expected.replace(
        str(started_at), str(started_at - 1)
    )


@pytest.mark.windows_only
def test_claim_acknowledges_the_handoff_with_the_desktops_nonce(tmp_path: Path) -> None:
    """B4: the Desktop cannot otherwise tell our claim from any same-user
    process writing a plausible marker body inside its wait window."""
    install_root = tmp_path / "hermes-agent"
    install_root.mkdir()
    marker = tmp_path / ".hermes-update-in-progress"
    ack = tmp_path / ".hermes-update-in-progress.ack"
    started_at = 1_700_000_000
    desktop_pid = os.getpid()
    marker.write_text(f"{desktop_pid}\n{started_at}\n", encoding="utf-8", newline="")

    result = _run_marker_claim(install_root, desktop_pid, started_at)
    assert result.returncode == 0, result.stdout + result.stderr

    claimed = marker.read_text(encoding="utf-8").splitlines()
    acked = ack.read_text(encoding="utf-8").splitlines()
    assert acked[0] == NONCE, "the ack must echo the nonce only we were given"
    assert acked[1] == claimed[0], "the ack must name the pid that owns the marker"
    assert acked[2] == claimed[1], "and the creation time the marker is stamped with"


@pytest.mark.windows_only
def test_claim_without_a_nonce_adopts_the_marker_but_writes_no_ack(tmp_path: Path) -> None:
    """A nonce-less caller is a Desktop predating the ack protocol. It has no
    way to verify an ack, so none is written -- and the claim still happens,
    because refusing would strand a stale bundle with no route back."""
    install_root = tmp_path / "hermes-agent"
    install_root.mkdir()
    marker = tmp_path / ".hermes-update-in-progress"
    started_at = 1_700_000_000
    desktop_pid = os.getpid()
    marker.write_text(f"{desktop_pid}\n{started_at}\n", encoding="utf-8", newline="")

    result = _run_marker_claim(install_root, desktop_pid, started_at, nonce="")

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / ".hermes-update-in-progress.ack").exists()
    assert int(marker.read_text(encoding="utf-8").splitlines()[0]) != desktop_pid


# ---------------------------------------------------------------------------
# Nonce compatibility, exercised through the REAL script with no -SelfTest*
# switch involved: the run goes step 0 -> step 1 -> step 2 -> step 3 and stops
# at the missing venv interpreter (exit 3), which is what proves it proceeded
# past the claim rather than aborting inside it.
# ---------------------------------------------------------------------------


def _reaped_pid() -> int:
    """A PID that has certainly exited, so step 1's wait returns immediately."""
    child = subprocess.Popen([sys.executable, "-c", "pass"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert child.wait(timeout=30) == 0
    return child.pid


def _run_full_handoff(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess:
    install_root = tmp_path / "hermes-agent"
    install_root.mkdir(exist_ok=True)
    marker = tmp_path / ".hermes-update-in-progress"
    started_at = 1_700_000_000
    desktop_pid = _reaped_pid()
    marker.write_text(f"{desktop_pid}\n{started_at}\n", encoding="utf-8", newline="")

    return subprocess.run(
        [
            str(_powershell()), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
            "-InstallRoot", str(install_root),
            "-DesktopPid", str(desktop_pid),
            "-NoUi", "-NoMarkerCleanup", *extra,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "HERMES_UPDATE_STARTED_AT": str(started_at)},
        cwd=REPO_ROOT,
        timeout=120,
    )


@pytest.mark.windows_only
def test_legacy_launch_without_a_nonce_claims_and_proceeds(tmp_path: Path) -> None:
    """A Desktop older than the ack protocol sends no nonce. That skew is
    reachable: the pull lands, the desktop rebuild fails, and the next Update
    click drives this newer script from the stale asar. Refusing there would
    leave that bundle with no route back through the UI."""
    marker = tmp_path / ".hermes-update-in-progress"
    ack = tmp_path / ".hermes-update-in-progress.ack"

    result = _run_full_handoff(tmp_path)

    assert result.returncode == 3, result.stdout + result.stderr
    assert not ack.exists(), "a nonce-less caller cannot verify an ack, so none is written"

    claimed = marker.read_text(encoding="utf-8").splitlines()
    assert int(claimed[0]) > 0 and int(claimed[1]) > 1_700_000_000, "the claim still happens"

    log = (tmp_path / "logs" / "desktop-update-handoff.log").read_text(encoding="utf-8")
    assert "sent no nonce" in log, log


@pytest.mark.windows_only
def test_nonce_launch_fails_closed_when_the_ack_cannot_be_written(tmp_path: Path) -> None:
    """A caller that DID send a nonce is waiting for the ack and will refuse the
    hand-off without it. Proceeding would mutate the install under a live
    Desktop -- the exact hazard the ack exists to close."""
    # A directory where the ack belongs makes every write to it fail.
    (tmp_path / ".hermes-update-in-progress.ack").mkdir()

    result = _run_full_handoff(tmp_path, "-HandoffNonce", NONCE)

    assert result.returncode == 8, result.stdout + result.stderr
    log = (tmp_path / "logs" / "desktop-update-handoff.log").read_text(encoding="utf-8")
    assert "could not acknowledge the hand-off" in log, log
    assert "sent no nonce" not in log, "a nonce WAS sent; this is not the legacy path"


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
