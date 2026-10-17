"""Regression: a relaunched Desktop must never end up running with no window.

``scripts/desktop-update/windows.ps1`` hands off to a new Desktop after an
update and then tries to bring its window forward. That step polls
``Process.MainWindowHandle``, which returns Zero unless the process owns a
window that ``IsWindowVisible`` reports as visible. Electron creates its
BrowserWindow hidden and shows it on ``ready-to-show``, so during startup the
app can own a real HWND that the property reports as "no window" — verified
directly: a process owning four windows, none of them visible, reports
``MainWindowHandle`` 0.

Observed on a real update (v0.21.5+6724.gd385b01 -> v0.21.5+6725.gc225c4a,
Windows 11). The hand-off logged ``desktop relaunched detached (pid 8160)``
followed by ``focused relaunched desktop window``, and the app was left with

  * a 1440x753 window at (26,26) whose ``IsWindowVisible`` was False, and
  * a second window titled "Hermes" parked at -32000,-32000 (minimised),

i.e. six live Hermes processes and no GUI. The poll's own timeout had no
branch at all: when the deadline passed nothing was logged and the caller
still saw a successful relaunch, so nothing downstream could react.

The fixture asserts the three outcomes the function owes its caller: a shown
window is focused, a window that never appears is reported as a visibility
miss and is NOT treated as a failed launch (which would spawn a second
instance), and a process that dies before showing anything is a failed launch
so the caller can fall back.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


@pytest.mark.platforms("windows")
def test_relaunch_reports_a_window_that_never_appears(tmp_path: Path) -> None:
    """Drive ``-SelfTestRelaunchWindow`` across its visible/hidden/dead arms.

    Each arm is a real child process. The child signals readiness with a file
    once its window exists and the probe runs only after that, because the arm
    otherwise races PowerShell's own startup — Add-Type of WinForms alone can
    take seconds, and a genuinely visible window then reads as a timeout. The
    child is deliberately not launched with ``-WindowStyle Hidden``: WinForms
    applies the process STARTUPINFO show state to its first window, so that
    flag suppresses the very window the visible arm exists to produce.
    """
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    powershell = (
        system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    if not powershell.is_file():
        pytest.skip(f"Windows PowerShell not found at {powershell}")

    env = {
        **os.environ,
        # The fixture writes its child script and ready files under TEMP; point
        # that at tmp_path so the test leaves nothing behind.
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        # The hidden arm has to sit out a full timeout, so keep it short. The
        # readiness handshake is what makes that safe rather than flaky.
        "HERMES_SELFTEST_WINDOW_TIMEOUT": "3",
    }

    result = subprocess.run(
        [
            str(powershell),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WINDOWS_PS1),
            "-SelfTestRelaunchWindow",
        ],
        capture_output=True,
        text=True,
        # Three arms, each bounded by a 45s readiness wait plus a 3s timeout.
        timeout=240,
        env=env,
        cwd=str(REPO_ROOT),
    )

    (tmp_path / "relaunch-window.stdout.log").write_text(result.stdout, encoding="utf-8")
    (tmp_path / "relaunch-window.stderr.log").write_text(result.stderr, encoding="utf-8")
    diagnosis = result.stdout[-6000:] + result.stderr[-6000:]
    if (
        "SELF-TEST relaunch-window: all arms passed" not in result.stdout
        or result.returncode != 0
    ):
        pytest.fail(diagnosis, pytrace=False)