"""Regression: the relaunch gate must reject windowless desktop pids.

``scripts/desktop-update/windows.ps1`` relaunches the Desktop after an
update through ``Start-DesktopRelaunch``. Before #102259's fix, a relaunch
was counted as "spawned" the moment the WMI ``Win32_Process.Create`` call
accepted and the pid existed -- the 20s window poll was best-effort focus
plumbing that timed out silently. A frozen main process (alive, ``Not
Responding``, no main window beyond the DDE Server Window) therefore read
as a SUCCESSFUL relaunch while it sat holding
``app.requestSingleInstanceLock()``, and every later user launch silently
exited 0 against it: the app appeared dead with no error anywhere.

The contract under test: a launch is only verifiably landed when the
relaunched pid exposes a main window within the bound. A live-but-windowless
pid is a failed launch -- the hand-off retries once via the explorer rung,
falls back to the tethered launch (output surfaced), and if nothing lands
the finally block downgrades the result to "Reopen Hermes to finish"
instead of reporting success. A dead pid is likewise never a landed
relaunch.

All arms live in the script's own ``-SelfTestRelaunch`` fixture: it drives
the real ``Confirm-DesktopWindow`` and the real ``Start-DesktopRelaunch``
against the real process table (a hidden PowerShell child is exactly the
windowless-zombie shape). ``platforms("windows")`` because Linux CI cannot
execute the PowerShell hand-off.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


@pytest.mark.platforms("windows")
def test_relaunch_gate_rejects_windowless_zombie_pids(tmp_path: Path) -> None:
    """Execute the real relaunch acceptance gate against all pid shapes.

    ``-SelfTestRelaunch`` runs four assertions against the production code:

    *zombie* -- a live hidden process with no main window (the #102259
    frozen-main-thread shape). ``Confirm-DesktopWindow`` must answer
    ``$false``, and the full ``Start-DesktopRelaunch`` ladder (WMI rung,
    explorer retry, tethered fallback) must end without blessing any
    windowless pid.

    *dead* -- a pid that exited before the poll. The gate must answer
    ``$false``; a relaunch that dies before showing a window is a failed
    launch, not a landed one.

    *healthy* -- a live windowed pid (the hand-off's own progress window).
    The gate must answer ``$true`` so the fix cannot regress into "reject
    everything"; headless runners without user32 degrade this arm to the
    liveness half only.
    """
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    powershell = (
        system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    if not powershell.is_file():
        pytest.skip(f"Windows PowerShell not found at {powershell}")

    env = {
        **os.environ,
        # The fixture writes its stub exe and hand-off log under TEMP; point
        # that at tmp_path so the test leaves nothing behind.
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        # Shrink the window poll so the fixture stays quick: the contract
        # under test is the verdict, not the 20s production bound.
        "HERMES_SELFTEST_RELAUNCH_WINDOW_SECONDS": "3",
    }

    result = subprocess.run(
        [
            str(powershell),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WINDOWS_PS1),
            "-SelfTestRelaunch",
        ],
        capture_output=True,
        text=True,
        # Worst case: three 3s polls plus the full ladder (WMI window poll +
        # explorer watch + fallback poll), comfortably under a CI runner's
        # patience, far over the fixture's own ~30s of real work.
        timeout=180,
        env=env,
        cwd=str(REPO_ROOT),
    )

    (tmp_path / "relaunch.stdout.log").write_text(result.stdout, encoding="utf-8")
    (tmp_path / "relaunch.stderr.log").write_text(result.stderr, encoding="utf-8")
    diagnosis = result.stdout[-6000:] + result.stderr[-6000:]
    if "RELAUNCH SELF-TEST: PASS" not in result.stdout or result.returncode != 0:
        pytest.fail(diagnosis, pytrace=False)
