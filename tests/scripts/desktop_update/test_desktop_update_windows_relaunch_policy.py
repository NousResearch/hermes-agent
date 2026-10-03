"""Regression: the exit-timeout abort must not relaunch a second Desktop.

#88332: the Windows hand-off waits for the original Desktop to exit before
touching the venv, and aborts with exit 4 when it does not ("the Hermes window
(pid N) did not exit within Ns. Nothing was changed."). The failure path in the
script's ``finally`` block then called ``Start-DesktopRelaunch`` for EVERY
non-zero result -- including that abort, whose whole premise is that the
original window is still alive. The updater therefore spawned a deterministic
SECOND Desktop instance over an update that never ran, while the error finale
claimed failure; the two windows then fight over the single-instance lock and
the user sees an abort loop with no update applied.

The contract under test (``Test-SafeToRelaunch`` in ``windows.ps1``): the
failure path may bring the Desktop back for any failure EXCEPT exit 4 while
the original pid is still alive -- that combination must be refused. A slow
quit that finished between the abort and the finally block (pid gone) keeps
the relaunch: it is safe and wanted. Every other failure code keeps the old
bring-it-back behavior, and success is never blocked.

The executable proof is the script's ``-SelfTestRelaunchPolicy`` arm
(``platforms("windows")`` below): it drives the REAL policy function against
the REAL process table, using a live hidden PowerShell child as the stand-in
desktop pid, across the (exit code, pid-liveness) matrix.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


@pytest.mark.platforms("windows")
def test_exit_timeout_abort_does_not_relaunch_over_live_desktop(
    tmp_path: Path,
) -> None:
    """Execute the real relaunch policy against the (code, liveness) matrix.

    ``-SelfTestRelaunchPolicy`` fails with a diagnosis when any of these
    regress:

    * exit 4 + a live desktop pid must REFUSE the relaunch (the deterministic
      second instance, #88332 requirement 2);
    * exit 4 + no live desktop pid must ALLOW the relaunch (a slow quit that
      finished after the abort — the relaunch is safe and wanted);
    * any other failure code (1, 8) must keep the old bring-it-back behavior;
    * success (0) must never be blocked by the policy.
    """
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    powershell = (
        system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    if not powershell.is_file():
        pytest.skip(f"Windows PowerShell not found at {powershell}")

    env = {
        **os.environ,
        # The arm writes its hand-off log under TEMP; point that at tmp_path
        # so the test leaves nothing behind.
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
    }

    result = subprocess.run(
        [
            str(powershell),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WINDOWS_PS1),
            "-SelfTestRelaunchPolicy",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(REPO_ROOT),
    )

    (tmp_path / "relaunch-policy.stdout.log").write_text(result.stdout, encoding="utf-8")
    (tmp_path / "relaunch-policy.stderr.log").write_text(result.stderr, encoding="utf-8")
    diagnosis = result.stdout[-6000:] + result.stderr[-6000:]
    if (
        "RELAUNCH-POLICY SELF-TEST: PASS" not in result.stdout
        or result.returncode != 0
    ):
        pytest.fail(diagnosis, pytrace=False)
