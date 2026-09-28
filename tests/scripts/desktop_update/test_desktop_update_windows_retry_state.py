"""Regression: a completed update must never be re-run by the timeout retry.

#96205: on native Windows, ``hermes update`` runs to completion (its output
shows "✓ Update complete!") but the step process does not exit cleanly -- the
post-update finalization (gateway-restart hand-off) stays alive and silent
until the idle watchdog in ``Invoke-HermesStep`` terminates the tree with
the timeout sentinel 124. The retry gates re-ran the update on any non-zero,
non-2 exit code, so the completed update was re-applied from scratch and the
Desktop updater popup hung for 70+ minutes. The sibling fix on the bootstrap
installer (``apps/bootstrap-installer/src-tauri/src/update.rs``) excludes
exit 124 from its own auto-retry, so both retry layers agree.

The fix is the retry state machine in ``scripts/desktop-update/windows.ps1``
(``Get-HermesUpdateRetryState`` + ``Resolve-HermesUpdateOutcome``): exit 124
with the completion marker is a terminal state, surfaced as success so the
hand-off verifies and relaunches the NEW build. The rule is deliberately
narrow -- a plain exit 1 after the marker is the stale-gateway verdict
(``hermes_cli/update_receipt.py`` prints "✗ Update not complete" after the
completion banner and must supersede it), and exit 2 is fail-closed.

The executable proof is the ``-SelfTestRetryState`` arm of ``windows.ps1``
(``platforms("windows")`` below): fixture states drive the state machine and
``Resolve-HermesUpdateOutcome`` directly, including the exact #96205 shape
(completed then watchdog-killed), the stale-gateway verdict that must NOT be
flattened into success, and the mid-update timeout that keeps its non-zero
exit. ``scripts/desktop-update/posix.sh`` carries the same terminal-state gate
for the sibling POSIX hand-off.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


@pytest.mark.platforms("windows")
def test_completed_update_surfaces_as_success_after_watchdog_kill(
    tmp_path: Path,
) -> None:
    """Execute the real retry state machine against fixture step results.

    ``-SelfTestRetryState`` drives ``Get-HermesUpdateRetryState`` with fixture
    ``(code, output)`` pairs and ``Resolve-HermesUpdateOutcome`` with the
    shapes that matter:

    * *completed then timeout* -- exit 124 after "✓ Update complete!" was
      printed. Must surface as exit 0 with the step output preserved, so the
      hand-off relaunches the new build instead of re-running the update.
    * *stale-gateway verdict* -- exit 1 after the marker. Must keep its
      non-zero exit: the fleet restart phase lands exit codes after the
      completion banner on purpose.
    * *mid-update timeout* -- exit 124 without the marker. Must keep its
      non-zero exit so the ordinary retry/failure paths decide.
    * *fail-closed* -- exit 2 ("close all Hermes windows"). Not remapped.
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
            "-SelfTestRetryState",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(REPO_ROOT),
    )

    (tmp_path / "retry-state.stdout.log").write_text(result.stdout, encoding="utf-8")
    (tmp_path / "retry-state.stderr.log").write_text(result.stderr, encoding="utf-8")
    if "RETRY-STATE SELF-TEST: PASS" not in result.stdout or result.returncode != 0:
        pytest.fail(
            "The update retry state machine regressed: a completed update can "
            "be re-run after a watchdog timeout (the #96205 retry storm that "
            "hangs the Desktop updater popup for 70+ minutes), or a state maps "
            "to the wrong outcome. Fixture diagnosis follows.\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
            pytrace=False,
        )
