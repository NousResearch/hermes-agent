"""Regression: an unhandled post-update exception must yield a truthful receipt.

#109627: on Windows, the desktop hand-off runs detached with stdio ignored and
initializes ``$finalCode = 1`` / ``$finalMsg = "update did not complete"``. The
main try had a ``finally`` but no ``catch``, so any terminating exception in the
post-update phase (before the verify step reassigned the finals) skipped straight
to ``finally``, which wrote those opaque defaults into
``.hermes-update-result.json`` — the receipt the relaunched Desktop consumes.
Every *successful* update then showed the failure dialog: exit 1, "update did
not complete".

The fix in ``scripts/desktop-update/windows.ps1`` has three parts:

* ``Invoke-HermesStep`` traps step-launch failures (``StartAssigned`` throws)
  and returns a structured ``Code = 1`` result instead of leaking a terminating
  exception to the outer try;
* the outer try gained a ``catch`` that logs the error with its script stack
  trace (``CRITICAL: unhandled error in hand-off``) and derives a truthful
  verdict via ``Get-HermesUnhandledOutcome``: a completed update (step exit 0)
  maps to exit 8 ("updated but a post-update step failed"), anything else keeps
  the step's own exit code, and the message always names the exception;
* the ``finally`` block refuses to write the opaque default when an exception
  was captured.

The executable proof is the ``-SelfTestReceipt`` arm of ``windows.ps1``
(``platforms("windows")`` below): fixture error records drive
``Get-HermesUnhandledOutcome`` (completed-then-crash, failed-step crash, and
pre-step crash), and the REAL ``Invoke-HermesStep`` launch-failure path runs
with a nonexistent executable — ``StartAssigned``'s ``CreateProcess`` fails and
the catch must return a structured failure carrying the Win32 error, not throw.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


@pytest.mark.platforms("windows")
def test_unhandled_post_update_exception_yields_truthful_receipt(
    tmp_path: Path,
) -> None:
    """Execute the real receipt-truth machinery against fixture failures.

    ``-SelfTestReceipt`` fails with a diagnosis when any of these regress:

    * a completed update (step exit 0) followed by a post-update crash maps to
      exit 8 and a message naming the exception — never "update did not
      complete";
    * a failed step keeps its own exit code in the receipt, message still
      truthful;
    * a crash before any step result maps to exit 1, message still truthful;
    * a step whose executable cannot spawn returns a structured failure
      (exit 1, the failure reason in the output) instead of a terminating
      exception that the outer try would flatten into the defaults.
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
            "-SelfTestReceipt",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(REPO_ROOT),
    )

    (tmp_path / "receipt.stdout.log").write_text(result.stdout, encoding="utf-8")
    (tmp_path / "receipt.stderr.log").write_text(result.stderr, encoding="utf-8")
    if "RECEIPT SELF-TEST: PASS" not in result.stdout or result.returncode != 0:
        pytest.fail(
            "The Windows hand-off receipt truth machinery regressed: an "
            "unhandled post-update exception can be flattened into the opaque "
            "'update did not complete' default even when the update step "
            "succeeded (#109627), or a step launch failure escapes as a "
            "terminating exception. Diagnosis follows.\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
            pytrace=False,
        )
