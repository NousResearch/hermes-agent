"""Regression: Windows Desktop update steps must drain and terminate reliably.

``scripts/desktop-update/windows.ps1`` runs each update step through
``Invoke-HermesStep``, which starts the step with ``RedirectStandardOutput`` /
``RedirectStandardError``. Reading those pipes back has two failure modes, and
this fixture covers both because they pull in opposite directions.

**Waiting for EOF (#90455).** The drain used to collect output with
``ReadToEndAsync().Result``. That task does not return when the *step* exits --
it returns when the *pipe* reaches EOF. On Windows the write end of a redirected
pipe is handed to the child as an inheritable handle, so every descendant
spawned without its own redirection holds a duplicate, and EOF waits for the
last of them to close it. ``hermes update`` deliberately runs build steps with
stdout inherited (the tee-stderr runner in ``hermes_cli/main.py``), so the
process tree under a step is arbitrarily deep and not something the hand-off can
enumerate. When one of those descendants is a resident gateway, the pipe never
closes and ``Invoke-HermesStep`` blocks for the life of the gateway.

Everything the hand-off owes the Desktop is downstream of that call:
``.hermes-update-result.json`` is never written, ``.hermes-update-in-progress``
is never cleared, and the Desktop is never relaunched -- so the app sits on
"Updating Hermes" until the user kills the gateway by hand, and the stale marker
then refuses the next update too.

**Trickling toward EOF.** The fix reads in chunks so an abandoned pipe still
yields what arrived. But a chunked drain that idles after every chunk it reads
is metered at one buffer per tick (16 KiB / 150ms ~ 107 KB/s), and because the
pipe then backs up that is backpressure on the *running* step, not just a slow
read -- a chatty step blocks on ``write()`` waiting for the reader. Measured on
this fixture's own flood arm: 4 MiB took 39.1s metered vs 0.09s unmetered, and a
step writing to both pipes took 18.3s vs 0.29s. ``hermes update`` is exactly this
shape; the Electron/vite build alone is megabytes.

**Live child stall (#95589).** A step can also finish its visible update work
but remain alive and silent in finalization. A post-exit pipe bound cannot help
that case. The hand-off must terminate the silent child so its existing retry
and finally paths can write the result, remove the marker, and relaunch Desktop.

So the contract is: bounded when a descendant holds the pipe open, never slower
than the step can write, and bounded when the step itself remains alive without
observable progress. All arms live in the script's own
``-SelfTestPipeDrain`` fixture, which is ``windows_only`` because Linux CI
cannot execute the PowerShell hand-off.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


@pytest.mark.windows_only
def test_update_step_survives_pipe_leak_flood_and_live_child_stall(
    tmp_path: Path,
) -> None:
    """Execute the real hand-off runner against all three step shapes.

    ``-SelfTestPipeDrain`` runs three steps through the real
    ``Invoke-HermesStep``:

    *leak* -- a step that spawns a grandchild with ``UseShellExecute = $false``
    and no redirection (the shape that makes the grandchild inherit the step's
    stdout/stderr), then exits 7 while the grandchild sleeps on. The fixture
    asserts the grandchild was **still alive** when ``Invoke-HermesStep``
    returned, so a pass cannot be a timing coincidence, and that the exit code
    and the step's output both survived the abandonment.

    *flood* -- a step that writes megabytes and holds nothing, exiting 5. It
    must complete in wall-clock far under what a sleep-per-chunk drain would
    take, and every byte must arrive.

    *stall* -- a step that emits one progress line and then remains alive and
    silent. It must be terminated with the timeout sentinel (124), preserve its
    output, and leave no child process behind.

    The existing leak/flood arms retain their measured Windows 11 / PowerShell
    5.1 budgets; the stall arm uses the same real runner and process table.
    """
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    powershell = (
        system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    if not powershell.is_file():
        pytest.skip(f"Windows PowerShell not found at {powershell}")

    env = {
        **os.environ,
        # The fixture writes its child scripts, pid file and hand-off log under
        # TEMP; point that at tmp_path so the test leaves nothing behind.
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        # Keep the test quick. The grace is what the fix bounds; the hold is
        # how long the leaking grandchild lives. hold >> grace is what makes a
        # regression measurable rather than lucky.
        "HERMES_UPDATE_PIPE_DRAIN_SECONDS": "3",
        "HERMES_UPDATE_STEP_IDLE_SECONDS": "3",
        "HERMES_SELFTEST_HOLD_SECONDS": "45",
    }

    result = subprocess.run(
        [
            str(powershell),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WINDOWS_PS1),
            "-SelfTestPipeDrain",
        ],
        capture_output=True,
        text=True,
        # Comfortably past both arms' worst cases (the 45s hold, and a metered
        # flood) so a regression fails with the fixture's own diagnosis instead
        # of an opaque timeout.
        timeout=300,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert "PIPE-DRAIN SELF-TEST: PASS" in result.stdout, (
        "The Windows update hand-off's step drain regressed: it either waited "
        "on a descendant holding the pipe open (the Desktop parks on 'Updating "
        "Hermes' forever) or metered a chatty step (backpressure on the running "
        f"update). Fixture diagnosis follows.\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert result.returncode == 0, (
        f"-SelfTestPipeDrain exited {result.returncode}.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
