"""Regression: concurrent writers must not lose hand-off log lines.

``scripts/desktop-update/windows.ps1`` used to append to
``desktop-update-handoff.log`` with ``Add-Content``. That cmdlet is unsafe
once a second writer is live: PowerShell caches a content writer per path, so
the second writer receives a handle the first one already consumed. Measured
on the shape this fixture reproduces (8 writers x 100 lines, #126152):

* Windows PowerShell 5.1 -- a ``GetContentWriterArgumentError`` ("stream is
  not readable") error flood;
* PowerShell 7 -- no errors and **58% of lines silently missing**.

The hand-off log is the only record of why a Desktop update failed, and a
second writer is a real event, not a hypothesis: two hand-off processes can
overlap (the marker claim is a plain write with no mutual exclusion, and its
failure path logs a warning and continues), and every future runspace writer
joins the same race. ``Write-HandoffLog`` now serializes appends behind a
named, per-log-path mutex and one ``FileStream`` per write.

The fixture runs the REAL ``Write-HandoffLog`` (fetched as function text and
re-homed into each runspace with its own lazily created mutex instance, the
same way an overlapping process joins the named mutex) from a runspace pool,
then holds the file to the exact line count, per-writer count, and intact
line shape. It is ``platforms("windows")`` because Linux CI cannot execute
the PowerShell hand-off.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


@pytest.mark.platforms("windows")
def test_concurrent_writers_lose_no_handoff_log_lines(tmp_path: Path) -> None:
    """Drive 8 concurrent runspaces through the real Write-HandoffLog.

    A pass means all 800 lines landed: the total matches, each writer's 100
    lines match, and no line lost its timestamp-prefix + payload shape (the
    signature of a torn interleave). Under the pre-fix ``Add-Content`` the
    same run drops lines on PowerShell 7 and errors on 5.1.
    """
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    powershell = (
        system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    if not powershell.is_file():
        pytest.skip(f"Windows PowerShell not found at {powershell}")

    env = dict(os.environ)
    # The fixture writes its target log under TEMP; point that at tmp_path
    # so the test leaves nothing behind.
    env["TEMP"] = os.fspath(tmp_path)
    env["TMP"] = os.fspath(tmp_path)

    argv = [
        os.fspath(powershell),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        os.fspath(WINDOWS_PS1),
        "-SelfTestLogConcurrency",
    ]
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=os.fspath(REPO_ROOT),
    )

    (tmp_path / "log-concurrency.stdout.log").write_text(
        result.stdout, encoding="utf-8"
    )
    (tmp_path / "log-concurrency.stderr.log").write_text(
        result.stderr, encoding="utf-8"
    )
    diagnosis = result.stdout[-6000:] + result.stderr[-6000:]
    if "LOG-CONCURRENCY SELF-TEST: PASS" not in result.stdout or result.returncode != 0:
        pytest.fail(diagnosis, pytrace=False)
