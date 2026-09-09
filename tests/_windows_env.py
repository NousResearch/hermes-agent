"""Shared minimal-but-valid Windows environment for PowerShell subprocess tests.

The canonical runner (``scripts/run_tests.sh``) executes under Git-Bash
``env -i``, which strips ``PATH`` / ``PATHEXT`` / ``ComSpec`` / ``SystemRoot`` /
``windir``. A PowerShell child launched with that environment cannot spawn any
grandchild process — ``& python`` / ``& cmd`` silently do nothing and
``$LASTEXITCODE`` is left ``$null``. Handing ``subprocess.run`` the environment
below restores just enough for a PowerShell probe to behave the way it would on a
real Windows host, under both the canonical runner and a bare ``pytest``.
"""

from __future__ import annotations

import os


def minimal_windows_subprocess_env() -> dict:
    """A minimal but valid Windows environment for a PowerShell probe subprocess.

    Supplies ``SystemRoot`` / ``windir`` (process + DLL init), ``PATH`` (System32
    plus the PowerShell directory, so children can load system DLLs and resolve a
    nested ``powershell``), ``PATHEXT`` / ``ComSpec`` (so ``& cmd`` name
    resolution works), and ``TEMP`` / ``TMP`` (Python ``tempfile``, PowerShell
    scratch). ``SystemRoot`` falls back through ``SYSTEMROOT`` then ``C:\\Windows``;
    ``TEMP`` / ``TMP`` fall back to ``<SystemRoot>\\Temp``.
    """
    win = os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT") or r"C:\Windows"
    return {
        "SystemRoot": win,
        "windir": win,
        "PATH": os.pathsep.join(
            [win + r"\System32", win, win + r"\System32\WindowsPowerShell\v1.0"]
        ),
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "ComSpec": win + r"\System32\cmd.exe",
        "TEMP": os.environ.get("TEMP", win + r"\Temp"),
        "TMP": os.environ.get("TMP", win + r"\Temp"),
    }
