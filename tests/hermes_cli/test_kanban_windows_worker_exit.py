from __future__ import annotations

import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db_dispatch as kbd


@pytest.mark.windows_only
def test_windows_reaper_preserves_real_worker_exit_code() -> None:
    """A tracked Windows worker must not collapse to ``unknown``."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(37)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    kbd._track_worker_process(proc)

    deadline = time.monotonic() + 5
    reaped: list[int] = []
    while time.monotonic() < deadline and proc.pid not in reaped:
        reaped.extend(kbd.reap_worker_zombies())
        if proc.pid not in reaped:
            time.sleep(0.01)

    assert proc.pid in reaped
    assert kbd._classify_worker_exit(proc.pid) == ("nonzero_exit", 37)
