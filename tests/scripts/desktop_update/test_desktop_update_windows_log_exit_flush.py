"""Every production exit of ``windows.ps1`` flushes the hand-off log lines it could not write yet.

A line logged while another process holds ``desktop-update-handoff.log`` against writers waits in
``Write-HandoffLog``'s queue. The main update drains that queue (bounded) in its ``finally``; the
marker helper ops and the early marker-load failure exit straight away, so on the base their only
log line vanished whenever the log was held at that moment. Each test holds the log, waits for the
line the script prints right after its failed append (a helper op's verdict, the console echo of
the abort), lets go, and requires the line in the file once the process exits.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import time

import pytest

from tests.scripts.desktop_update.windows_handoff_support import (
    MARKER,
    POWERSHELL,
    SCRIPT,
    _dead_pid,
    _env,
    _hold_against_writers,
    _log,
    _release,
)


def _held_log(home: Path) -> int:
    log = home / "logs" / "desktop-update-handoff.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"")
    return _hold_against_writers(log)


def _finish(proc: subprocess.Popen, handle: int) -> None:
    if handle:
        _release(handle)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
        proc.wait(timeout=30)


@pytest.mark.platforms("windows")
def test_a_marker_op_log_line_lands_after_the_log_is_released(tmp_path: Path) -> None:
    (tmp_path / MARKER).write_bytes(f"{_dead_pid()}\n{int(time.time())}\nct:5.000\n".encode())
    handle = _held_log(tmp_path)
    proc = subprocess.Popen(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
         "-MarkerOp", "reclaim", "-InstallRoot", str(tmp_path / "hermes-agent")],
        cwd=tmp_path, env=_env(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    try:
        # The verdict is printed after the op logged (and failed to append) its line.
        assert proc.stdout.readline() == "reclaimed\n"
        _release(handle)
        handle = 0
        assert proc.wait(timeout=60) == 0
    finally:
        _finish(proc, handle)
    assert "marker-op reclaim: removed a " in _log(tmp_path), _log(tmp_path)


@pytest.mark.platforms("windows")
def test_the_marker_load_failure_line_lands_after_the_log_is_released(tmp_path: Path) -> None:
    # A copy of the hand-off without marker.ps1 beside it: the early "could not be loaded" exit.
    alone = tmp_path / "script"
    alone.mkdir()
    shutil.copy2(SCRIPT, alone / "windows.ps1")
    home = tmp_path / "home"
    handle = _held_log(home)
    proc = subprocess.Popen(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(alone / "windows.ps1"),
         "-InstallRoot", str(home / "hermes-agent"), "-NoUi"],
        cwd=tmp_path, env=_env(home), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    try:
        # Write-HandoffLog echoes to the console after its append failed.
        assert "marker.ps1 could not be loaded" in proc.stdout.readline()
        _release(handle)
        handle = 0
        assert proc.wait(timeout=60) == 3
    finally:
        _finish(proc, handle)
    assert "marker.ps1 could not be loaded" in _log(home), _log(home)
