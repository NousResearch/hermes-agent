"""The Desktop-exit gate must treat a zombie PID as gone.

``kill -0`` answers "does this PID exist", not "is this process running". A
process whose parent never reaps it stays a zombie and keeps its PID
indefinitely, so a Desktop launched from a non-reaping parent (app launchers,
Stream Deck/OpenDeck plugin hosts, some IDE and file-manager spawners) leaves a
PID that answers ``kill -0`` forever after the window closes. The hand-off's
exit gate then burns its full 30s and aborts an update that was safe to run:

    Update aborted: the Hermes window (pid N) did not exit within 30s.

The abort path relaunches through ``setsid``, reparenting the new Desktop to
init — which DOES reap — so the retry always sees a real exit. That is the
"fails nearly every time, works on the second try" shape of the report.

Drives the real ``--self-test-pid-alive`` entry point of ``posix.sh`` against
real processes: a real fork for the zombie (no simulated process state), and a
genuinely running child for the safety direction.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

POSIX_SH = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "desktop-update" / "posix.sh"

pytestmark = pytest.mark.linux_only


def _gate_says_alive(pid: int) -> bool:
    out = subprocess.run(
        ["bash", str(POSIX_SH), "--self-test-pid-alive", str(pid)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return out.stdout.strip() == "alive"


def test_running_desktop_still_blocks_the_update():
    """The gate's safety property: a live window must still be waited out."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert _gate_says_alive(proc.pid) is True
    finally:
        proc.kill()
        proc.wait()


def test_zombie_desktop_does_not_block_the_update():
    """A PID that only exists because nobody reaped it must not cost 30s."""
    # A parent that forks, lets the child exit, and never waits — the launcher
    # behaviour that produces the field reports.
    parent = subprocess.Popen(
        [sys.executable, "-c",
         "import os, time\n"
         "pid = os.fork()\n"
         "if pid == 0: os._exit(0)\n"
         "print(pid, flush=True)\n"
         "time.sleep(60)\n"],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert parent.stdout is not None
        zombie_pid = int(parent.stdout.readline().strip())

        deadline = time.monotonic() + 10
        state = ""
        while time.monotonic() < deadline:
            try:
                status = Path(f"/proc/{zombie_pid}/status").read_text(encoding="utf-8")
            except OSError:
                break
            state = next((ln.split(":", 1)[1].strip()
                          for ln in status.splitlines() if ln.startswith("State:")), "")
            if state.startswith("Z"):
                break
            time.sleep(0.1)
        if not state.startswith("Z"):
            pytest.skip(f"could not stage a zombie (state={state!r})")

        # The premise: existence alone cannot tell this apart from a live window.
        os.kill(zombie_pid, 0)  # raises if the PID is gone; it is not

        # The contract: the gate must still treat it as exited.
        assert _gate_says_alive(zombie_pid) is False
    finally:
        parent.kill()
        parent.wait()
