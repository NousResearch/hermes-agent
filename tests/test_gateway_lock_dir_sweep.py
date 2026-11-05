"""The conftest sweep of per-process gateway lock dirs never signals a PID.

``tests/conftest.py`` removes the ``HERMES_GATEWAY_LOCK_DIR`` of every pytest
process that is gone. It used to probe each PID with ``os.kill(pid, 0)``,
which on Windows is ``CTRL_C_EVENT`` to the target's console process group
(bpo-14484): it either interrupts a live sibling worker or raises ``OSError``,
which the sweep read as "dead" and deleted that worker's directory.
"""

from __future__ import annotations

import os
import subprocess
import sys

from tests._fixtures.gateway_lock_dirs import sweep_stale_lock_dirs


def _exited_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_sweep_keeps_live_dirs_removes_dead_ones_and_never_signals(tmp_path, monkeypatch):
    signalled = []

    def _no_signals(pid, sig):
        signalled.append((pid, sig))
        raise OSError("os.kill must not be used as a liveness probe")

    monkeypatch.setattr(os, "kill", _no_signals)

    prefix = "hermes-test-gateway-locks-"
    live = tmp_path / f"{prefix}{os.getpid()}"
    dead = tmp_path / f"{prefix}{_exited_pid()}"
    unrelated = tmp_path / f"{prefix}not-a-pid"
    for d in (live, dead, unrelated):
        d.mkdir()

    sweep_stale_lock_dirs(tmp_path, prefix)

    assert signalled == []
    assert live.is_dir()
    assert not dead.exists()
    assert unrelated.is_dir()
