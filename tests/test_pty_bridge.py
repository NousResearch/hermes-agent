import re
import signal
import sys
import time

import psutil
import pytest

from hermes_cli import pty_bridge


class _FakeProc:
    """Minimal ptyprocess stand-in for unit-testing PtyBridge.close()."""

    fd = 7
    pid = 4321

    def __init__(self):
        self.alive = True
        self.leader_signals = []

    def isalive(self):
        return self.alive

    def kill(self, sig):
        self.leader_signals.append(sig)
        self.alive = False

    def close(self, force=False):
        pass


def test_close_signals_process_group_not_the_leader(monkeypatch):
    group_signals = []
    proc = _FakeProc()
    proc.kill = lambda sig: pytest.fail("must signal the group, not the leader")  # type: ignore

    monkeypatch.setattr(pty_bridge.os, "getpgid", lambda pid: 9999)
    monkeypatch.setattr(pty_bridge.os, "getpgrp", lambda: 1)  # the dashboard's own group

    def fake_killpg(pgid, sig):
        assert pgid == 9999, "must never signal a different (e.g. the dashboard's) group"
        if sig == 0:  # liveness probe
            if not proc.alive:
                raise ProcessLookupError
            return
        group_signals.append(sig)
        proc.alive = False

    monkeypatch.setattr(pty_bridge.os, "killpg", fake_killpg)

    pty_bridge.PtyBridge(proc).close()
    assert group_signals  # the whole group was signalled


def test_close_falls_back_to_leader_when_pgid_unavailable(monkeypatch):
    proc = _FakeProc()
    monkeypatch.setattr(pty_bridge.os, "getpgid", lambda pid: (_ for _ in ()).throw(OSError()))

    bridge = pty_bridge.PtyBridge(proc)
    assert bridge._pgid is None
    bridge.close()
    assert proc.leader_signals and proc.leader_signals[0] == signal.SIGHUP


def test_pgid_not_captured_when_it_is_the_dashboards_own_group(monkeypatch):
    monkeypatch.setattr(pty_bridge.os, "getpgid", lambda pid: 555)
    monkeypatch.setattr(pty_bridge.os, "getpgrp", lambda: 555)  # leader shares our group
    # Capturing it would make close() killpg our own group — never do that.
    assert pty_bridge.PtyBridge(_FakeProc())._pgid is None


# Grandchild that outlives a leader-only kill — the whole point of group teardown.
_LEADER = (
    "import os, sys, subprocess, time;"
    "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
    "sys.stdout.write('GPID=%d\\n' % g.pid); sys.stdout.flush();"
    "time.sleep(60)"
)


@pytest.mark.integration
@pytest.mark.live_system_guard_bypass
def test_close_reaps_grandchild_via_process_group():
    if not pty_bridge.PtyBridge.is_available():
        pytest.skip("no PTY on this platform")

    bridge = pty_bridge.PtyBridge.spawn([sys.executable, "-c", _LEADER])
    try:
        buf = ""
        deadline = time.monotonic() + 5
        while "GPID=" not in buf and time.monotonic() < deadline:
            data = bridge.read(timeout=0.2)
            if data:
                buf += data.decode("utf-8", "ignore")
        match = re.search(r"GPID=(\d+)", buf)
        assert match, f"grandchild pid never reported: {buf!r}"
        grandchild = int(match.group(1))
        assert psutil.pid_exists(grandchild)

        bridge.close()

        deadline = time.monotonic() + 5
        while psutil.pid_exists(grandchild) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not psutil.pid_exists(grandchild) or (
            psutil.Process(grandchild).status() == psutil.STATUS_ZOMBIE
        ), "grandchild survived the process-group teardown"
    finally:
        bridge.close()
