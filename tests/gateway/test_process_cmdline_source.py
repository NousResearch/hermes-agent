"""``_read_process_cmdline`` asks the cheapest source that can answer, and still answers.

The gateway-identity check reads a live PID's command line, and `list_profiles` runs it per profile
with a live gateway — on every 5s roster poll, per connection. On macOS there is no ``/proc``, so
this used to fork ``ps`` every time (measured 4.2ms) when psutil answers the same string in 0.02ms.
psutil cannot always answer (macOS raises ``AccessDenied`` across users), so ``ps`` must remain the
fallback, not be replaced.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from gateway import status


@pytest.fixture
def no_proc(monkeypatch):
    """Force the non-Linux shape: /proc unreadable, so the source choice is psutil vs ps."""
    real_read_bytes = status.Path.read_bytes

    def _read_bytes(self, *a, **kw):
        if str(self).startswith("/proc/"):
            raise OSError("no /proc")
        return real_read_bytes(self, *a, **kw)

    monkeypatch.setattr(status.Path, "read_bytes", _read_bytes)


def _no_fork(monkeypatch):
    """Fail loudly if anything forks a subprocess."""
    def _boom(*a, **kw):
        raise AssertionError(f"forked a subprocess: {a[0] if a else kw}")

    monkeypatch.setattr(status.subprocess, "run", _boom)


def test_psutil_answers_without_forking_ps(no_proc, monkeypatch):
    _no_fork(monkeypatch)

    cmdline = status._read_process_cmdline(os.getpid())

    assert cmdline and sys.executable.split("/")[-1] in cmdline


def test_ps_still_answers_when_psutil_cannot(no_proc, monkeypatch):
    """macOS raises AccessDenied across users; ps still reports those, so it stays the fallback."""
    import psutil

    def _denied(_pid):
        raise psutil.AccessDenied(_pid)

    monkeypatch.setattr(psutil, "Process", _denied)
    forked: list = []
    real_run = status.subprocess.run

    def _watching(cmd, *a, **kw):
        forked.append(cmd)
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(status.subprocess, "run", _watching)

    cmdline = status._read_process_cmdline(os.getpid())

    assert cmdline, "the ps fallback must still answer when psutil refuses"
    assert forked and forked[0][:2] == ["ps", "-p"]


def test_a_dead_pid_reports_nothing_from_either_source(no_proc):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()

    assert status._read_process_cmdline(dead.pid) in (None, "")


def test_both_sources_agree_on_a_live_process(no_proc, monkeypatch):
    """The consumer matches on this string, so the two sources must not disagree."""
    import psutil

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "gateway", "run"])
    try:
        for _ in range(50):
            if psutil.pid_exists(proc.pid):
                break
            time.sleep(0.05)
        via_psutil = status._read_process_cmdline(proc.pid)

        denied = psutil.AccessDenied

        def _raise(_pid):
            raise denied(_pid)

        monkeypatch.setattr(psutil, "Process", _raise)
        via_ps = status._read_process_cmdline(proc.pid)
    finally:
        proc.kill()
        proc.wait()

    assert via_psutil == via_ps
