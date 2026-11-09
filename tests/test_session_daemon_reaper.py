"""The session teardown stops processes left running against this run's Hermes homes.

A test that runs a real ``hermes chat -q`` child gets a grandchild ``gateway run`` in its own
session; no per-test subprocess patch sees it, and pytest's tmp cleanup removes only directories.
"""
from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests import conftest as suite_conftest
from tests.waiting import wait_until


@pytest.mark.platforms("posix")
def test_process_scan_continues_after_wrapped_permission_denial(tmp_path, monkeypatch):
    import psutil

    def denied_environ():
        raise SystemError("proc_environ returned a result with an exception set") from PermissionError(13, "denied")

    denied = SimpleNamespace(pid=-1, environ=denied_environ)
    owned = SimpleNamespace(pid=-2, environ=lambda: {"HERMES_HOME": str(tmp_path / "session")})
    outside = SimpleNamespace(pid=-3, environ=lambda: {"HERMES_HOME": str(tmp_path.parent / "other")})
    monkeypatch.setattr(psutil, "process_iter", lambda: iter((denied, owned, outside)))

    assert suite_conftest._processes_with_hermes_home_under([tmp_path]) == [owned.pid]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("cause", [None, ValueError("invalid process data")])
def test_process_scan_propagates_unrelated_system_errors(tmp_path, monkeypatch, cause):
    import psutil

    error = SystemError("unexpected process inspection failure")

    def broken_environ():
        raise error from cause

    monkeypatch.setattr(psutil, "process_iter", lambda: iter((SimpleNamespace(pid=-1, environ=broken_environ),)))

    with pytest.raises(SystemError) as caught:
        suite_conftest._processes_with_hermes_home_under([tmp_path])
    assert caught.value is error


def _detached(env_home, marker):
    code = "import time\nwhile True: time.sleep(0.2)\n"
    env = {**os.environ, "HERMES_HOME": str(env_home), "LIVE_REAP_MARKER": marker}
    return subprocess.Popen([sys.executable, "-c", code], env=env, start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.mark.platforms("linux", "macos")
@pytest.mark.live_system_guard_bypass  # signals only the two children this test spawned
def test_session_teardown_terminates_daemons_under_the_session_basetemp_only(tmp_path):
    basetemp, elsewhere = tmp_path / "basetemp", tmp_path / "elsewhere"
    (basetemp / "t0" / "hermes").mkdir(parents=True)
    elsewhere.mkdir()
    inside = _detached(basetemp / "t0" / "hermes", "inside")
    outside = _detached(elsewhere, "outside")
    try:
        # Both children have exec'd with their HERMES_HOME (what the reaper reads via psutil environ()).
        wait_until(lambda: {inside.pid, outside.pid} <= set(suite_conftest._processes_with_hermes_home_under(
            [basetemp, elsewhere])), timeout=10, what='both detached children started')
        config = SimpleNamespace(_tmp_path_factory=SimpleNamespace(_basetemp=basetemp))
        reaped = suite_conftest._reap_session_hermes_processes(config)
        assert inside.pid in reaped and outside.pid not in reaped
        assert inside.wait(timeout=5) is not None
        assert outside.poll() is None, "a process outside this session's homes was signalled"
    finally:
        for proc in (inside, outside):
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
