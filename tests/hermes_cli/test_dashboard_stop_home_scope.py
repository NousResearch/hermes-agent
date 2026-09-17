"""``hermes dashboard --stop`` is profile-scoped: it must never kill another
install's/profile's dashboards (#113978).

The scoping keys on the target process's real ``HERMES_HOME`` environ
(``_hermes_home_for_pid``), exercised here with real subprocesses carrying
different homes — two homes, A→B→A, not one temp home.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
import time

import pytest

from hermes_cli import dashboard_procs
from hermes_cli.main import cmd_dashboard

_SLEEPER = textwrap.dedent(
    """
    import pathlib, sys, time
    pathlib.Path(sys.argv[1]).write_text("ready")
    time.sleep(300)
    """
)


def _spawn_with_home(home, tmp_path, tag) -> subprocess.Popen:
    ready = tmp_path / f"ready-{tag}"
    env = dict(os.environ, HERMES_HOME=str(home))
    child = subprocess.Popen(
        [sys.executable, "-c", _SLEEPER, str(ready)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env,
    )
    deadline = time.monotonic() + 10.0
    while not ready.exists():
        if child.poll() is not None:
            raise AssertionError("sleeper exited before signaling ready")
        if time.monotonic() > deadline:
            raise AssertionError("sleeper never signaled ready")
        time.sleep(0.02)
    return child


def test_stop_kill_scopes_to_own_hermes_home(tmp_path, monkeypatch):
    """Two live backends under different homes: only the caller's home is killed."""
    home_a, home_b = tmp_path / "homeA", tmp_path / "homeB"
    home_a.mkdir()
    home_b.mkdir()
    child_a = _spawn_with_home(home_a, tmp_path, "a")
    child_b = _spawn_with_home(home_b, tmp_path, "b")
    sent: list[int] = []
    try:
        monkeypatch.setattr(
            dashboard_procs, "_scan_dashboard_processes",
            lambda *, exclude_pids=None: [
                (child_a.pid, "hermes dashboard"), (child_b.pid, "hermes serve")])
        # Record, don't signal: the contract is which PIDs reach the killer.
        monkeypatch.setattr(dashboard_procs, "_kill_pids_posix",
                            lambda pids, killed, failed: sent.extend(pids))
        monkeypatch.setattr(dashboard_procs, "_kill_pids_windows",
                            lambda pids, killed, failed: sent.extend(pids))
        result = dashboard_procs._kill_stale_dashboard_processes(
            reason="test", scope_to_home=str(home_a))
    finally:
        for child in (child_a, child_b):
            child.terminate()
            child.wait(timeout=10)
    assert sent == [child_a.pid]
    assert result["matched"] == [child_a.pid]


def _ns(**kw):
    defaults = dict(port=9119, host="127.0.0.1", no_open=False, insecure=False,
                    stop=False, status=False)
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_stop_wiring_scopes_precheck_and_kill_to_caller_home(monkeypatch):
    """--stop's pre-check and kill both receive the caller's home (respects -p)."""
    from hermes_constants import get_hermes_home
    find_calls: list = []
    kill_kwargs: dict = {}

    def fake_find(**kwargs):
        find_calls.append(kwargs)
        return [424242]

    def fake_kill(*args, **kwargs):
        kill_kwargs.update(kwargs)
        return {"matched": [], "killed": [], "failed": [], "unrecovered": []}

    monkeypatch.setattr("hermes_cli.main._find_stale_dashboard_pids", fake_find)
    monkeypatch.setattr(
        "hermes_cli.dashboard_procs._kill_stale_dashboard_processes", fake_kill)
    with pytest.raises(SystemExit) as exc:
        cmd_dashboard(_ns(stop=True))
    own_home = str(get_hermes_home())
    assert exc.value.code == 0
    assert [c.get("scope_to_home") for c in find_calls] == [own_home]
    assert kill_kwargs.get("scope_to_home") == own_home
