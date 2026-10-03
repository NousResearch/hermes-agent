"""Regression cover for #120691: one app launch must never leave two managed routers running.

The managed llama-server router spawns per-model children that hold the weights resident. When a
replacement boot stops an incumbent and spawns regardless, both routers stay up and each autoloads
its own copy of the model (observed: 2x26 GB on a 64 GB machine). The stop must reap the whole
tree and must report honestly, and the boot must refuse to spawn beside a still-live incumbent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil

from hermes_cli.local_runtime import bootstrap


def _alive(pid: int) -> bool:
    if not psutil.pid_exists(pid):
        return False
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _spawn_router_shaped_proc():
    """A parent process with one child (router + model child shape), both alive for a minute."""
    code = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "time.sleep(60)"
    )
    parent = subprocess.Popen([sys.executable, "-c", code])
    child = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            kids = psutil.Process(parent.pid).children(recursive=True)
        except psutil.Error:
            kids = []
        if kids:
            child = kids[0]
            break
        time.sleep(0.1)
    assert child is not None, "child never appeared"
    return parent, child


def _cleanup(*procs) -> None:
    for p in procs:
        try:
            psutil.Process(p.pid if hasattr(p, "pid") else p).kill()
        except psutil.Error:
            pass


def test_stop_state_server_reaps_router_and_model_children():
    parent, child = _spawn_router_shaped_proc()
    try:
        assert bootstrap._stop_state_server({"pid": parent.pid}) is True
        for _ in range(50):
            if not _alive(parent.pid) and not _alive(child.pid):
                break
            time.sleep(0.1)
        assert not _alive(parent.pid), "router survived the stop"
        assert not _alive(child.pid), "model child survived the stop (weights stay resident)"
    finally:
        _cleanup(parent, child)


def test_stop_state_server_reports_a_surviving_incumbent(monkeypatch):
    parent, child = _spawn_router_shaped_proc()
    # A router that refuses to die: every signal is a no-op. The verdict must be False so the
    # caller does not spawn a replacement beside it.
    monkeypatch.setattr(psutil.Process, "terminate", lambda self: None)
    monkeypatch.setattr(psutil.Process, "kill", lambda self: None)
    monkeypatch.setattr(bootstrap.os, "kill", lambda *a, **kw: None)
    try:
        assert bootstrap._stop_state_server({"pid": parent.pid}) is False
    finally:
        monkeypatch.undo()
        _cleanup(parent, child)


def test_stop_state_server_treats_a_pid_free_record_as_stopped():
    # A record that never named a usable pid describes no live router of ours: the boot may
    # proceed (a spawn beside THIS is not a second router, because there is no router).
    assert bootstrap._stop_state_server({"pid": -1}) is True
    assert bootstrap._stop_state_server({"pid": "not-a-pid"}) is False, (
        "a pid we cannot read is one we cannot verify gone — refuse to replace")
    assert bootstrap._stop_state_server({}) is False, (
        "an endpoint dict that dropped the pid must not be reported as verified-gone")


def test_ensure_never_boots_a_second_router_beside_a_live_incumbent(monkeypatch, tmp_path):
    """The vacuity probe: this must FAIL with the guard deleted from ensure_local_runtime.

    Drives the real code path: a real state file (written the way supervisor._write_state writes
    it — modern identity fields included, so recorded_process() verifies the live incumbent) and
    a REAL live incumbent that ignores our signals for the stop window. The real
    _state_endpoint() resolves it (and now carries the pid), a stubbed installed_engine lets the
    boot proceed past the no-engine early return, and the Recorder supervisor's construction is
    exactly the bug the guard must prevent.
    """
    import hermes_cli.local_runtime.recovery as recovery
    import hermes_cli.local_runtime.supervisor as supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path, raising=False)
    # Staleness forces the replace path; without it the adopt branch returns before the guard.
    monkeypatch.setattr(bootstrap, "_presets_stale", lambda: True)
    # Ensure_local_runtime's spawned supervisor would read models_dir() under the real home.
    monkeypatch.setattr(bootstrap, "models_dir", lambda: tmp_path / "models")

    # A live incumbent that ignores every signal we send within the stop window: the real
    # _stop_state_server must run against it and report False.
    parent, child = _spawn_router_shaped_proc()
    proc = psutil.Process(parent.pid)
    state = {"base_url": "http://127.0.0.1:18434/v1", "api_key": "k",
             "pid": proc.pid, "create_time": proc.create_time(), "executable": proc.exe(),
             "owner_pid": os.getpid(), "owner_create_time": psutil.Process().create_time()}
    supervisor.state_path().parent.mkdir(parents=True, exist_ok=True)
    supervisor.state_path().write_text(json.dumps(state), encoding="utf-8")

    # "Llama-server ignores SIGTERM for the stop window": signal delivery is real, death is not.
    monkeypatch.setattr(psutil.Process, "terminate", lambda self: None)
    monkeypatch.setattr(psutil.Process, "kill", lambda self: None)

    booted = []

    class _Recorder:
        def __init__(self, *a, **kw):
            booted.append(a)

    monkeypatch.setattr(supervisor, "LlamaServerSupervisor", _Recorder)
    # A fake engine so the boot proceeds past the no-engine early return to the supervisor
    # constructor — exactly where the guard must have refused to go.
    monkeypatch.setattr("hermes_cli.local_runtime.binaries.installed_engine",
                        lambda *a, **k: SimpleNamespace(binary=tmp_path / "llama-server"))

    try:
        result = bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True}}, force=True)
        assert result is None, "a still-live incumbent must be adopted, never replaced by a spawn"
        assert not booted, "a second router was booted beside a live incumbent"
    finally:
        monkeypatch.undo()
        _cleanup(parent, child)
