"""Regression cover for #120691: one app launch must never leave two managed routers running.

The managed llama-server router spawns per-model children that hold the weights resident. When a
replacement boot stops an incumbent and spawns regardless, both routers stay up and each autoloads
its own copy of the model (observed: 2x26 GB on a 64 GB machine). The stop must reap the whole
tree and must report honestly, and the boot must refuse to spawn beside a still-live incumbent.
"""
from __future__ import annotations

import subprocess
import sys
import time

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


def test_stop_state_server_treats_a_missing_router_as_stopped():
    assert bootstrap._stop_state_server({"pid": -1}) is True
    assert bootstrap._stop_state_server({"pid": "not-a-pid"}) is True
    assert bootstrap._stop_state_server({}) is True


def test_ensure_never_boots_a_second_router_beside_a_live_incumbent(monkeypatch, tmp_path):
    import hermes_cli.local_runtime.endpoint as endpoint
    import hermes_cli.local_runtime.supervisor as supervisor

    monkeypatch.setattr(bootstrap, "runtimes_root", lambda: tmp_path, raising=False)
    monkeypatch.setattr(bootstrap, "_presets_stale", lambda: True)
    monkeypatch.setattr(bootstrap, "_stop_state_server", lambda state: False)
    monkeypatch.setattr(endpoint, "_state_endpoint",
                        lambda: {"base_url": "http://127.0.0.1:18434/v1", "pid": 12345})
    booted = []

    class _Recorder:
        def __init__(self, *a, **kw):
            booted.append(a)

    monkeypatch.setattr(supervisor, "LlamaServerSupervisor", _Recorder)

    result = bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True}}, force=True)
    assert result is None, "a still-live incumbent must be adopted, never replaced by a spawn"
    assert not booted, "a second router was booted beside a live incumbent"
