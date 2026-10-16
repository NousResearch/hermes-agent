"""The router watchdog gives up instead of respawning forever.

A router that can never come back (its port now served by another process, its model files gone,
the runtime switched off) used to be respawned once a minute for the life of the owning process.
"""
from __future__ import annotations

import socket
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def crashing(tmp_path, monkeypatch):
    """A supervised router that dies on every spawn, with no real backoff waits."""
    from hermes_cli import config
    from hermes_cli.local_runtime import supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path)
    runtime = {"enabled": True}
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"local_runtime": runtime})
    held_ports = set()
    monkeypatch.setattr(supervisor, "_port_in_use", held_ports.__contains__, raising=False)
    sup = supervisor.LlamaServerSupervisor(tmp_path, tmp_path, port=59998)
    sup.proc = SimpleNamespace(pid=100, poll=lambda: 1)
    spawned = []

    def spawn():
        spawned.append(True)
        sup.proc = SimpleNamespace(pid=100 + len(spawned), poll=lambda: 1)
        if len(spawned) > 50:  # a runaway watchdog fails the test instead of hanging it
            sup.stop()

    def never_healthy(*_):
        raise RuntimeError("llama-server exited rc=1 during startup")

    monkeypatch.setattr(sup, "_spawn", spawn)
    monkeypatch.setattr(sup, "_wait_health", never_healthy)
    monkeypatch.setattr(sup, "_reap_orphaned_children", lambda: None)
    monkeypatch.setattr(sup._stop_event, "wait", lambda *_: False)
    return SimpleNamespace(sup=sup, spawned=spawned, runtime=runtime, held_ports=held_ports)


def test_restarts_stop_at_the_cap(crashing):
    from hermes_cli.local_runtime import supervisor

    crashing.sup._watch()
    assert len(crashing.spawned) == supervisor._MAX_RESTARTS


def test_a_stable_run_earns_a_fresh_restart_budget(crashing, monkeypatch):
    from hermes_cli.local_runtime import supervisor

    polls = iter([None, 1])  # up for a long while, then crashes
    crashing.sup.proc = SimpleNamespace(pid=100, poll=lambda: next(polls, 1))
    crashing.sup._restarts = supervisor._MAX_RESTARTS  # budget spent on crashes long ago
    crashing.sup._spawned_at = time.monotonic() - supervisor._RESTART_RESET_S
    monkeypatch.setattr(supervisor.time, "sleep", lambda *_: None)
    crashing.sup._watch()
    assert len(crashing.spawned) == supervisor._MAX_RESTARTS


def test_no_restart_once_the_runtime_is_switched_off(crashing):
    crashing.runtime["enabled"] = False
    crashing.sup._watch()
    assert crashing.spawned == []


@pytest.mark.parametrize("force,port_taken", [(False, False), (True, False), (True, True)])
def test_boot_force_controls_disabled_restarts(crashing, monkeypatch, tmp_path, force, port_taken):
    from hermes_cli.local_runtime import binaries, bootstrap, endpoint, supervisor

    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "runtimes_root", lambda: tmp_path)
    monkeypatch.setattr(bootstrap, "models_dir", lambda: tmp_path / "models")
    monkeypatch.setattr(bootstrap, "adopt_legacy_models", lambda: [])
    monkeypatch.setattr(bootstrap, "staged_models", lambda: [tmp_path / "model.gguf"])
    monkeypatch.setattr(endpoint, "_state_endpoint", lambda: None)
    monkeypatch.setattr(binaries, "installed_engine", lambda *_: SimpleNamespace(
        binary=tmp_path / "llama-server", backend="cpu", tag="test"))
    monkeypatch.setattr(bootstrap, "_generate_presets", lambda *_: None)
    monkeypatch.setattr(bootstrap, "_admitted_models_max", lambda *_: 4)
    monkeypatch.setattr(bootstrap, "_start_idle_sweeper", lambda *_: None)
    monkeypatch.setattr(supervisor.LlamaServerSupervisor, "start", lambda *_: None)

    sup = bootstrap.ensure_local_runtime(
        {"local_runtime": {"enabled": not force, "port": 59998}}, force=force)
    assert sup is not None
    crashing.runtime["enabled"] = False
    if port_taken:
        crashing.held_ports.add(sup.port)
    sup.proc = SimpleNamespace(pid=100, poll=lambda: 1)
    attempts = []

    def spawn():
        attempts.append(True)
        if len(attempts) > 50:
            pytest.fail("watchdog exceeded the bounded test loop")

    monkeypatch.setattr(sup, "_spawn", spawn)
    monkeypatch.setattr(sup, "_wait_health", crashing.sup._wait_health)
    monkeypatch.setattr(sup, "_reap_orphaned_children", lambda: None)
    monkeypatch.setattr(sup._stop_event, "wait", lambda *_: False)
    sup._watch()
    assert len(attempts) == (supervisor._MAX_RESTARTS if force and not port_taken else 0)


def test_no_restart_onto_a_port_another_process_serves(crashing):
    crashing.held_ports.add(crashing.sup.port)
    crashing.sup._watch()
    assert crashing.spawned == []


def test_port_probe_sees_another_listener():
    from hermes_cli.local_runtime import supervisor

    with socket.socket() as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        assert supervisor._port_in_use(other.getsockname()[1])


def test_start_replaces_a_supervisor_whose_watchdog_gave_up(monkeypatch):
    from hermes_cli.local_runtime import bootstrap

    live = SimpleNamespace(watching=lambda: True)
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", live)
    assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True}}) is live

    stopped = []
    monkeypatch.setattr(bootstrap, "_SUPERVISOR",
                        SimpleNamespace(watching=lambda: False, stop=lambda: stopped.append(True)))
    monkeypatch.setattr(bootstrap, "adopt_legacy_models", lambda: [])
    monkeypatch.setattr(bootstrap, "staged_models", lambda: [])  # end the fresh boot early
    assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True}}) is None
    assert stopped == [True]
    assert bootstrap._SUPERVISOR is None
