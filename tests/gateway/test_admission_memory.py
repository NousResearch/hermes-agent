"""Memory-aware turn admission (gateway/admission_memory.py) and child OOM preference."""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from gateway import admission_memory as am
from hermes_platform.memory import MemoryReading

MB = 1024 * 1024


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _gate(free_mb, *, ceiling_mb=4096, **settings):
    state = {"free": free_mb}
    clock = _Clock()
    gate = am.MemoryGate(
        settings=am.MemoryGuardSettings(**{"critical_mb": 256, "low_mb": 512, "reserve_mb": 0, **settings}),
        reader=lambda: MemoryReading(state["free"] * MB, ceiling_mb * MB, "cgroup"),
        clock=clock,
    )
    return gate, state, clock


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_people_start_at_low_while_background_waits_behind_them():
    gate, state, clock = _gate(400)  # between critical (256) and low (512)
    assert gate.try_admit(am.PERSON)
    assert not gate.try_admit(am.BACKGROUND)
    state["free"], clock.now = 100, clock.now + 2  # critical: the person waits too
    person = gate.enqueue(am.PERSON)
    state["free"], clock.now = 4000, clock.now + 2
    # Plenty of memory again, but a person is still queued: background may not jump ahead.
    assert not gate.try_admit(am.BACKGROUND)
    assert gate.try_admit(am.PERSON, person)
    assert gate.try_admit(am.BACKGROUND)


def test_reservation_stops_a_released_burst_from_starting_into_the_same_headroom():
    gate, _, clock = _gate(700, reserve_mb=300)
    assert gate.try_admit(am.PERSON)  # 700 free -> reserves 300
    assert gate.try_admit(am.PERSON)  # 400 effective >= 256
    assert not gate.try_admit(am.PERSON)  # 100 effective < 256
    clock.now += am._RESERVATION_WINDOW_S + 1
    assert gate.try_admit(am.PERSON)


def test_wait_is_bounded_and_publishes_waiting_count(tmp_path):
    gate, _, clock = _gate(10, max_wait_seconds=5)
    gate.status_path = tmp_path / "state" / "memory_guard.json"
    seen = []

    def sleep(_):
        gate.flush_status()
        seen.append(am.read_status(tmp_path, os.getpid())["waiting_people"])
        clock.now += 2

    assert am.wait_blocking(am.PERSON, gate=gate, sleep=sleep) == "timeout"
    assert seen and seen[0] == 1
    gate.flush_status()
    assert am.read_status(tmp_path, os.getpid())["waiting_people"] == 0
    assert am.wait_blocking(am.PERSON, gate=gate, still_wanted=lambda: False, sleep=sleep) == "abandoned"


def test_unknown_memory_never_blocks():
    gate = am.MemoryGate(reader=lambda: None)
    assert gate.try_admit(am.BACKGROUND)


def test_disabled_guard_admits_everything():
    gate, _, _ = _gate(1, enabled=False)
    assert gate.try_admit(am.BACKGROUND)


class _Runner:
    from gateway.run_inbound_hooks import GatewayInboundHooksMixin as _M

    _hm_memory_admission = _M._hm_memory_admission

    def __init__(self):
        self.sent = []
        self.generation = 1

    def _delivery_adapter_for(self, source):
        return object()

    async def _send_busy_reply(self, event, adapter, text):
        self.sent.append(text)

    def _is_session_run_current(self, key, generation):
        return generation == self.generation


def _event():
    from gateway.config import Platform

    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_id="1")
    return SimpleNamespace(source=source), source


def test_inbound_turn_waits_with_notice_then_refuses_after_the_bound(monkeypatch):
    gate, _, clock = _gate(10, max_wait_seconds=60)
    monkeypatch.setattr(am, "_gate", gate)

    async def fast_sleep(_):
        clock.now += 30

    monkeypatch.setattr(am.asyncio, "sleep", fast_sleep)
    runner, (event, source) = _Runner(), _event()
    reply = asyncio.run(runner._hm_memory_admission(event, source, "k", False, 1))
    assert len(runner.sent) == 1 and "memory" in runner.sent[0]
    assert reply and "memory" in reply  # refused, not started


def test_inbound_turn_starts_once_memory_frees(monkeypatch):
    gate, state, clock = _gate(10, max_wait_seconds=600)
    monkeypatch.setattr(am, "_gate", gate)

    async def fast_sleep(_):
        clock.now += 2
        state["free"] = 2048

    monkeypatch.setattr(am.asyncio, "sleep", fast_sleep)
    runner, (event, source) = _Runner(), _event()
    assert asyncio.run(runner._hm_memory_admission(event, source, "k", False, 1)) is None
    assert len(runner.sent) == 1


def test_an_older_background_waiter_never_starts_ahead_of_a_waiting_person():
    gate, state, clock = _gate(100)
    background = gate.enqueue(am.BACKGROUND)
    person = gate.enqueue(am.PERSON)
    state["free"], clock.now = 4000, clock.now + 2
    assert not gate.try_admit(am.BACKGROUND, background), "background polled first and jumped the person"
    assert gate.try_admit(am.PERSON, person)
    assert gate.try_admit(am.BACKGROUND, background)


def test_status_is_written_outside_the_gate_lock(tmp_path, monkeypatch):
    import utils
    gate, _, _ = _gate(10)
    gate.status_path = tmp_path / "state" / "memory_guard.json"
    held_lock = []
    monkeypatch.setattr(utils, "atomic_json_write", lambda path, data: held_lock.append(gate._lock.locked()))
    gate.try_admit(am.PERSON)
    getattr(gate, "flush_status", lambda: None)()
    assert held_lock == [False], "the status file write (with fsync) ran under the admission lock"


def test_status_trusts_verified_gateway_discovery_not_the_snapshot_file(tmp_path, monkeypatch):
    monkeypatch.setattr(am, "read_memory", lambda pid=None: MemoryReading(2048 * MB, 4096 * MB, "cgroup"))
    # A fresh idle gateway that has published nothing yet is running, with nobody waiting.
    monkeypatch.setattr(am, "_running_gateway_pid", lambda home: 4242, raising=False)
    assert "0 waiting" in am.status_line({}, tmp_path)
    # A snapshot naming a live but unrelated PID (reused) is not a gateway.
    _write(tmp_path / "state" / "memory_guard.json", json.dumps({"pid": os.getpid(), "waiting_people": 3}))
    monkeypatch.setattr(am, "_running_gateway_pid", lambda home: None, raising=False)
    line = am.status_line({}, tmp_path)
    assert "no gateway running" in line and "3 waiting" not in line


def test_memory_guard_settings_load_from_config_yaml_at_gateway_start(tmp_path, monkeypatch):
    from gateway.run import _load_gateway_config
    (tmp_path / "config.yaml").write_text(
        "gateway:\n  memory_guard:\n    enabled: false\n    critical_mb: 100\n    reserve_mb: 64\n", encoding="utf-8")
    gate = am.configure(_load_gateway_config(tmp_path / "config.yaml"), tmp_path)
    assert (gate.settings.enabled, gate.settings.critical_mb, gate.settings.reserve_mb) == (False, 100, 64)
    assert gate.settings.low_mb == "auto"


def test_a_managed_worker_turn_waits_for_memory_before_spawning(tmp_path, monkeypatch):
    from gateway import session_managed_worker

    gate, _, clock = _gate(10, max_wait_seconds=2)
    monkeypatch.setattr(am, "_gate", gate)

    async def fast_sleep(_):
        clock.now += 2

    monkeypatch.setattr(am.asyncio, "sleep", fast_sleep)
    spawned = []

    async def execute_managed(*args):
        spawned.append(args)

    monkeypatch.setattr(session_managed_worker, "execute_managed", execute_managed)
    monkeypatch.setattr("gateway.session_policy.workspace_ready", lambda policy: True)
    authority = SimpleNamespace(pending_results={}, pending_stops={}, executing={"s": 1})
    row = {"admission_id": "a1", "generation": 1, "payload": {"text": "hi"}}
    policy = SimpleNamespace(kanban_json=None, cwd=str(tmp_path))
    reply = asyncio.run(session_managed_worker.admit_managed(authority, SimpleNamespace(session_id="s"), row, policy))
    assert spawned == [], "a managed worker spawned below the memory floor"
    assert reply and "memory" in reply and authority.pending_results["a1"]["result"]["final_response"] == reply
