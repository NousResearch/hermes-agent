"""Contracts for SSH-isolated cron admission during idle-exit drain."""

from __future__ import annotations

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def test_admission_gate_waits_for_dispatch_before_draining():
    from hermes_cli.web_server_idle_exit import CronAdmissionGate

    gate = CronAdmissionGate()
    with gate.dispatch_guard() as admitted:
        assert admitted is True
        assert gate.try_begin_drain() is False
        assert gate.can_dispatch() is True

    assert gate.try_begin_drain() is True
    assert gate.can_dispatch() is False
    gate.cancel_drain()
    assert gate.can_dispatch() is True


def test_admission_gate_rejects_new_dispatch_after_drain():
    from hermes_cli.web_server_idle_exit import CronAdmissionGate

    gate = CronAdmissionGate()
    assert gate.try_begin_drain() is True

    with gate.dispatch_guard() as admitted:
        assert admitted is False


def test_admission_gate_close_is_terminal_and_irreversible():
    """Terminal teardown close is unconditional, irreversible, and distinguishable from watchdog drain."""
    from hermes_cli.web_server_idle_exit import CronAdmissionGate

    gate = CronAdmissionGate()
    with gate.dispatch_guard() as admitted:
        assert admitted is True
        # Close while dispatch reservation is currently active
        gate.close()
        assert gate.is_closed is True
        assert gate.can_dispatch() is False

        # Any new entry is immediately rejected even while older reservation is finishing
        with gate.dispatch_guard() as second_admitted:
            assert second_admitted is False

        # Watchdog cancel_drain must NOT reopen admission if closed permanently
        gate.cancel_drain()
        assert gate.can_dispatch() is False
        assert gate.is_closed is True

    # After active reservation exits, admission remains closed and try_begin_drain fails
    assert gate.can_dispatch() is False
    assert gate.try_begin_drain() is False
    gate.cancel_drain()
    assert gate.can_dispatch() is False


def test_admission_gate_close_without_active_reservation():
    """Terminal teardown with no active reservation closes permanently and ignores cancel_drain."""
    from hermes_cli.web_server_idle_exit import CronAdmissionGate

    gate = CronAdmissionGate()
    gate.close()
    assert gate.is_closed is True
    assert gate.can_dispatch() is False
    assert gate.try_begin_drain() is False

    with gate.dispatch_guard() as admitted:
        assert admitted is False

    gate.cancel_drain()
    assert gate.can_dispatch() is False


def test_desktop_ticker_wires_admission_gate_to_builtin_provider(monkeypatch):
    from cron.scheduler_provider import InProcessCronScheduler
    from hermes_cli import web_server
    from hermes_cli.web_server_idle_exit import CronAdmissionGate

    stop = threading.Event()
    gate = CronAdmissionGate()
    received = {}

    def fake_start(_provider, stop_event, **kwargs):
        received.update(kwargs)
        stop_event.set()

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: InProcessCronScheduler(),
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(InProcessCronScheduler, "start", fake_start)

    web_server._start_desktop_cron_ticker(stop, interval=0, admission_gate=gate)

    assert received["can_dispatch"] == gate.can_dispatch
    assert received["dispatch_guard"] == gate.dispatch_guard


def test_inprocess_ticker_holds_admission_guard_around_tick():
    from cron.scheduler_provider import InProcessCronScheduler

    stop = threading.Event()
    guard_entered = threading.Event()
    guard_active = []

    @contextlib.contextmanager
    def dispatch_guard():
        guard_active.append(True)
        guard_entered.set()
        try:
            yield True
        finally:
            guard_active.pop()

    def fake_tick(*_args, **_kwargs):
        assert guard_active == [True]
        stop.set()
        return 0

    with patch("cron.scheduler.tick", side_effect=fake_tick):
        thread = threading.Thread(
            target=InProcessCronScheduler().start,
            args=(stop,),
            kwargs={"interval": 0, "dispatch_guard": dispatch_guard},
            daemon=True,
        )
        thread.start()
        assert guard_entered.wait(timeout=15)
        thread.join(timeout=15)

    assert not thread.is_alive()
    assert guard_active == []


def test_multiplex_ticker_teardown_closes_admission_and_blocks_second_profile(tmp_path):
    """Multiplex ticker teardown closes admission during an active profile tick, preventing subsequent profile jobs."""
    import cron.scheduler as s
    import cron.scheduler_provider as p
    from hermes_cli.web_server_idle_exit import CronAdmissionGate
    from hermes_constants import get_hermes_home

    gate = CronAdmissionGate()
    stop = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    submitted = threading.Event()

    homes = [("first", tmp_path / "first"), ("second", tmp_path / "second")]
    for _, h in homes:
        h.mkdir(parents=True, exist_ok=True)

    errors = []

    def due():
        if Path(get_hermes_home()).resolve() == homes[0][1].resolve():
            entered.set()
            assert release.wait(15)
            return []
        return [{
            "id": "probe-job",
            "name": "synthetic",
            "schedule": {"kind": "interval", "minutes": 1},
        }]

    def worker(job, adapters, loop, verbose):
        assert stop.is_set()
        submitted.set()
        return True

    def run():
        try:
            p.InProcessCronScheduler().start(
                stop,
                interval=60,
                profile_homes=homes,
                can_dispatch=gate.can_dispatch,
                dispatch_guard=gate.dispatch_guard,
            )
        except BaseException as exc:
            errors.append(repr(exc))

    with ThreadPoolExecutor(max_workers=1) as pool, \
         patch.object(s, "_get_parallel_pool", return_value=pool), \
         patch.object(s, "_process_due_job", side_effect=worker), \
         patch.object(s, "_should_yield_tick_to_fresh_gateway", return_value=None), \
         patch.object(s, "_maybe_run_worktree_maintenance"), \
         patch.object(s, "_maybe_reap_dead_owners"), \
         patch.object(s, "_sweep_mcp_orphans"), \
         patch.object(s, "get_due_jobs", side_effect=due):
        t = threading.Thread(target=run, daemon=True)
        t.start()
        try:
            assert entered.wait(15), errors
            # Teardown ordering: close gate permanently, then signal stop
            gate.close()
            stop.set()
            release.set()
            t.join(15)
        finally:
            stop.set()
            release.set()
            t.join(15)

    assert not t.is_alive() and not errors, errors
    assert not submitted.is_set(), "new profile job submitted after teardown stop"


def test_desktop_lifespan_teardown_closes_admission_gate(monkeypatch, _isolate_hermes_home):
    """FastAPI lifespan on Desktop initializes CronAdmissionGate and closes it upon shutdown."""
    from starlette.testclient import TestClient
    import hermes_cli.web_server as ws

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setattr(ws, "_warm_gateway_module", lambda: None)
    monkeypatch.setattr(ws, "_start_desktop_cron_ticker", lambda *_args, **_kwargs: None)
    import hermes_cli.gateway as g
    monkeypatch.setattr(g, "_reap_unsupervised_gateway_orphans", lambda: True)

    client = TestClient(ws.app)
    with client:
        gate = ws.app.state.cron_admission_gate
        assert gate is not None
        assert gate.is_closed is False
        assert gate.can_dispatch() is True

    assert gate.is_closed is True
    assert gate.can_dispatch() is False


def test_idle_watchdog_refuses_drain_while_busy():
    """Watchdog refuses to exit while a ticker dispatch reservation is in flight."""
    from hermes_cli.web_server_idle_exit import (
        CronAdmissionGate,
        IdleClientTracker,
        start_idle_watchdog,
    )

    clock = {"value": 0.0}
    tracker = IdleClientTracker(now=lambda: clock["value"])
    clock["value"] = 10.0
    server = SimpleNamespace(should_exit=False)
    gate = CronAdmissionGate()
    refused_probe_invoked = threading.Event()

    with gate.dispatch_guard() as admitted:
        assert admitted is True

        def probe():
            refused_probe_invoked.set()
            return False

        watchdog = start_idle_watchdog(
            server,
            tracker,
            grace_s=1.0,
            poll_s=0.01,
            probe=probe,
            admission_gate=gate,
        )
        assert refused_probe_invoked.wait(timeout=15)
        assert server.should_exit is False
        assert gate.can_dispatch() is True

    # Once the reservation is released, the watchdog can drain and initiate exit
    watchdog.join(timeout=15)
    assert server.should_exit is True
    assert gate.can_dispatch() is False


def test_idle_watchdog_can_use_admission_gate_before_exit():
    from hermes_cli.web_server_idle_exit import (
        CronAdmissionGate,
        IdleClientTracker,
        start_idle_watchdog,
    )

    clock = {"value": 0.0}
    tracker = IdleClientTracker(now=lambda: clock["value"])
    clock["value"] = 10.0
    server = SimpleNamespace(should_exit=False)
    gate = CronAdmissionGate()

    start_idle_watchdog(
        server,
        tracker,
        grace_s=1.0,
        poll_s=0.01,
        probe=lambda: False,
        admission_gate=gate,
    ).join(timeout=15)

    assert server.should_exit is True
    assert gate.can_dispatch() is False


def test_idle_watchdog_reopens_admission_when_liveness_changes():
    from hermes_cli.web_server_idle_exit import (
        CronAdmissionGate,
        IdleClientTracker,
        start_idle_watchdog,
    )

    clock = {"value": 0.0}
    tracker = IdleClientTracker(now=lambda: clock["value"])
    clock["value"] = 10.0
    server = SimpleNamespace(should_exit=False)
    gate = CronAdmissionGate()
    probes = []

    def probe():
        probes.append(True)
        if len(probes) == 2:
            server.should_exit = True
            return True
        return False

    start_idle_watchdog(
        server,
        tracker,
        grace_s=1.0,
        poll_s=0.01,
        probe=probe,
        admission_gate=gate,
    ).join(timeout=15)

    assert server.should_exit is True
    assert gate.can_dispatch() is True
    assert len(probes) == 2
