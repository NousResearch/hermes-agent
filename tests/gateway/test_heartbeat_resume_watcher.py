"""Watcher-level tests for heartbeat restart-recovery: that
``_resume_heartbeat_watches`` actually rebuilds ``_heartbeat_watch`` from
persisted state, self-heals a transient failure on a later tick instead of
orphaning the heartbeat forever, and that the resulting watch entry lets a
due heartbeat fire through the real poller — not just that the CLI-layer
persistence (hermes_cli/heartbeat.py) round-trips.

Routing is resolved via ``_gateway_session_origin_for_id`` — the same
durable ``SessionEntry.origin`` lookup ``/resume`` already relies on —
rather than a heartbeat-local routing snapshot, so a heartbeat that
predates this fix (no routing of its own) still recovers, and whatever
``SessionSource`` origin carries (including profile/scope qualifiers) is
handed to the poller unmodified rather than being rebuilt from a reduced
dict.

Exercised against the real GatewayRunner methods bound onto a lightweight
stand-in (booting a full gateway is unnecessary for this logic and would be
slow/flaky), following the pattern in test_scale_to_zero_watcher.py.
"""

from __future__ import annotations

import asyncio

import pytest

import hermes_cli.heartbeat as heartbeat_mod
from gateway.run import GatewayRunner
from hermes_cli.heartbeat import HeartbeatState


def _runner_with(monkeypatch, *, origins=None):
    """``origins`` maps session_id -> SessionSource (or None), mirroring
    ``_gateway_session_origin_for_id``'s persisted-lookup contract.
    """
    origins = origins or {}
    r = GatewayRunner.__new__(GatewayRunner)
    r._running = True
    r._heartbeat_watch = {}
    r._background_tasks = set()
    r._running_agents = {}

    async def _warm(*_a, **_k):
        return None

    monkeypatch.setattr(r, "_warm_goals_session_db", _warm, raising=False)
    monkeypatch.setattr(r, "_start_heartbeat_poller", lambda: None, raising=False)
    monkeypatch.setattr(
        r, "_gateway_session_origin_for_id", lambda sid: origins.get(sid), raising=False
    )
    monkeypatch.setattr(
        r, "_session_key_for_source", lambda _source: "quick_key", raising=False
    )
    return r


def _use_instant_sleep(monkeypatch):
    """Collapse both the startup delay and the tick interval to a scheduler
    yield, so the ``while self._running`` loop can be driven deterministically
    without a real 5s+ wait.
    """
    real_sleep = asyncio.sleep

    async def _instant(_seconds):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _instant)


async def _run_one_tick(r, monkeypatch):
    _use_instant_sleep(monkeypatch)
    task = asyncio.create_task(r._resume_heartbeat_watches(interval=0.01))
    # Two yields: one past the startup sleep, one past the scan + tick sleep.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    r._running = False
    await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_resume_scan_registers_persisted_heartbeat(monkeypatch):
    fake_source = object()
    r = _runner_with(monkeypatch, origins={"sid-1": fake_source})

    state = HeartbeatState(prompt="check ci", interval_seconds=600, status="active")
    monkeypatch.setattr(
        heartbeat_mod, "list_active_heartbeats", lambda: [("sid-1", state)]
    )

    await _run_one_tick(r, monkeypatch)

    assert r._heartbeat_watch == {"quick_key": (fake_source, "sid-1")}


@pytest.mark.asyncio
async def test_resume_scan_rearms_a_legacy_heartbeat_with_no_persisted_route(monkeypatch):
    """A heartbeat set before this fix carries no routing of its own — it
    must still recover as long as its session has a persisted origin, since
    routing no longer comes from the heartbeat row at all.
    """
    fake_source = object()
    r = _runner_with(monkeypatch, origins={"sid-1": fake_source})

    # No route-shaped data on the state at all — this is the legacy shape.
    state = HeartbeatState(prompt="check ci", interval_seconds=600, status="active")
    assert not hasattr(state, "route")
    monkeypatch.setattr(
        heartbeat_mod, "list_active_heartbeats", lambda: [("sid-1", state)]
    )

    await _run_one_tick(r, monkeypatch)

    assert r._heartbeat_watch == {"quick_key": (fake_source, "sid-1")}


@pytest.mark.asyncio
async def test_resume_scan_ignores_sessions_with_no_persisted_origin(monkeypatch):
    """CLI-only sessions (and any session the store has no origin for) have
    nothing to re-arm — they're driven by their own watchdog thread.
    """
    r = _runner_with(monkeypatch, origins={})

    state = HeartbeatState(prompt="check ci", interval_seconds=600, status="active")
    monkeypatch.setattr(
        heartbeat_mod, "list_active_heartbeats", lambda: [("sid-1", state)]
    )

    await _run_one_tick(r, monkeypatch)

    assert r._heartbeat_watch == {}


@pytest.mark.asyncio
async def test_resume_scan_preserves_the_origins_own_qualified_source(monkeypatch):
    """A profile/workspace-qualified session must re-arm through the exact
    SessionSource its SessionEntry.origin carries — not a reconstruction
    that could drop qualifiers like profile or scope_id.
    """
    qualified_source = object()  # stands in for a SessionSource with profile/scope_id set
    seen = []
    r = _runner_with(monkeypatch, origins={"sid-1": qualified_source})
    monkeypatch.setattr(
        r,
        "_session_key_for_source",
        lambda source: seen.append(source) or "quick_key",
        raising=False,
    )

    state = HeartbeatState(prompt="check ci", interval_seconds=600, status="active")
    monkeypatch.setattr(
        heartbeat_mod, "list_active_heartbeats", lambda: [("sid-1", state)]
    )

    await _run_one_tick(r, monkeypatch)

    assert seen == [qualified_source]
    assert r._heartbeat_watch == {"quick_key": (qualified_source, "sid-1")}


@pytest.mark.asyncio
async def test_resume_scan_self_heals_after_a_transient_failure(monkeypatch):
    """A one-shot scan would orphan this heartbeat forever if the very first
    attempt raises. The persistent ticker must retry on the next tick and
    still succeed — this is the correctness gap the reviewer flagged in
    _resume_heartbeat_watches (gateway/run.py).
    """
    fake_source = object()
    r = _runner_with(monkeypatch, origins={"sid-1": fake_source})

    state = HeartbeatState(prompt="check ci", interval_seconds=600, status="active")
    monkeypatch.setattr(
        heartbeat_mod, "list_active_heartbeats", lambda: [("sid-1", state)]
    )

    attempts = {"n": 0}
    real_session_key_for_source = r._session_key_for_source

    def _flaky_session_key(source):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("session_store not warm yet")
        return real_session_key_for_source(source)

    monkeypatch.setattr(r, "_session_key_for_source", _flaky_session_key, raising=False)

    _use_instant_sleep(monkeypatch)
    task = asyncio.create_task(r._resume_heartbeat_watches(interval=0.01))
    # Three yields: startup sleep, the failing first tick, the healing second.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    r._running = False
    await asyncio.wait_for(task, timeout=2)

    assert attempts["n"] >= 2
    assert r._heartbeat_watch == {"quick_key": (fake_source, "sid-1")}


@pytest.mark.asyncio
async def test_resume_scan_leaves_already_registered_sessions_alone(monkeypatch):
    """Idempotency: a session already re-armed (by an earlier tick, or by the
    user re-touching /heartbeat) must not be re-scanned/re-registered.
    """
    r = _runner_with(monkeypatch, origins={"sid-1": object()})
    existing_source = object()
    r._heartbeat_watch = {"already-there": (existing_source, "sid-1")}

    calls = []
    monkeypatch.setattr(
        r,
        "_gateway_session_origin_for_id",
        lambda sid: calls.append(sid) or object(),
        raising=False,
    )

    state = HeartbeatState(prompt="check ci", interval_seconds=600, status="active")
    monkeypatch.setattr(
        heartbeat_mod, "list_active_heartbeats", lambda: [("sid-1", state)]
    )

    await _run_one_tick(r, monkeypatch)

    assert calls == []
    assert r._heartbeat_watch == {"already-there": (existing_source, "sid-1")}


@pytest.mark.asyncio
async def test_resumed_watch_lets_a_due_heartbeat_fire_through_the_poller(monkeypatch):
    """End-to-end: after the resume scan re-arms the watch, the real poll
    loop (_start_heartbeat_poller) must actually enqueue the due prompt —
    proving the rebuilt watch entry is functional, not just present.
    """
    fake_source = object()
    r = _runner_with(monkeypatch, origins={"sid-1": fake_source})

    state = HeartbeatState(prompt="check ci", interval_seconds=600, status="active")
    monkeypatch.setattr(
        heartbeat_mod, "list_active_heartbeats", lambda: [("sid-1", state)]
    )

    await _run_one_tick(r, monkeypatch)
    assert r._heartbeat_watch == {"quick_key": (fake_source, "sid-1")}

    # _runner_with stubs _start_heartbeat_poller to a no-op so the resume
    # scan above doesn't spin up the real poller early; restore it now that
    # we actually want to exercise it.
    monkeypatch.setattr(
        r, "_start_heartbeat_poller", GatewayRunner._start_heartbeat_poller.__get__(r)
    )

    class _FakeMgr:
        def __init__(self, session_id):
            assert session_id == "sid-1"

        def has_heartbeat(self):
            return True

        def due_prompt(self):
            return "[Heartbeat] check ci"

    monkeypatch.setattr(heartbeat_mod, "HeartbeatManager", _FakeMgr)
    monkeypatch.setattr(heartbeat_mod, "POLL_SECONDS", 0.0)

    fired = []
    monkeypatch.setattr(r, "_adapter_for_source", lambda _source: object(), raising=False)
    monkeypatch.setattr(
        r,
        "_enqueue_fifo",
        lambda quick_key, evt, _adapter: fired.append((quick_key, evt.text)),
        raising=False,
    )

    r._start_heartbeat_poller()
    poll_task = r._heartbeat_poll_task
    try:
        for _ in range(1000):
            if fired:
                break
            await asyncio.sleep(0)
        else:
            pytest.fail("heartbeat poll loop never fired the due prompt")
    finally:
        poll_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await poll_task

    assert fired == [("quick_key", "[Heartbeat] check ci")]
