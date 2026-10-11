"""Failed hot-serve initialization retires all authority task families before release."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import run_runtime
from gateway.session_authority import LiveSession
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch
from tools.bot_live_delivery import _locked, _write


class _Registry:
    def __init__(self):
        self.removed = []

    def for_home(self, home):
        return None

    def remove(self, home):
        self.removed.append(Path(home).resolve())


@pytest.mark.asyncio
async def test_failed_hot_serve_retires_session_and_bot_recovery_tasks(monkeypatch, tmp_path):
    home = tmp_path.resolve()
    registry = _Registry()
    runner = SimpleNamespace(
        session_authorities=registry,
        session_control_server=object(),
    )

    db = SessionDB(home / "state.db")
    db.create_session("s", source="gui")
    epoch = begin_runtime_epoch(db, instance_id="hot-serve")
    admission = admit_session_input(
        db,
        epoch=epoch,
        principal_id="owner",
        session_id="s",
        request_id="bot:" + "a" * 32,
        payload={"text": "queued bot delivery"},
    )

    live = LiveSession(SimpleNamespace(platform=None, user_id='owner'), 'route')
    authority = SimpleNamespace(
        runner=runner,
        db=db,
        profile_id="default",
        sessions={"s": live},
        waiters={},
        native_waiters=set(),
        hosted_room_service=None,
    )

    with _locked(home) as root:
        _write(
            root / f"{'a' * 32}.json",
            {
                "status": "canonical",
                "admission_id": admission["admission_id"],
                "delivery_id": "a" * 32,
                "profile_home": str(home),
                "session_id": "s",
                "principal_id": "owner",
                "message": "queued bot delivery",
            },
        )

    pump_started = asyncio.Event()
    pump_released = asyncio.Event()
    watcher_seen = asyncio.Event()
    watcher = None

    async def build(*args, **kwargs):
        return authority

    def recover_local(_authority, schedule):
        assert schedule is True

        async def background():
            pump_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                pump_released.set()

        live.task = asyncio.create_task(background())

    async def recover_webhook(_authority):
        return None

    async def fail_hosted(*args, **kwargs):
        nonlocal watcher
        await pump_started.wait()
        tasks = getattr(authority, "_bot_receipt_tasks", set())
        assert len(tasks) == 1
        watcher = next(iter(tasks))
        assert not watcher.done()
        watcher_seen.set()
        raise RuntimeError("late hot-serve failure")

    unbound = []
    monkeypatch.setattr(run_runtime, "_build_profile_authority", build)
    monkeypatch.setattr(
        "gateway.session_local_recovery.recover_local_sessions", recover_local)
    monkeypatch.setattr(
        "gateway.platforms.webhook_ingress.recover_webhook_finalizations",
        recover_webhook,
    )
    monkeypatch.setattr(
        "gateway.session_hosted_service._ensure_hosted_service", fail_hosted)
    monkeypatch.setattr(
        "gateway.session_cron.unbind_owner", lambda value: unbound.append(value))

    try:
        with pytest.raises(RuntimeError, match="late hot-serve failure"):
            await run_runtime.serve_profile_runtime(runner, "cold", home)

        assert watcher_seen.is_set()
        assert registry.removed == [home]
        assert live.task.done() and live.task.cancelled()
        assert pump_released.is_set()
        assert watcher is not None and watcher.done() and watcher.cancelled()
        assert not getattr(authority, "_bot_receipt_tasks", set())
        assert unbound == [authority]
    finally:
        db.close()

@pytest.mark.asyncio
async def test_settle_gateway_runtime_waits_for_bot_receipt_tasks():
    release = asyncio.Event()
    finished = asyncio.Event()

    async def receipt_writer():
        try:
            await release.wait()
        finally:
            finished.set()

    task = asyncio.create_task(receipt_writer())
    authority = SimpleNamespace(sessions={}, _bot_receipt_tasks={task})
    runner = SimpleNamespace(session_authority=authority)
    authority.runner = runner

    settling = asyncio.create_task(run_runtime.settle_gateway_runtime(runner))
    await asyncio.sleep(0)
    assert not settling.done()

    release.set()
    await asyncio.wait_for(settling, 2)

    assert task.done()
    assert finished.is_set()


@pytest.mark.asyncio
async def test_failed_hot_serve_with_a_live_writer_parks_without_aborting_or_re_adopting(monkeypatch, tmp_path):
    """A serve-time failure whose retirement misses its deadline keeps the reservation and the
    retiring authority, parks the profile without adapters, lets the rest of the batch serve, and
    is never handed back as served by the next reconcile."""
    from gateway.run_profile_reconcile import GatewayProfileReconcileMixin
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import SessionAuthorities

    launch = tmp_path / "root"
    homes = {n: (tmp_path / "profiles" / n).resolve() for n in ("beta", "gamma")}
    for home in (launch, *homes.values()):
        home.mkdir(parents=True)
    registry = SessionAuthorities(launch, multiplexed=True)
    writer = asyncio.create_task(asyncio.Event().wait())  # a turn that outlives its Stop
    started, parked = [], {}
    runner = SimpleNamespace(
        session_authorities=registry, session_control_server=None, _running_agents={},
        config=SimpleNamespace(multiplex_profiles=False, _runtime_profile_homes=()),
        session_runtime_descriptor={"served_profiles": []},
        session_ticket_store=SimpleNamespace(profile_ids=frozenset()),
        _served_profile_homes={}, _served_profile_signatures={},
        _live_resource_claims=lambda active: {}, _record_served_profiles=lambda active, homes: None)
    runner.served_profile_names = lambda: sorted(runner._served_profile_homes)
    runner._serve_profile_runtime = GatewayProfileReconcileMixin._serve_profile_runtime.__get__(runner)

    async def start_adapters(name, home, claimed):
        started.append(name)
        return 0

    async def after_added(profile_homes):
        return None

    runner._start_one_profile_adapters, runner._after_profiles_added = start_adapters, after_added

    async def build(runner_, name, home, *, register):
        live = SimpleNamespace(task=writer if name == "beta" else None, route=name,
                               event_stream=SimpleNamespace(execution={"execution_generation": 1}))
        authority = SimpleNamespace(runner=runner_, profile_id=str(home), sessions={"s": live},
                                    pending_stops={}, retiring=False, hosted_room_service=None)
        registry.add(home, authority, name=name)
        return authority

    async def recover_bot(authority):
        if authority.sessions["s"].task is writer:
            raise RuntimeError("serve-time recovery failure")

    async def no_webhooks(authority):
        return None

    monkeypatch.setattr(run_runtime, "_build_profile_authority", build)
    monkeypatch.setattr(run_runtime, "TURN_SETTLE_SECONDS", 0.05)
    monkeypatch.setattr(run_runtime, "_record_parked_profiles", lambda value: parked.update(value))
    monkeypatch.setattr("gateway.session_bot.recover_bot_deliveries", recover_bot)
    monkeypatch.setattr("gateway.session_local_recovery.recover_local_sessions", lambda a, schedule: None)
    monkeypatch.setattr("gateway.platforms.webhook_ingress.recover_webhook_finalizations", no_webhooks)
    monkeypatch.setattr("gateway.session_cron.unbind_owner", lambda authority: None)
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda **kw: list(homes.items()))
    try:
        for attempt in range(2):
            result = await GatewayProfileReconcileMixin._apply_profile_changes(
                runner, {}, list(homes) if attempt == 0 else ["beta"], [], [], reason="control-socket",
                live=dict(homes))
            assert result["parked"] == ["beta"], result
            assert "beta" in parked and "beta" not in started, (attempt, started)
            assert process_ownership.owns(homes["beta"]), "the live writer's home was released"
            assert registry.for_home(homes["beta"]).retiring is True
        assert started == ["gamma"], "one profile's failed serve aborted the batch"
        assert result["served_profiles"] == []  # served_profile_names reads the stubbed record
    finally:
        writer.cancel()
        await asyncio.gather(writer, return_exceptions=True)
        for home in homes.values():
            process_ownership.release(home)
