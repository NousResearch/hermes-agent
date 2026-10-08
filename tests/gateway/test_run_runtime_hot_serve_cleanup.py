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
