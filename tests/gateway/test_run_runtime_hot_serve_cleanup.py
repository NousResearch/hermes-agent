"""Failed hot-serve initialization must retire recovery work before ownership can be released."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import run_runtime


class _Registry:
    def __init__(self):
        self.removed = []

    def for_home(self, home):
        return None

    def remove(self, home):
        self.removed.append(Path(home).resolve())


@pytest.mark.asyncio
async def test_failed_hot_serve_cancels_scheduled_authority_tasks(monkeypatch, tmp_path):
    home = tmp_path.resolve()
    registry = _Registry()
    live = SimpleNamespace(task=None)
    authority = SimpleNamespace(
        sessions={"s": live},
        hosted_room_service=None,
    )
    runner = SimpleNamespace(
        session_authorities=registry,
        session_control_server=object(),
    )
    started = asyncio.Event()
    released = asyncio.Event()

    async def build(*args, **kwargs):
        return authority

    async def recover_bot(_authority):
        return None

    def recover_local(_authority, schedule):
        assert schedule is True

        async def background():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()

        live.task = asyncio.create_task(background())

    async def recover_webhook(_authority):
        return None

    async def fail_hosted(*args, **kwargs):
        await started.wait()
        raise RuntimeError("late hot-serve failure")

    unbound = []
    monkeypatch.setattr(run_runtime, "_build_profile_authority", build)
    monkeypatch.setattr("gateway.session_bot.recover_bot_deliveries", recover_bot)
    monkeypatch.setattr("gateway.session_local_recovery.recover_local_sessions", recover_local)
    monkeypatch.setattr("gateway.platforms.webhook_ingress.recover_webhook_finalizations", recover_webhook)
    monkeypatch.setattr("gateway.session_hosted_service._ensure_hosted_service", fail_hosted)
    monkeypatch.setattr("gateway.session_cron.unbind_owner", lambda value: unbound.append(value))

    with pytest.raises(RuntimeError, match="late hot-serve failure"):
        await run_runtime.serve_profile_runtime(runner, "cold", home)

    assert registry.removed == [home]
    assert live.task.done() and live.task.cancelled()
    assert released.is_set()
    assert unbound == [authority]
