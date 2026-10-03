"""Exercise the actual watcher loop, not only its classification helper."""
import asyncio
import logging
from types import SimpleNamespace

import pytest

import gateway.kanban_watchers as watchers
from hermes_cli import kanban_db_dispatch as kbd


@pytest.mark.parametrize("reason,expected", [
    ("rate_limit_cooldown", False), ("blocker_auth", True),
])
def test_watcher_warning(monkeypatch, caplog, reason, expected):
    async def immediate(fn, *args):
        return fn(*args)

    async def no_sleep(*args):
        pass

    monkeypatch.setattr(watchers, "_to_thread_process_service", immediate)
    monkeypatch.setattr(watchers.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(watchers, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(kbd, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(watchers, "_resolve_dispatcher_settings", lambda *args: SimpleNamespace(interval=1))
    monkeypatch.setattr(watchers, "_resolve_auto_decompose_settings", lambda *args: (False, 0))
    runner = watchers.GatewayKanbanWatchersMixin()
    runner._running = True
    ticks = []

    def tick():
        ticks.append(1)
        if len(ticks) == watchers._HEALTH_WINDOW:
            runner._running = False
        return [("b", kbd.DispatchResult(respawn_guarded=[("t", reason)]))]

    monkeypatch.setattr(runner, "_kanban_dispatcher_boot", lambda: (lambda: {}, None, {}))
    monkeypatch.setattr(watchers, "_KanbanDispatcher", lambda *args: SimpleNamespace(
        tick_once=tick, ready_nonempty=lambda: True,
    ))
    with caplog.at_level(logging.WARNING):
        asyncio.run(runner._kanban_dispatcher_watcher())
    assert len(ticks) == watchers._HEALTH_WINDOW
    assert ("kanban dispatcher stuck" in caplog.text) is expected
