"""ESTOP pauses must explain queued Kanban work and announce recovery."""

import asyncio
import logging
from types import SimpleNamespace

from gateway.kanban_watchers import GatewayKanbanWatchersMixin


def test_dispatcher_announces_estop_pause_and_resume(monkeypatch, caplog):
    """Ready work stays queued during ESTOP, then gets an explicit resume signal."""
    runner = object.__new__(GatewayKanbanWatchersMixin)
    runner._running = True
    settings = SimpleNamespace(interval=1)
    runner._kanban_dispatcher_boot = lambda: (lambda: {}, object(), {})

    class Dispatcher:
        def __init__(self, _kb, _settings):
            pass

        def auto_decompose_tick(self, _per_tick):
            return 0

        def tick_once(self):
            return []

        def ready_nonempty(self):
            runner._running = False
            return True

    paused = iter((False, True))

    async def to_thread(func, *args):
        return func(*args)

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("gateway.kanban_watchers._KanbanDispatcher", Dispatcher)
    monkeypatch.setattr("gateway.kanban_watchers._kanban_dispatch_allowed", lambda: next(paused))
    monkeypatch.setattr("gateway.kanban_watchers._resolve_dispatcher_settings", lambda _cfg, _kb: settings)
    monkeypatch.setattr("gateway.kanban_watchers._resolve_auto_decompose_settings", lambda _load: (False, 1))
    monkeypatch.setattr("gateway.kanban_watchers._to_thread_process_service", to_thread)
    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", no_sleep)
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch.reap_worker_zombies", lambda: [])

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        asyncio.run(runner._kanban_dispatcher_watcher())

    messages = [record.getMessage() for record in caplog.records]
    assert any("ESTOP is active" in message and "ready tasks remain queued" in message for message in messages)
    assert any("ESTOP cleared" in message and "resuming dispatch" in message for message in messages)
