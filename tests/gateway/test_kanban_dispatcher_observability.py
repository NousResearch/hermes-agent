"""Dispatcher tick observability contracts (regression for #140028)."""

from __future__ import annotations

import asyncio
import logging
import re


def test_dispatcher_logs_completed_idle_tick(monkeypatch, caplog):
    """An empty dispatch still leaves a timed heartbeat in the gateway log."""
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin
    import gateway.kanban_watchers as watchers

    class _Runner(GatewayKanbanWatchersMixin):
        _running = True

        def _kanban_dispatcher_boot(self):
            return dict, object(), {}

        def _release_kanban_dispatcher_lock(self):
            return None

        async def _sleep_between_ticks(self, _interval):
            self._running = False

    class _Settings:
        interval = 1.0

    class _Dispatcher:
        def __init__(self, _kb, _settings):
            pass

        def tick_once(self):
            return []

        def ready_nonempty(self):
            return False

    async def _to_thread(func, *args):
        return func(*args)

    original_sleep = asyncio.sleep
    monkeypatch.setattr(watchers, "_resolve_dispatcher_settings", lambda *_args: _Settings())
    monkeypatch.setattr(watchers, "_KanbanDispatcher", _Dispatcher)
    monkeypatch.setattr(watchers, "_to_thread_process_service", _to_thread)
    monkeypatch.setattr(watchers, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(watchers, "_resolve_auto_decompose_settings", lambda _load: (False, 3))
    monkeypatch.setattr(watchers.asyncio, "sleep", lambda _delay: original_sleep(0))

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        asyncio.run(_Runner()._kanban_dispatcher_watcher())

    messages = [record.getMessage() for record in caplog.records]
    assert any(re.fullmatch(
        r"kanban dispatcher tick: boards=0 spawned=0 ready=0 duration=\d+\.\d{3}s", message,
    ) for message in messages)
