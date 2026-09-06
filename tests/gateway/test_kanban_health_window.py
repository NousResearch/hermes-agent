"""Regression tests for embedded kanban dispatcher health telemetry."""

from __future__ import annotations

import asyncio
import logging

from gateway.run import GatewayRunner
from gateway.kanban_watchers import GatewayKanbanWatchersMixin


def test_dispatcher_health_telemetry_does_not_raise_nameerror(monkeypatch, caplog):
    """A completed health branch must evaluate its window constant safely."""
    import gateway.kanban_watchers as watchers

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_dispatcher_lock_handle = None

    monkeypatch.setattr(
        GatewayKanbanWatchersMixin,
        "_kanban_dispatcher_boot",
        lambda self: (lambda: {}, object(), {}),
    )
    monkeypatch.setattr(
        watchers,
        "_resolve_dispatcher_settings",
        lambda _cfg, _kb: type("Settings", (), {"interval": 1.0})(),
    )
    monkeypatch.setattr(watchers, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(
        watchers,
        "_resolve_auto_decompose_settings",
        lambda _load_config: (False, 0),
    )

    calls = []

    async def _to_thread(fn, *args, **kwargs):
        calls.append(fn.__name__)
        if fn.__name__ == "_ready_nonempty":
            runner._running = False
        return [] if fn.__name__ != "_ready_nonempty" else False

    async def _sleep(_delay):
        return None

    monkeypatch.setattr(watchers, "_to_thread_process_service", _to_thread)
    monkeypatch.setattr(watchers.asyncio, "sleep", _sleep)

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=3.0))

    assert "_ready_nonempty" in calls
    assert not any("HEALTH_WINDOW" in record.getMessage() for record in caplog.records)
    assert not any("unexpected watcher error" in record.getMessage() for record in caplog.records)
    assert not any(record.exc_info for record in caplog.records)
