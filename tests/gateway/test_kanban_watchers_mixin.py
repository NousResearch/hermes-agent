"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect

from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_gateway_dispatcher_stuck_warning_names_guard_reason(tmp_path, monkeypatch, caplog):
    """The embedded dispatcher's "stuck" warning names the respawn-guard reason
    holding the ready queue (#111910) instead of a bare zero-spawn count."""
    import asyncio
    import logging

    import gateway.kanban_watchers as kw
    from hermes_cli import kanban_db_dispatch as kbd

    held = kbd.DispatchResult(respawn_guarded=[("t_held", "active_pr")])
    runner = object.__new__(kw.GatewayKanbanWatchersMixin)
    runner._running = True

    class _Kb:
        @staticmethod
        def kanban_home():
            return tmp_path

    monkeypatch.setattr(runner, "_kanban_dispatcher_boot", lambda: (lambda: {}, _Kb(), {}))

    class _Dispatcher:
        def __init__(self, *a, **k):
            pass

        def tick_once(self):
            return [("board", held)]

        def ready_nonempty(self):
            return True

    ticks = {"n": 0}

    async def _direct(fn, *args):
        return fn(*args)

    async def _sleep(_delay):
        ticks["n"] += 1
        if ticks["n"] > kw._HEALTH_WINDOW:
            runner._running = False

    monkeypatch.setattr(kw, "_KanbanDispatcher", _Dispatcher)
    monkeypatch.setattr(kw, "_resolve_dispatcher_settings", lambda cfg, kb: type("S", (), {"interval": 1.0})())
    monkeypatch.setattr(kw, "_to_thread_process_service", _direct)
    monkeypatch.setattr(kw, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(kw, "_resolve_auto_decompose_settings", lambda load_config: (False, 0))
    monkeypatch.setattr(kbd, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(kw.asyncio, "sleep", _sleep)

    with caplog.at_level(logging.WARNING, logger=kw.logger.name):
        asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=5.0))

    stuck = [r.getMessage() for r in caplog.records if "dispatcher stuck" in r.getMessage()]
    assert stuck, [r.getMessage() for r in caplog.records]
    assert "Last tick held back: active_pr=1." in stuck[0]
