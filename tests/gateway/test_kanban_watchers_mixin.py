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


def test_gateway_dispatcher_stuck_warning_names_guard_reason(monkeypatch, caplog):
    """The embedded dispatcher's "stuck" warning names the respawn-guard reason
    holding the ready queue (#111910) instead of a bare zero-spawn count."""
    import asyncio
    import logging

    import gateway.kanban_watchers as kw
    from hermes_cli import kanban_db_dispatch as kbd

    held = kbd.DispatchResult(respawn_guarded=[("t_held", "active_pr")])
    runner = object.__new__(kw.GatewayKanbanWatchersMixin)
    runner._running = True
    monkeypatch.setattr(runner, "_kanban_dispatcher_boot", lambda: (lambda: {}, object(), {}))

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


def test_dispatcher_settings_refresh_applies_config_edit_without_restart(
    monkeypatch, caplog
):
    """A `kanban.*` dispatch-settings edit reaches the live dispatcher on the
    next tick (#117734): settings are re-resolved only when the config dict
    changed, and the dispatcher swaps to them without a gateway restart."""
    import asyncio
    import logging

    import gateway.kanban_watchers as kw
    from gateway.kanban_watchers_dispatcher import _DispatcherSettings
    from hermes_cli import kanban_db_dispatch as kbd

    def _settings(max_spawn):
        return _DispatcherSettings(
            interval=1.0,
            max_spawn=max_spawn,
            max_in_progress=None,
            failure_limit=3,
            stale_timeout_seconds=0,
            reconcile_orphans=True,
            default_assignee=None,
            max_in_progress_per_profile=None,
        )

    resolve_calls = []

    def _fake_resolve(cfg, _kb):
        resolve_calls.append(dict(cfg))
        return _settings(cfg.get("max_spawn"))

    loader_calls = {"n": 0}

    def _loader():
        # Boot-shaped config for the first re-read, then the operator's edit.
        kanban = {"max_spawn": None} if loader_calls["n"] == 0 else {"max_spawn": 2}
        loader_calls["n"] += 1
        return {"kanban": kanban}

    created = []

    class _Dispatcher:
        def __init__(self, kb, settings):
            self.settings = settings
            self.seen = []
            created.append(self)

        def tick_once(self):
            self.seen.append(self.settings.max_spawn)
            return []

        def ready_nonempty(self):
            return False

    runner = object.__new__(kw.GatewayKanbanWatchersMixin)
    runner._running = True
    monkeypatch.setattr(
        runner,
        "_kanban_dispatcher_boot",
        lambda: (_loader, object(), {"max_spawn": None}),
    )

    ticks = {"n": 0}

    async def _direct(fn, *args):
        return fn(*args)

    async def _sleep(_delay):
        ticks["n"] += 1
        if ticks["n"] >= 4:
            runner._running = False

    monkeypatch.setattr(kw, "_KanbanDispatcher", _Dispatcher)
    monkeypatch.setattr(kw, "_resolve_dispatcher_settings", _fake_resolve)
    monkeypatch.setattr(kw, "_to_thread_process_service", _direct)
    monkeypatch.setattr(kw, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(
        kw, "_resolve_auto_decompose_settings", lambda load_config: (False, 0)
    )
    monkeypatch.setattr(kbd, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(kw.asyncio, "sleep", _sleep)

    with caplog.at_level(logging.INFO, logger=kw.logger.name):
        asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=5.0))

    # Tick 1 still runs the boot snapshot (config not yet edited); the edit
    # applies on the very next tick and sticks without a gateway restart.
    assert created[0].seen == [None, 2, 2]
    # Boot resolution + exactly one re-resolution: unchanged ticks skip it.
    assert len(resolve_calls) == 2
    applied = [
        r.getMessage()
        for r in caplog.records
        if "applying changed kanban.*" in r.getMessage()
    ]
    assert applied and "max_spawn None->2" in applied[0]
