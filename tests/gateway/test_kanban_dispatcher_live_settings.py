"""Live ``kanban.*`` dispatcher settings (#117734).

The embedded dispatcher resolved its caps once at boot, so ``hermes config set
kanban.max_in_progress ...`` was ignored until the gateway restarted. Each tick
now re-resolves the settings when the ``kanban`` section changed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from gateway import kanban_watchers_dispatcher as kwd


def _settings(**overrides):
    base = dict(interval=60.0, max_spawn=None, max_in_progress=4, failure_limit=2,
                stale_timeout_seconds=0, reconcile_orphans=True, default_assignee=None,
                max_in_progress_per_profile=None)
    return kwd._DispatcherSettings(**{**base, **overrides})


def _dispatcher(kanban_cfg):
    kb = SimpleNamespace(DEFAULT_BOARD="default", DEFAULT_FAILURE_LIMIT=2)
    return kwd._KanbanDispatcher(kb, _settings(), kanban_cfg)


def test_next_tick_uses_cap_set_after_boot(monkeypatch):
    """Drive the real watcher loop: a cap written between ticks applies on the next tick."""
    from gateway.run import GatewayRunner
    import hermes_cli.config as cfg_mod
    import hermes_cli.kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    cfg_mod.set_config_value("kanban.max_in_progress", "4")
    cfg_mod.set_config_value("kanban.auto_decompose", "false")
    boot_cfg = cfg_mod.load_config()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_dispatcher_boot = lambda: (cfg_mod.load_config, kb, boot_cfg["kanban"])

    seen = []

    def _tick_once(self):
        seen.append(self.settings.max_in_progress)
        if len(seen) == 1:
            cfg_mod.set_config_value("kanban.max_in_progress", "2")  # `hermes config set`
        if len(seen) == 3:
            runner._running = False
        return []

    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(kwd._KanbanDispatcher, "tick_once", _tick_once)
    monkeypatch.setattr(kwd._KanbanDispatcher, "ready_nonempty", lambda self: False)
    monkeypatch.setattr(kbd, "reap_worker_zombies", list)
    monkeypatch.setattr("gateway.kanban_watchers._kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", _no_sleep)

    asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=10.0))

    assert seen == [4, 2, 2]


def test_unchanged_config_keeps_settings_object():
    dispatcher = _dispatcher({"max_in_progress": 4})
    before = dispatcher.settings
    dispatcher.refresh_settings(lambda: {"kanban": {"max_in_progress": 4}})
    assert dispatcher.settings is before


def test_failed_config_read_keeps_current_settings():
    dispatcher = _dispatcher({"max_in_progress": 4})

    def _boom():
        raise OSError("config.yaml unreadable")

    dispatcher.refresh_settings(_boom)
    assert dispatcher.settings.max_in_progress == 4
