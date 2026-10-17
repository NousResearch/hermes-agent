"""The embedded kanban dispatcher re-reads its concurrency caps every tick.

`kanban.max_in_progress` is the lever a user pulls when the host is under memory
pressure; resolving it once at boot meant lowering it did nothing until a gateway
restart. Same class as auto_decompose (#49638).
"""

from __future__ import annotations


def test_gateway_dispatcher_rereads_max_in_progress_every_tick(monkeypatch):
    """Lowering `kanban.max_in_progress` while the host swaps must take effect on
    the next tick, like `auto_decompose` (#49638) — not only after a restart."""
    import asyncio

    import gateway.kanban_watchers as kw
    from hermes_cli import kanban_db_dispatch as kbd

    cfg = {"kanban": {"max_in_progress": 7}}
    seen = []
    runner = object.__new__(kw.GatewayKanbanWatchersMixin)
    runner._running = True
    monkeypatch.setattr(runner, "_kanban_dispatcher_boot", lambda: (lambda: cfg, object(), cfg["kanban"]))

    class _Dispatcher:
        def __init__(self, kb, settings):
            self.settings = settings

        def tick_once(self):
            seen.append(self.settings.max_in_progress)
            # The user lowers the cap right after the first tick.
            if len(seen) == 1:
                cfg["kanban"] = {"max_in_progress": 4}
            if len(seen) >= 3:
                runner._running = False
            return []

        def ready_nonempty(self):
            return False

    async def _direct(fn, *args):
        return fn(*args)

    async def _sleep(_delay):
        return None

    def _settings(kcfg, kb):
        return type("S", (), {"interval": 1.0, "max_in_progress": kcfg.get("max_in_progress")})()

    monkeypatch.setattr(kw, "_KanbanDispatcher", _Dispatcher)
    monkeypatch.setattr(kw, "_resolve_dispatcher_settings", _settings)
    monkeypatch.setattr(kw, "_to_thread_process_service", _direct)
    monkeypatch.setattr(kw, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(kw, "_resolve_auto_decompose_settings", lambda load_config: (False, 0))
    monkeypatch.setattr(kbd, "reap_worker_zombies", list)
    monkeypatch.setattr(kw.asyncio, "sleep", _sleep)

    asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=5.0))

    assert seen[0] == 7
    assert seen[1:] and all(v == 4 for v in seen[1:]), seen
