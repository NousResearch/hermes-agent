"""A gateway that loses the dispatcher-lock race at startup takes over when the holder exits."""

import asyncio

from gateway.kanban_watchers_common import (
    _acquire_singleton_lock,
    _release_singleton_lock,
)
from gateway.run import GatewayRunner


def test_contended_dispatcher_takes_over_when_holder_releases(monkeypatch):
    import hermes_cli.config as _cfg_mod
    import hermes_cli.kanban_db as _kb
    from hermes_cli import kanban_db_dispatch as _kbd

    monkeypatch.setattr(
        _cfg_mod,
        "load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True, "dispatch_interval_seconds": 1}},
    )
    lock_path = _kb.kanban_home() / "kanban" / ".dispatcher.lock"
    holder, state = _acquire_singleton_lock(lock_path)
    assert state == "held"

    runner = object.__new__(GatewayRunner)
    runner._running = True
    sleeps = 0
    dispatched_after_release = []

    async def _sleep(_delay):
        # Sleep 1 is the startup delay, sleep 2 ends the first contended tick.
        nonlocal holder, sleeps
        sleeps += 1
        if sleeps == 2:
            _release_singleton_lock(holder)
            holder = None

    def _reap():
        dispatched_after_release.append(holder is None)
        runner._running = False
        return []

    monkeypatch.setattr("gateway.run.asyncio.sleep", _sleep)
    monkeypatch.setattr(_kbd, "reap_worker_zombies", _reap)

    try:
        asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=5.0))
    finally:
        if holder is not None:
            _release_singleton_lock(holder)

    assert dispatched_after_release, "the watcher never dispatched after the holder released"
    assert all(dispatched_after_release), "the watcher dispatched while another process held the lock"
