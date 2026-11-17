"""Who may dispatch when the dispatcher lock is contended or unavailable."""

import asyncio
import logging

import pytest

import gateway.kanban_watchers as _kw
from gateway.kanban_watchers_common import (
    _acquire_singleton_lock,
    _release_singleton_lock,
)
from gateway.run import GatewayRunner


@pytest.fixture
def watcher(monkeypatch):
    """Run the dispatcher watcher until the first reap; record what each reap saw."""
    import hermes_cli.config as _cfg_mod
    import hermes_cli.kanban_db as _kb
    from hermes_cli import kanban_db_dispatch as _kbd

    monkeypatch.setattr(
        _cfg_mod,
        "load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True, "dispatch_interval_seconds": 1}},
    )

    runner = object.__new__(GatewayRunner)
    runner._running = True
    reaps = []
    sleeps = 0

    async def _no_sleep(_delay):
        # The sleep never yields, so wait_for cannot time out: a watcher that
        # never dispatches must be stopped here instead.
        nonlocal sleeps
        sleeps += 1
        if sleeps > 50:
            runner._running = False

    monkeypatch.setattr("gateway.run.asyncio.sleep", _no_sleep)

    def _reap():
        reaps.append(runner._owns_kanban_dispatcher_lock())
        runner._running = False
        return []

    monkeypatch.setattr(_kbd, "reap_worker_zombies", _reap)

    def run():
        asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=5.0))
        return reaps

    run.lock_path = _kb.kanban_home() / "kanban" / ".dispatcher.lock"
    return run


def test_contended_dispatcher_takes_over_when_holder_releases(watcher, monkeypatch):
    holder, state = _acquire_singleton_lock(watcher.lock_path)
    assert state == "held"
    attempts = []

    def _acquire(path):
        nonlocal holder
        result = _acquire_singleton_lock(path)
        attempts.append(result[1])
        # Call 1 is startup. The holder exits only after a contended tick re-try.
        if len(attempts) >= 2 and result[1] == "contended":
            _release_singleton_lock(holder)
            holder = None
        return result

    monkeypatch.setattr(_kw, "_acquire_singleton_lock", _acquire)

    try:
        reaps = watcher()
    finally:
        if holder is not None:
            _release_singleton_lock(holder)

    assert reaps, "the watcher never dispatched after the holder released"
    assert attempts[:2] == ["contended", "contended"]
    assert all(reaps), "the watcher dispatched without holding the lock"


def test_lock_unavailable_at_startup_keeps_config_only_dispatch(watcher, monkeypatch):
    monkeypatch.setattr(_kw, "_acquire_singleton_lock", lambda _path: (None, "unavailable"))

    reaps = watcher()

    assert reaps, "the unlocked fallback stopped dispatching"
    assert set(reaps) == {False}


def test_lock_unavailable_after_startup_pauses_dispatch_and_warns_once(watcher, monkeypatch, caplog):
    states = iter(["contended", "unavailable", "unavailable", "held"])

    def _acquire(path):
        state = next(states)
        return _acquire_singleton_lock(path) if state == "held" else (None, state)

    monkeypatch.setattr(_kw, "_acquire_singleton_lock", _acquire)

    with caplog.at_level(logging.WARNING, logger=_kw.logger.name):
        reaps = watcher()

    assert reaps, "the watcher never dispatched once the lock was taken"
    assert all(reaps), "the watcher dispatched while the lock was unavailable"
    unavailable = [r for r in caplog.records if "dispatcher lock unavailable" in r.getMessage()]
    assert len(unavailable) == 1
