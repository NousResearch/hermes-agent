"""Same-tick auto-decompose must not block ready-task spawn (#106985).

The embedded dispatcher used to ``await auto_decompose_tick`` to completion
before ``tick_once``. A stuck triage decompose then delayed already-ready
work on that tick. Spawn must start without waiting for decompose to finish;
a decompose exception must not cancel the spawn path.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace

from gateway.kanban_watchers import GatewayKanbanWatchersMixin


_SPAWN_WAIT_S = 0.8


class _RecordingDispatcher:
    """In-thread dispatcher stand-in with observable decompose/spawn timing."""

    def __init__(self, *, decompose_error: BaseException | None = None):
        self.decompose_calls: list[int] = []
        self.spawn_calls = 0
        self.ready_calls = 0
        self.spawn_started = threading.Event()
        self.decompose_error = decompose_error
        self.spawn_blocked_by_decompose = False

    def auto_decompose_tick(self, per_tick: int) -> int:
        self.decompose_calls.append(per_tick)
        if self.decompose_error is not None:
            raise self.decompose_error
        # Serial await: tick_once has not started, so this times out.
        # Concurrent gather: tick_once sets the event and we return promptly.
        if not self.spawn_started.wait(timeout=_SPAWN_WAIT_S):
            self.spawn_blocked_by_decompose = True
        return 0

    def tick_once(self):
        self.spawn_calls += 1
        self.spawn_started.set()
        return []

    def ready_nonempty(self) -> bool:
        self.ready_calls += 1
        return False


def _run_one_dispatcher_tick(
    monkeypatch,
    dispatcher: _RecordingDispatcher,
    *,
    auto_decompose: bool = True,
    per_tick: int = 3,
    dispatch_allowed: bool = True,
) -> None:
    runner = GatewayKanbanWatchersMixin()
    runner._running = True
    runner._kanban_dispatcher_lock_handle = None

    def load_config():
        return {
            "kanban": {
                "auto_decompose": auto_decompose,
                "auto_decompose_per_tick": per_tick,
            }
        }

    monkeypatch.setattr(
        runner, "_kanban_dispatcher_boot", lambda: (load_config, object(), {})
    )
    monkeypatch.setattr(
        "gateway.kanban_watchers._resolve_dispatcher_settings",
        lambda _cfg, _kb: SimpleNamespace(interval=1.0),
    )
    monkeypatch.setattr(
        "gateway.kanban_watchers._KanbanDispatcher",
        lambda _kb, _settings: dispatcher,
    )
    monkeypatch.setattr(
        "gateway.kanban_watchers._kanban_dispatch_allowed",
        lambda: dispatch_allowed,
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_db_dispatch.reap_worker_zombies",
        lambda: [],
    )

    async def _no_sleep(_delay=0):
        return None

    async def _stop_after_tick(_interval):
        runner._running = False

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", _no_sleep)
    monkeypatch.setattr(runner, "_sleep_between_ticks", _stop_after_tick)

    asyncio.run(runner._kanban_dispatcher_watcher())


def test_tick_once_starts_before_slow_auto_decompose_finishes(monkeypatch):
    dispatcher = _RecordingDispatcher()
    _run_one_dispatcher_tick(monkeypatch, dispatcher, auto_decompose=True, per_tick=4)

    assert dispatcher.decompose_calls == [4], "auto_decompose_per_tick cap must be forwarded"
    assert dispatcher.spawn_calls == 1
    assert dispatcher.spawn_started.is_set()
    assert dispatcher.spawn_blocked_by_decompose is False, (
        "tick_once must start before auto_decompose_tick finishes; "
        "a slow decompose must not serialize the spawn side of the same tick"
    )


def test_auto_decompose_disabled_skips_decompose_and_still_spawns(monkeypatch):
    dispatcher = _RecordingDispatcher()
    _run_one_dispatcher_tick(monkeypatch, dispatcher, auto_decompose=False)

    assert dispatcher.decompose_calls == []
    assert dispatcher.spawn_calls == 1
    assert dispatcher.ready_calls == 1


def test_decompose_exception_does_not_skip_spawn(monkeypatch, caplog):
    dispatcher = _RecordingDispatcher(decompose_error=RuntimeError("simulated decompose crash"))
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        _run_one_dispatcher_tick(monkeypatch, dispatcher, auto_decompose=True)

    assert dispatcher.decompose_calls == [3]
    assert dispatcher.spawn_calls == 1
    assert dispatcher.ready_calls == 1
    assert any("auto-decompose" in rec.getMessage() for rec in caplog.records)


def test_estop_skips_both_decompose_and_spawn(monkeypatch):
    dispatcher = _RecordingDispatcher()
    _run_one_dispatcher_tick(monkeypatch, dispatcher, dispatch_allowed=False)

    assert dispatcher.decompose_calls == []
    assert dispatcher.spawn_calls == 0
    assert dispatcher.ready_calls == 0
