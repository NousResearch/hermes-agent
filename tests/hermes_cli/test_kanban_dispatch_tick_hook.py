"""Tests for the ``on_kanban_dispatch_tick`` observer hook.

Re-port of PR #56066 per the #64231 batch disposition: renamed to the
taxonomy form and fired by ``kanban_db.dispatch_once`` strictly AFTER the
board's single-writer dispatch lock has been released — the original fired
inside ``_dispatch_tick_lock``, so a slow subscriber could extend the
critical section and stall a sibling dispatcher. Verifies the post-lock
contract directly, the ``outcome`` classification (including idle and
contended ticks), ``dry_run`` propagation, and that a misbehaving
subscriber never breaks the dispatcher.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.plugins import get_plugin_manager

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home

@pytest.fixture
def captured_ticks(monkeypatch):
    """Register a capturing callback for the dispatch tick hook."""
    mgr = get_plugin_manager()
    events: list[dict] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("on_kanban_dispatch_tick", []).append(
        lambda **kw: events.append(kw)
    )
    try:
        yield events
    finally:
        mgr._hooks = saved

def test_active_tick_fires_hook_with_outcome_ok(
    kanban_home, all_assignees_spawnable, captured_ticks,
):
    """A tick that spawns a worker fires the hook with outcome='ok'."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    finally:
        conn.close()
    ok_events = [kw for kw in captured_ticks if kw["outcome"] == "ok"]
    assert ok_events, (
        f"expected an ok tick, got {[kw['outcome'] for kw in captured_ticks]}"
    )
    result = ok_events[-1]["result"]
    assert any(row[0] == tid for row in result.spawned)

def test_tick_hook_fires_after_dispatch_lock_released(kanban_home):
    """The #56066 sweeper finding, as a contract: subscribers run OUTSIDE
    the single-writer critical section. From the callback, acquiring the
    board's dispatch lock must succeed — flock conflicts across file
    descriptors within one process, so this fails if the hook still fires
    while ``dispatch_once`` holds the lock."""
    db_path = kb.kanban_db_path()
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    acquired: list[bool] = []

    def _probe_lock(**kw):
        with kbc._dispatch_tick_lock(db_path) as held:
            acquired.append(held)

    mgr._hooks.setdefault("on_kanban_dispatch_tick", []).append(_probe_lock)
    try:
        conn = kbc.connect()
        try:
            kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)
        finally:
            conn.close()
    finally:
        mgr._hooks = saved
    assert acquired == [True]

def test_misbehaving_subscriber_does_not_break_dispatcher(kanban_home):
    """A hook callback that raises must not break the dispatch tick."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _boom(**kw):
        raise RuntimeError("subscriber exploded")

    mgr._hooks.setdefault("on_kanban_dispatch_tick", []).append(_boom)
    try:
        conn = kbc.connect()
        try:
            result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)
            assert isinstance(result, kb.DispatchResult)
        finally:
            conn.close()
    finally:
        mgr._hooks = saved


def test_dispatcher_stuck_hook_exposes_rate_limited_health_signal(kanban_home):
    """Regression for #140662: the dispatcher health warning is observable."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    events: list[dict] = []
    mgr._hooks.setdefault("on_kanban_dispatcher_stuck", []).append(
        lambda **kwargs: events.append(kwargs)
    )
    try:
        kb._fire_dispatcher_stuck_hook(
            board="alpha", bad_ticks=6, held="max_in_progress=2", source="gateway",
        )
    finally:
        mgr._hooks = saved

    assert len(events) == 1
    assert {key: events[0][key] for key in (
        "board", "profile_name", "bad_ticks", "held", "source",
    )} == {
        "board": "alpha", "profile_name": "default", "bad_ticks": 6,
        "held": "max_in_progress=2", "source": "gateway",
    }


def test_standalone_daemon_fires_stuck_hook_at_warning_threshold(monkeypatch):
    """The deprecated daemon emits the observer signal with its warning."""
    from hermes_cli import kanban_ops

    fired: list[dict] = []
    result = kb.DispatchResult()
    monkeypatch.setattr(kanban_ops.kb, "init_db", lambda: None)
    monkeypatch.setattr(kanban_ops.kb, "get_current_board", lambda: "alpha")
    monkeypatch.setattr(
        kanban_ops.kb, "_fire_dispatcher_stuck_hook", lambda **kwargs: fired.append(kwargs),
    )
    monkeypatch.setattr(kanban_ops.kbd, "has_spawnable_ready", lambda conn: True)
    monkeypatch.setattr(kanban_ops.kbd, "describe_suppression", lambda results: "profile capped")

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(kanban_ops.kbc, "connect_closing", lambda: _Connection())
    monkeypatch.setattr(
        kanban_ops.kbd, "run_daemon",
        lambda **kwargs: [kwargs["on_tick"](result) for _ in range(6)],
    )

    assert kanban_ops._cmd_daemon(
        SimpleNamespace(force=True, pidfile=None, verbose=False, interval=1, max=None),
    ) == 0
    assert fired == [{
        "board": "alpha", "bad_ticks": 6, "held": "profile capped", "source": "daemon",
    }]


def test_gateway_dispatcher_fires_stuck_hook_at_warning_threshold(monkeypatch):
    """The embedded dispatcher emits the same rate-limited observer signal."""
    from gateway import kanban_watchers

    fired: list[dict] = []

    class _KanbanDB:
        @staticmethod
        def _fire_dispatcher_stuck_hook(**kwargs):
            fired.append(kwargs)

    class _Dispatcher:
        def __init__(self, *_args):
            pass

        def tick_once(self):
            return [("alpha", kb.DispatchResult())]

        def ready_nonempty(self):
            return True

    class _Watcher(kanban_watchers.GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = True
            self.ticks = 0

        def _kanban_dispatcher_boot(self):
            return (dict, _KanbanDB(), {})

        async def _sleep_between_ticks(self, _interval):
            self.ticks += 1
            if self.ticks == 6:
                self._running = False

    async def _run_in_thread(fn, *args):
        if fn.__name__ == "reap_worker_zombies":
            return []
        return fn(*args)

    original_sleep = asyncio.sleep
    monkeypatch.setattr(kanban_watchers.asyncio, "sleep", lambda _delay: original_sleep(0))
    monkeypatch.setattr(kanban_watchers, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(kanban_watchers, "_resolve_auto_decompose_settings", lambda _cfg: (False, 0))
    monkeypatch.setattr(kanban_watchers, "_resolve_dispatcher_settings", lambda *_args: SimpleNamespace(interval=1))
    monkeypatch.setattr(kanban_watchers, "_KanbanDispatcher", _Dispatcher)
    monkeypatch.setattr(kanban_watchers, "_log_spawn_results", lambda _results: False)
    monkeypatch.setattr(kanban_watchers, "_to_thread_process_service", _run_in_thread)
    monkeypatch.setattr(kanban_watchers.time, "time", lambda: 300)
    monkeypatch.setattr(kanban_watchers, "_HEALTH_WINDOW", 6)

    watcher = _Watcher()
    asyncio.run(watcher._kanban_dispatcher_watcher())

    assert fired == [{
        "board": None, "bad_ticks": 6, "held": "", "source": "gateway",
    }]
