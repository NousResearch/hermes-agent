"""Board-local health classification for the embedded kanban dispatcher (#46800).

The watcher used to decide "stuck" from two aggregate inputs — "is there
spawnable work on ANY board" and "did ANY board spawn" — so a board whose only
slot was occupied read exactly like a board whose profile venv was gone.

Three contracts, one test each: ``_classify_idle_boards`` judges every board on
its OWN result; the real ``_kanban_dispatcher_watcher`` loop warns naming only
the board that built its own streak (a deferred tick holds the count, a capped
sibling is never named); and ``spawnable_boards`` visits every board without a
short-circuit or a fatal failure.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from gateway import kanban_watchers as kw
from gateway import kanban_watchers_dispatcher as kwd
from gateway.kanban_watchers import _HEALTH_WINDOW
from hermes_cli.kanban_db_dispatch import DispatchResult


def _settings() -> kwd._DispatcherSettings:
    return kwd._DispatcherSettings(
        interval=60.0,
        max_spawn=None,
        max_in_progress=None,
        failure_limit=2,
        stale_timeout_seconds=0,
        reconcile_orphans=True,
        default_assignee=None,
        max_in_progress_per_profile=None,
    )


def test_classify_idle_boards_judges_each_board_on_its_own_result():
    """The board-local invariant asked for on #65581 / #71979.

    A capped board is deferred with its reason; a board that spawned is
    neither; a board with no spawnable work is ignored even without a reason;
    an unexplained zero-spawn board, a board whose tick failed outright
    (``None``) and a spawnable board missing from the results are all stalled
    — and none of the deferred/spawned siblings explains any of them away.
    """
    results = [
        ("full", DispatchResult(capacity_deferred="max_in_progress")),
        ("launched", DispatchResult(spawned=[("t1", "alice", "/tmp/ws")])),
        ("drained", DispatchResult()),
        ("broken", DispatchResult()),
        ("crashed", None),
    ]
    spawnable = {"full", "launched", "broken", "crashed", "ghost"}

    stalled, deferred = kwd._classify_idle_boards(results, spawnable)

    assert stalled == ["broken", "crashed", "ghost"]
    assert deferred == {"full": "capacity:max_in_progress"}


def test_watcher_warns_once_naming_only_the_board_that_stalled(monkeypatch, caplog):
    """Drive the real gateway loop: per-board streak, hold on deferral, warn.

    ``full`` is at its cap on every tick and ``broken`` has a free slot but
    launches nothing. ``broken`` stalls for ``_HEALTH_WINDOW - 1`` ticks, is
    guard-deferred for one tick (the streak must HOLD, not reset), then stalls
    once more. Exactly one warning fires, it names ``broken`` alone, and the
    capped sibling that deferred on every one of those ticks is never named —
    the false alarm in #46800.
    """
    from gateway.run import GatewayRunner

    from hermes_cli import kanban_db as _kb

    stalled_tick = [
        ("full", DispatchResult(capacity_deferred="max_spawn")),
        ("broken", DispatchResult()),
    ]
    deferred_tick = [
        ("full", DispatchResult(capacity_deferred="max_spawn")),
        ("broken", DispatchResult(respawn_guarded=[("t1", "active_pr")])),
    ]
    script = [stalled_tick] * (_HEALTH_WINDOW - 1) + [deferred_tick] + [stalled_tick]

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_dispatcher_boot = lambda: (lambda: {}, _kb, {"dispatch_interval_seconds": 1})

    def _tick_once(self):
        results = script.pop(0)
        if not script:
            runner._running = False
        return results

    async def _run_inline(fn, *args):
        return fn(*args)

    async def _no_sleep(*_args, **_kwargs):
        return None

    monkeypatch.setattr(kwd._KanbanDispatcher, "tick_once", _tick_once)
    monkeypatch.setattr(kwd._KanbanDispatcher, "spawnable_boards", lambda self: {"full", "broken"})
    monkeypatch.setattr(kw, "_to_thread_process_service", _run_inline)
    monkeypatch.setattr(kw, "_resolve_auto_decompose_settings", lambda _load: (False, 3))
    monkeypatch.setattr(kw, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(kw.asyncio, "sleep", _no_sleep)

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=5.0))

    assert script == [], "the loop stopped before the scripted ticks ran out"
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "dispatcher stuck" in r.getMessage()]
    assert len(warnings) == 1
    assert "boards=['broken']" in warnings[0]
    assert "unexplained ticks" in warnings[0]
    # The deferred ticks were logged with their reason, never as a stall.
    deferred_lines = [r.getMessage() for r in caplog.records if "deferred (" in r.getMessage()]
    assert "kanban dispatcher [full]: deferred (capacity:max_spawn)" in deferred_lines
    assert "kanban dispatcher [broken]: deferred (respawn_guarded)" in deferred_lines


def test_spawnable_boards_visits_every_board_and_skips_one_that_fails_to_open(monkeypatch):
    """The SET of boards with work: no short-circuit, and a corrupt board is not fatal."""
    connected: list[str] = []

    def _connect(board=None):
        connected.append(board)
        if board == "bad":
            raise RuntimeError("file is not a database")
        return SimpleNamespace(slug=board, close=lambda: None)

    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["bad", "busy", "idle"])
    monkeypatch.setattr(kwd, "_kbc", lambda: SimpleNamespace(connect=_connect))
    monkeypatch.setattr(kwd, "_kbd", lambda: SimpleNamespace(
        review_dispatch_enabled=lambda: False,
        has_spawnable_ready=lambda conn: conn.slug == "busy",
        has_spawnable_review=lambda conn: False,
    ))
    dispatcher = kwd._KanbanDispatcher(kb=object(), settings=_settings())

    assert dispatcher.spawnable_boards() == {"busy"}
    # Every board is visited once, even after the first hit.
    assert connected == ["bad", "busy", "idle"]
