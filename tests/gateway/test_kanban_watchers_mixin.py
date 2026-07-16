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


def test_gateway_runner_inherits_mixin():
    # Import here so a heavy gateway import only happens if the first test passed.
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewayKanbanWatchersMixin)
    # Each kanban method resolves to the mixin's implementation via the MRO.
    for m in KANBAN_METHODS:
        owner = next(c for c in GatewayRunner.__mro__ if m in c.__dict__)
        assert owner is GatewayKanbanWatchersMixin, (
            f"{m} resolved to {owner.__name__}, expected the mixin"
        )


def test_watcher_loops_are_coroutines():
    # The two long-running watchers are async loops.
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


def test_singleton_dispatcher_lock_is_exclusive(tmp_path):
    """Only one holder of the dispatcher lock at a time — the backstop that
    stops concurrent dispatchers double reclaiming and corrupting shared
    kanban SQLite index pages under wal_autocheckpoint=0."""
    import os

    from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock

    lock = tmp_path / "kanban" / ".dispatcher.lock"

    h1, st1 = _acquire_singleton_lock(lock)
    assert st1 == "held" and h1 is not None

    # A second acquire while the first is held must be refused, not granted.
    h2, st2 = _acquire_singleton_lock(lock)
    assert st2 == "contended" and h2 is None

    # Releasing the first lets a fresh acquire succeed (lock is reusable).
    _release_singleton_lock(h1)
    h3, st3 = _acquire_singleton_lock(lock)
    assert st3 == "held" and h3 is not None
    _release_singleton_lock(h3)


def test_dispatcher_caps_live_workers_across_active_boards(monkeypatch, tmp_path):
    """The gateway's live width is shared by every non-archived board."""
    import asyncio
    from types import SimpleNamespace

    import hermes_cli.config as config
    from hermes_cli import kanban_db as kb

    runner = GatewayKanbanWatchersMixin()
    runner._running = True
    running = {"first": 12, "second": 3}
    dispatched = []

    class Connection:
        def __init__(self, board):
            self.board = board

        def execute(self, query):
            assert "status = 'running'" in query
            return SimpleNamespace(fetchone=lambda: (running[self.board],))

        def close(self):
            pass

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "auto_decompose": False,
                "max_in_progress": 20,
                "max_spawn": 20,
            }
        },
    )
    monkeypatch.setattr(
        kb, "list_boards", lambda include_archived=False: [
            {"slug": "first"}, {"slug": "second"},
        ],
    )
    monkeypatch.setattr(kb, "connect", lambda *, board: Connection(board))
    monkeypatch.setattr(kb, "kanban_home", lambda: tmp_path)
    monkeypatch.setattr(kb, "kanban_db_path", lambda board: tmp_path / board)
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(kb, "has_spawnable_review", lambda conn: False)
    monkeypatch.setattr(
        "gateway.kanban_watchers._acquire_singleton_lock",
        lambda path: (object(), "held"),
    )

    def dispatch_once(conn, **kwargs):
        cap = min(kwargs["max_in_progress"], kwargs["max_spawn"])
        headroom = cap - running[conn.board]
        # The first board leaves three slots unused. The second board must
        # receive exactly those three slots, not a fresh per-board limit.
        spawned = min(headroom, 2 if conn.board == "first" else 99)
        running[conn.board] += spawned
        dispatched.append((conn.board, kwargs["max_in_progress"], kwargs["max_spawn"], spawned))
        if conn.board == "second":
            runner._running = False
        return SimpleNamespace(
            spawned=[("task", "worker", "")] * spawned,
            reclaimed=0,
            crashed=[],
            timed_out=[],
            promoted=0,
            auto_blocked=[],
        )

    monkeypatch.setattr(kb, "dispatch_once", dispatch_once)

    async def no_sleep(_delay):
        pass

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", no_sleep)
    asyncio.run(runner._kanban_dispatcher_watcher())

    assert dispatched == [
        ("first", 17, 17, 2),
        ("second", 6, 6, 3),
    ]
    assert sum(running.values()) == 20


def test_dispatcher_caps_profile_workers_across_active_boards(monkeypatch, tmp_path):
    """One profile's live-worker cap is shared by every gateway-owned board."""
    import asyncio

    import hermes_cli.config as config
    import hermes_cli.profiles as profiles
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    for slug in ("first", "second"):
        kb.create_board(slug)
        with kb.connect(board=slug) as conn:
            kb.create_task(conn, title=slug, assignee="alpha")

    runner = GatewayKanbanWatchersMixin()
    runner._running = True
    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "auto_decompose": False,
                "max_in_progress_per_profile": 1,
            }
        },
    )
    monkeypatch.setattr(
        kb,
        "list_boards",
        lambda include_archived=False: [{"slug": "first"}, {"slug": "second"}],
    )
    monkeypatch.setattr(kb, "_default_spawn", lambda *args, **kwargs: None)
    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    monkeypatch.setattr(
        "gateway.kanban_watchers._acquire_singleton_lock",
        lambda path: (object(), "held"),
    )
    sleep_calls = 0

    async def no_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            runner._running = False

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", no_sleep)
    asyncio.run(runner._kanban_dispatcher_watcher())

    running = 0
    for slug in ("first", "second"):
        with kb.connect(board=slug) as conn:
            running += conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
            ).fetchone()[0]
    assert running == 1


def test_capped_dispatcher_skips_tick_when_board_enumeration_fails(monkeypatch, tmp_path):
    """Aggregate caps must not dispatch from a partial board list."""
    import asyncio
    from types import SimpleNamespace

    import hermes_cli.config as config
    from hermes_cli import kanban_db as kb

    runner = GatewayKanbanWatchersMixin()
    runner._running = True
    dispatched = []

    class Connection:
        def execute(self, query):
            return SimpleNamespace(fetchone=lambda: (0,))

        def close(self):
            pass

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 1,
            "auto_decompose": False,
            "max_spawn": 1,
        }},
    )
    monkeypatch.setattr(kb, "list_boards", lambda include_archived=False: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(kb, "connect", lambda *, board: Connection())
    monkeypatch.setattr(kb, "kanban_home", lambda: tmp_path)
    monkeypatch.setattr(kb, "kanban_db_path", lambda board: tmp_path / board)
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda: (setattr(runner, "_running", False), [])[-1])
    monkeypatch.setattr(kb, "dispatch_once", lambda conn, **kwargs: dispatched.append(kwargs))
    monkeypatch.setattr(
        "gateway.kanban_watchers._acquire_singleton_lock",
        lambda path: (object(), "held"),
    )

    async def no_sleep(_delay):
        pass

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", no_sleep)
    asyncio.run(runner._kanban_dispatcher_watcher())

    assert dispatched == []


def test_dispatcher_fails_closed_when_singleton_lock_is_unavailable(monkeypatch, tmp_path):
    """A dispatcher must not run without its cross-process admission lock."""
    import asyncio
    from types import SimpleNamespace

    import hermes_cli.config as config
    from hermes_cli import kanban_db as kb

    runner = GatewayKanbanWatchersMixin()
    runner._running = True
    dispatched = []

    class Connection:
        def execute(self, query):
            return SimpleNamespace(fetchone=lambda: (0,))

        def close(self):
            pass

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 1,
            "auto_decompose": False,
        }},
    )
    monkeypatch.setattr(kb, "list_boards", lambda include_archived=False: [{"slug": "only"}])
    monkeypatch.setattr(kb, "connect", lambda *, board: Connection())
    monkeypatch.setattr(kb, "kanban_home", lambda: tmp_path)
    monkeypatch.setattr(kb, "kanban_db_path", lambda board: tmp_path / board)
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda: (setattr(runner, "_running", False), [])[-1])
    monkeypatch.setattr(kb, "dispatch_once", lambda conn, **kwargs: dispatched.append(kwargs))
    monkeypatch.setattr(
        "gateway.kanban_watchers._acquire_singleton_lock",
        lambda path: (None, "unavailable"),
    )

    async def no_sleep(_delay):
        pass

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", no_sleep)
    asyncio.run(runner._kanban_dispatcher_watcher())

    assert dispatched == []


def test_capped_dispatcher_rotates_board_admission_across_ticks(monkeypatch, tmp_path):
    """A continuously ready first board cannot starve the next board at cap one."""
    import asyncio
    from types import SimpleNamespace

    import hermes_cli.config as config
    from hermes_cli import kanban_db as kb

    runner = GatewayKanbanWatchersMixin()
    runner._running = True
    running = {"first": 0, "second": 0}
    spawned = []
    sleep_calls = 0

    class Connection:
        def __init__(self, board):
            self.board = board

        def execute(self, query):
            return SimpleNamespace(fetchone=lambda: (running[self.board],))

        def close(self):
            pass

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 1,
            "auto_decompose": False,
            "max_spawn": 1,
        }},
    )
    monkeypatch.setattr(
        kb, "list_boards", lambda include_archived=False: [
            {"slug": "first"}, {"slug": "second"},
        ],
    )
    monkeypatch.setattr(kb, "connect", lambda *, board: Connection(board))
    monkeypatch.setattr(kb, "kanban_home", lambda: tmp_path)
    monkeypatch.setattr(kb, "kanban_db_path", lambda board: tmp_path / board)
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(kb, "has_spawnable_review", lambda conn: False)
    monkeypatch.setattr(
        "gateway.kanban_watchers._acquire_singleton_lock",
        lambda path: (object(), "held"),
    )

    def dispatch_once(conn, **kwargs):
        if kwargs["max_spawn"] > running[conn.board]:
            running[conn.board] += 1
            spawned.append(conn.board)
        return SimpleNamespace(
            spawned=[], reclaimed=0, crashed=[], timed_out=[], promoted=0, auto_blocked=[],
        )

    monkeypatch.setattr(kb, "dispatch_once", dispatch_once)

    async def no_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            running.update(first=0, second=0)
        elif sleep_calls == 3:
            runner._running = False

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", no_sleep)
    asyncio.run(runner._kanban_dispatcher_watcher())

    assert spawned == ["first", "second"]
