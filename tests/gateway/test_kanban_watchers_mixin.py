"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from types import SimpleNamespace

import pytest

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


class _CountConn:
    def __init__(self, count: int):
        self.count = count
        self.closed = False

    def execute(self, _sql: str):
        return self

    def fetchone(self):
        return (self.count,)

    def close(self):
        self.closed = True


class _FakeKanbanDb:
    DEFAULT_BOARD = "default"

    def __init__(self, tmp_path, counts: dict[str, int], shared_db: bool = False):
        self._tmp_path = tmp_path
        self.counts = counts
        self.shared_db = shared_db
        self.connected: list[str] = []
        self.connections: list[_CountConn] = []

    def list_boards(self, include_archived: bool = False):
        return [
            {
                "slug": slug,
                "db_path": str(
                    self._tmp_path
                    / ("shared.db" if self.shared_db else f"{slug}.db")
                ),
            }
            for slug in self.counts
        ]

    def kanban_db_path(self, board: str | None = None):
        slug = board or self.DEFAULT_BOARD
        return self._tmp_path / f"{slug}.db"

    def connect(self, board: str | None = None):
        slug = board or self.DEFAULT_BOARD
        self.connected.append(slug)
        conn = _CountConn(self.counts[slug])
        self.connections.append(conn)
        return conn


@pytest.mark.parametrize(
    ("ready", "spawned", "capped", "cap_full", "expected"),
    [
        (True, False, False, True, False),
        (True, False, True, False, False),
        (True, True, False, False, False),
        (False, False, False, False, False),
        (True, False, False, False, True),
    ],
)
def test_dispatcher_stalled_decision(ready, spawned, capped, cap_full, expected):
    from gateway.kanban_watchers import _kanban_dispatcher_stalled

    assert _kanban_dispatcher_stalled(ready, spawned, capped, cap_full) is expected


@pytest.mark.parametrize(
    ("counts", "max_spawn", "max_in_progress", "expected", "connected", "shared_db"),
    [
        ({"default": 2}, 2, None, True, ["default"], False),
        ({"default": 3}, 2, None, True, ["default"], False),
        ({"default": 1}, 2, None, False, ["default"], False),
        ({"default": 2}, None, 2, True, ["default"], False),
        ({"default": 1, "project": 1}, 2, None, True, ["default", "project"], False),
        ({"default": 2, "alias": 2}, 3, None, False, ["default"], True),
        ({"default": 10}, None, None, False, [], False),
        ({"default": 2}, "2", None, True, ["default"], False),
        ({"default": 10}, "invalid", None, False, [], False),
        ({"default": 0}, 0, None, True, ["default"], False),
        ({"default": 0}, -1, None, True, ["default"], False),
        ({"default": 1}, True, None, True, ["default"], False),
        ({"default": 0}, False, None, True, ["default"], False),
    ],
)
def test_dispatcher_cap_full_counts_running_workers(
    tmp_path, counts, max_spawn, max_in_progress, expected, connected, shared_db
):
    from gateway.kanban_watchers import _kanban_dispatcher_cap_full

    kb = _FakeKanbanDb(tmp_path, counts, shared_db=shared_db)

    assert _kanban_dispatcher_cap_full(
        kb,
        max_spawn=max_spawn,
        max_in_progress=max_in_progress,
    ) is expected
    assert kb.connected == connected
    assert all(conn.closed for conn in kb.connections)


def test_dispatcher_cap_full_falls_back_open_on_database_failure(tmp_path):
    from gateway.kanban_watchers import _kanban_dispatcher_cap_full

    class FailingKb(_FakeKanbanDb):
        def connect(self, board: str | None = None):
            raise RuntimeError("database unavailable")

    kb = FailingKb(tmp_path, {"default": 2})

    assert _kanban_dispatcher_cap_full(
        kb,
        max_spawn=2,
        max_in_progress=None,
    ) is False


def test_gateway_dispatcher_does_not_warn_when_external_workers_fill_cap(
    monkeypatch, tmp_path, caplog
):
    from gateway.run import GatewayRunner
    import hermes_cli.config as _cfg_mod
    import hermes_cli.kanban_db as _kb

    runner = object.__new__(GatewayRunner)
    runner._running = True
    kb = _FakeKanbanDb(tmp_path, {"default": 2})
    ticks = {"count": 0}

    monkeypatch.setattr(
        _cfg_mod,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "max_spawn": 2,
                "auto_decompose": False,
            }
        },
    )
    monkeypatch.setattr(_kb, "list_boards", kb.list_boards)
    monkeypatch.setattr(_kb, "kanban_db_path", kb.kanban_db_path)
    monkeypatch.setattr(_kb, "connect", kb.connect)
    monkeypatch.setattr(_kb, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(
        _kb,
        "dispatch_once",
        lambda *args, **kwargs: SimpleNamespace(
            spawned=[],
            skipped_per_profile_capped=[],
        ),
    )
    monkeypatch.setattr(_kb, "has_spawnable_ready", lambda conn: True)

    async def _to_thread(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        if getattr(fn, "__name__", "") == "_tick_once":
            ticks["count"] += 1
            if ticks["count"] >= 6:
                runner._running = False
        return result

    async def _sleep(_delay):
        return None

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", _sleep)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        asyncio.run(
            asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=3.0)
        )

    assert "kanban dispatcher stuck" not in caplog.text
