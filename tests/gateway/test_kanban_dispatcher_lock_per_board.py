"""The embedded gateway kanban dispatcher's singleton lock is per-board.

Prior behaviour: one machine-global ``kanban/.dispatcher.lock`` serialised
ALL gateways sharing a kanban tree — whichever gateway acquired it first
locked every other gateway (including isolated worker-profile gateways
with no board overlap) out of dispatching entirely. Now the lock lives at
``kanban/boards/<slug>/.dispatcher.lock`` (see ``kanban_db.dispatcher_lock_path``)
so independent gateways can each dispatch the boards they actually own.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from gateway.kanban_watchers_common import _acquire_singleton_lock, _release_singleton_lock
from gateway.kanban_watchers_dispatcher import _KanbanDispatcher, _DispatcherSettings
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


def _make_runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    return runner


def _settings() -> _DispatcherSettings:
    return _DispatcherSettings(
        interval=60.0, max_spawn=None, max_in_progress=None, failure_limit=2,
        stale_timeout_seconds=0, reconcile_orphans=True, default_assignee=None,
        max_in_progress_per_profile=None,
    )


def test_dispatcher_lock_path_is_per_board(tmp_path, monkeypatch):
    """Two different boards must resolve to two different lock file paths."""
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    path_a = kb.dispatcher_lock_path("board-a")
    path_b = kb.dispatcher_lock_path("board-b")

    assert path_a != path_b
    assert path_a.parent != path_b.parent
    assert path_a.name == ".dispatcher.lock"
    assert path_b.name == ".dispatcher.lock"
    # Scoped under the boards tree, never the shared kanban root.
    assert "boards" in path_a.parts


def test_gateway_can_dispatch_one_board_while_another_gateway_owns_a_different_board(
    tmp_path, monkeypatch,
):
    """A second gateway process holding board B's lock must not block a
    first gateway from acquiring and dispatching board A — the exact
    scenario that was impossible under the old machine-global lock."""
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Simulate a second gateway process already holding board B's lock.
    other_gateway_handle, other_state = _acquire_singleton_lock(kb.dispatcher_lock_path("board-b"))
    assert other_state == "held"
    try:
        runner = _make_runner()

        assert runner._kanban_board_lock_gate(kb, "board-a") is True
        assert runner._kanban_board_lock_gate(kb, "board-b") is False
        assert runner._owns_kanban_dispatcher_lock() is True

        dispatcher = _KanbanDispatcher(
            MagicMock(), _settings(),
            lock_gate=lambda slug: runner._kanban_board_lock_gate(kb, slug),
        )

        # Board enumeration filters out the board this gateway couldn't lock.
        gated_slugs = [s for s in ("board-a", "board-b") if dispatcher.lock_gate(s)]
        assert gated_slugs == ["board-a"]

        runner._release_kanban_dispatcher_lock()
        assert runner._owns_kanban_dispatcher_lock() is False
    finally:
        _release_singleton_lock(other_gateway_handle)


def test_release_drops_every_held_board_lock(tmp_path, monkeypatch):
    """Releasing the dispatcher lock frees ALL boards this gateway had
    acquired, not just the first — a second gateway must be able to take
    over any of them afterwards."""
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    runner = _make_runner()
    assert runner._kanban_board_lock_gate(kb, "board-a") is True
    assert runner._kanban_board_lock_gate(kb, "board-b") is True

    runner._release_kanban_dispatcher_lock()

    # A fresh acquisition attempt on both boards must succeed now that the
    # first gateway released them.
    handle_a, state_a = _acquire_singleton_lock(kb.dispatcher_lock_path("board-a"))
    handle_b, state_b = _acquire_singleton_lock(kb.dispatcher_lock_path("board-b"))
    try:
        assert state_a == "held"
        assert state_b == "held"
    finally:
        _release_singleton_lock(handle_a)
        _release_singleton_lock(handle_b)
