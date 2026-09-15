"""A pinned ``HERMES_KANBAN_BOARD`` confines the dispatcher's lock gate.

Isolated worker gateways (one OS user per profile, sharing a kanban tree via
``HERMES_KANBAN_HOME``) pin ``HERMES_KANBAN_BOARD`` to their own delivery
board. Without this guard the embedded dispatcher enumerates and races the
lock for EVERY board it can see, including ones that belong to sibling
workers — winning that race would spawn the other board's task workers
under the wrong OS UID, defeating per-worker process isolation.
"""

from __future__ import annotations

from gateway.kanban_watchers import GatewayKanbanWatchersMixin


class _Runner(GatewayKanbanWatchersMixin):
    pass


class _FakeKB:
    @staticmethod
    def dispatcher_lock_path(slug: str):
        raise AssertionError(f"lock_path resolved for out-of-scope board {slug!r}")


def test_pinned_board_refuses_foreign_board_without_attempting_lock(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "arnold-delivery-tim-personal")
    runner = _Runner()

    # Foreign board: refused outright, lock path never even resolved.
    assert runner._kanban_board_lock_gate(_FakeKB(), "arnold-delivery-nucleon-holdings") is False
    assert runner._kanban_board_lock_gate(_FakeKB(), "default") is False


def test_pinned_board_still_locks_its_own_board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "arnold-delivery-tim-personal")
    runner = _Runner()

    class _KB:
        @staticmethod
        def dispatcher_lock_path(slug: str):
            return tmp_path / f"{slug}.dispatcher.lock"

    assert runner._kanban_board_lock_gate(_KB(), "arnold-delivery-tim-personal") is True
    assert runner._owns_kanban_dispatcher_lock() is True


def test_unpinned_gateway_may_still_lock_any_board(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    runner = _Runner()

    class _KB:
        @staticmethod
        def dispatcher_lock_path(slug: str):
            return tmp_path / f"{slug}.dispatcher.lock"

    assert runner._kanban_board_lock_gate(_KB(), "default") is True
    assert runner._kanban_board_lock_gate(_KB(), "arnold-delivery-nucleon-holdings") is True
