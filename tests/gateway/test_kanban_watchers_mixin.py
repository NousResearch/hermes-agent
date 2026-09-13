"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

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


def test_auto_decompose_dispatcher_uses_cooldown_aware_wrapper():
    """Cooling-down tasks do not spend auto-decompose's per-tick attempt budget."""
    from gateway.kanban_watchers_dispatcher import _KanbanDispatcher

    calls = []

    class FakeDecomposer:
        @staticmethod
        def auto_decompose_task(tid, *, author, board):
            calls.append((tid, author, board))
            return SimpleNamespace(ok=False, reason="auto-decompose cooling down", fanout=False)

    assert _KanbanDispatcher._decompose_one(FakeDecomposer, "board-a", "task-1") is None
    assert calls == [("task-1", "auto-decomposer", "board-a")]


