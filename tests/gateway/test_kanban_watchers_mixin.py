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


def test_completed_notification_uses_full_payload_summary_preview():
    summary_preview = "x" * 210
    task = SimpleNamespace(result="legacy summary that should not win")

    msg = GatewayKanbanWatchersMixin._kanban_notification_text(
        kind="completed",
        task_id="t_123",
        title="Example title",
        assignee="sentinel",
        event_payload={"summary": summary_preview},
        task=task,
    )

    assert msg == f"✔ @sentinel Kanban t_123 done — Example title\n{summary_preview}"


def test_completed_slack_blocks_are_structured_and_labeled():
    task = SimpleNamespace(result=None)

    blocks = GatewayKanbanWatchersMixin._kanban_notification_slack_blocks(
        kind="completed",
        task_id="t_123",
        title="Implement **mrkdwn** cleanup",
        assignee="sentinel",
        board_slug="hermes-setup",
        event_payload={"summary": "First line preview"},
        task=task,
    )

    assert blocks[0]["text"]["text"] == "✔ **Kanban task completed**"
    assert {field["text"] for field in blocks[1]["fields"]} == {
        "**Task**\n`t_123`",
        "**Status**\n`done`",
        "**Worker**\n`sentinel`",
        "**Board**\n`hermes-setup`",
    }
    assert blocks[2]["text"]["text"] == "**Title**\nImplement **mrkdwn** cleanup"
    assert blocks[3]["text"]["text"] == "**Summary**\nFirst line preview"
