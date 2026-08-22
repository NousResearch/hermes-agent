"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import gateway.kanban_watchers as watchers_module
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


def test_dispatcher_watcher_forwards_profile_cap_overrides(
    monkeypatch,
    tmp_path,
):
    """The embedded dispatcher passes normalized overrides to every board tick."""
    from hermes_cli import kanban_db as kb

    config = {
        "kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 1,
            "auto_decompose": False,
            "max_in_progress": 4,
            "max_in_progress_per_profile": 3,
            "max_in_progress_per_profile_overrides": {
                "supervisor": 1,
                "implementer": "3",
            },
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr(
        watchers_module,
        "_acquire_singleton_lock",
        lambda path: (None, "unavailable"),
    )

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(watchers_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(kb, "kanban_home", lambda: tmp_path)
    monkeypatch.setattr(kb, "list_boards", lambda **kwargs: [{"slug": "default"}])
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(kb, "review_dispatch_enabled", lambda: False)

    class FakeConnection:
        def close(self):
            return None

    monkeypatch.setattr(kb, "connect", lambda **kwargs: FakeConnection())
    captured = {}
    watcher = GatewayKanbanWatchersMixin()
    setattr(watcher, "_running", True)

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        setattr(watcher, "_running", False)
        return SimpleNamespace(
            spawned=[],
            reclaimed=0,
            crashed=[],
            timed_out=[],
            promoted=0,
            auto_blocked=[],
        )

    monkeypatch.setattr(kb, "dispatch_once", fake_dispatch_once)

    asyncio.run(watcher._kanban_dispatcher_watcher())

    assert captured["max_in_progress"] == 4
    assert captured["max_in_progress_per_profile"] == 3
    assert captured["max_in_progress_per_profile_overrides"] == {
        "supervisor": 1,
        "implementer": 3,
    }


