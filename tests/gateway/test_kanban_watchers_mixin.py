"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import asyncio
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


class RecordingArtifactAdapter:
    def __init__(self):
        self.documents = []
        self.images = []
        self.videos = []

    def extract_local_files(self, text):
        return [], text

    async def send_multiple_images(self, chat_id, images, metadata=None):
        self.images.append((chat_id, images, metadata or {}))

    async def send_document(self, chat_id, file_path, metadata=None):
        self.documents.append((chat_id, file_path, metadata or {}))

    async def send_video(self, chat_id, video_path, metadata=None):
        self.videos.append((chat_id, video_path, metadata or {}))


def test_artifact_delivery_uses_preserved_completion_evidence(tmp_path):
    """Notifier should upload durable evidence paths, not only legacy payload paths."""
    durable = tmp_path / "report.pdf"
    durable.write_bytes(b"durable report")
    missing = tmp_path / "missing.pdf"
    adapter = RecordingArtifactAdapter()

    asyncio.run(
        GatewayKanbanWatchersMixin()._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat-1",
            metadata={"source": "test"},
            event_payload={
                "completion_artifact_evidence": [
                    {"status": "preserved", "stored_path": str(durable)},
                    {
                        "status": "missing",
                        "original_path": str(missing),
                        "reason": "source_missing_at_completion",
                    },
                ]
            },
            task=None,
        )
    )

    assert adapter.documents == [("chat-1", str(durable), {"source": "test"})]
    assert adapter.images == []
    assert adapter.videos == []
