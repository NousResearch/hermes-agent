"""Tests for hermes_cli.kanban_event_bridge — KanbanEventBridge.

Covers:
- Event filtering (by task_id, by kind)
- Cursor persistence (load/save, atomic rename, crash recovery)
- No duplicate events across poll() calls
- Clean shutdown (context manager, manual, idempotent)
- subscribe/unsubscribe
- Empty task set returns no events
- Cursor file corruption tolerance
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Isolate hermes home before importing kanban_db
from hermes_cli import kanban_db as kb


@pytest.fixture
def isolated_kanban(tmp_path, monkeypatch):
    """Set up a fresh kanban DB in a temp dir and return (db_path, cursor_path)."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    kanban_dir = hermes_home / "kanban"
    kanban_dir.mkdir()
    db_path = hermes_home / "kanban.db"
    cursor_path = hermes_home / "kanban-event-bridge-cursors.json"

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Prevent profile override from interfering
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)

    # Initialize the DB
    kb.init_db(db_path, board="default")

    return db_path, cursor_path


@pytest.fixture
def bridge_module(tmp_path, monkeypatch, isolated_kanban):
    """Import kanban_event_bridge with HERMES_HOME pointing to tmp_path."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)

    # Force reimport so kanban_db picks up the new HERMES_HOME
    import importlib
    import hermes_cli.kanban_event_bridge as keb

    importlib.reload(kb)
    importlib.reload(keb)
    return keb


def _seed_events(db_path: Path, task_id: str, kinds: list[str]):
    """Insert synthetic task_events into the kanban DB."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    for i, kind in enumerate(kinds, start=1):
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, 1, kind, json.dumps({"seq": i}), i * 1000),
        )
    conn.commit()
    conn.close()


class TestEventFiltering:
    """Bridge returns only events for subscribed tasks and requested kinds."""

    def test_poll_returns_events_for_subscribed_task(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        task_id = "t_filter01"
        _seed_events(db_path, task_id, ["created", "claimed", "completed"])

        bridge = bridge_module.KanbanEventBridge(
            task_ids=[task_id], cursor_file=str(cursor_path)
        )
        try:
            events = bridge.poll()
            assert len(events) == 3
            assert all(e.task_id == task_id for e in events)
            assert [e.kind for e in events] == ["created", "claimed", "completed"]
        finally:
            bridge.shutdown()

    def test_poll_filters_by_kind(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        task_id = "t_filter02"
        _seed_events(db_path, task_id, ["created", "claimed", "completed"])

        bridge = bridge_module.KanbanEventBridge(
            task_ids=[task_id],
            kinds={"completed"},
            cursor_file=str(cursor_path),
        )
        try:
            events = bridge.poll()
            assert len(events) == 1
            assert events[0].kind == "completed"
        finally:
            bridge.shutdown()

    def test_poll_kind_override_per_call(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        task_id = "t_filter03"
        _seed_events(db_path, task_id, ["created", "claimed", "completed"])

        bridge = bridge_module.KanbanEventBridge(
            task_ids=[task_id], cursor_file=str(cursor_path)
        )
        try:
            # No instance-level filter, but override on this call
            events = bridge.poll(kinds={"claimed"})
            assert len(events) == 1
            assert events[0].kind == "claimed"
        finally:
            bridge.shutdown()

    def test_poll_ignores_unsubscribed_tasks(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        _seed_events(db_path, "t_sub", ["created"])
        _seed_events(db_path, "t_other", ["created", "completed"])

        bridge = bridge_module.KanbanEventBridge(
            task_ids=["t_sub"], cursor_file=str(cursor_path)
        )
        try:
            events = bridge.poll()
            assert len(events) == 1
            assert events[0].task_id == "t_sub"
        finally:
            bridge.shutdown()

    def test_poll_empty_task_set_returns_no_events(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        bridge = bridge_module.KanbanEventBridge(cursor_file=str(cursor_path))
        try:
            events = bridge.poll()
            assert events == []
        finally:
            bridge.shutdown()


class TestCursorPersistence:
    """Cursors are persisted after poll and survive restarts."""

    def test_no_duplicate_events_after_restart(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        task_id = "t_cursor01"
        _seed_events(db_path, task_id, ["created", "claimed"])

        # First session: read events
        bridge = bridge_module.KanbanEventBridge(
            task_ids=[task_id], cursor_file=str(cursor_path)
        )
        events1 = bridge.poll()
        assert len(events1) == 2
        bridge.shutdown()

        # Second session: should get nothing (cursor persisted)
        bridge2 = bridge_module.KanbanEventBridge(
            task_ids=[task_id], cursor_file=str(cursor_path)
        )
        events2 = bridge2.poll()
        assert len(events2) == 0
        bridge2.shutdown()

    def test_cursor_file_is_json_and_inspectable(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        task_id = "t_cursor02"
        _seed_events(db_path, task_id, ["created"])

        bridge = bridge_module.KanbanEventBridge(
            task_ids=[task_id], cursor_file=str(cursor_path)
        )
        bridge.poll()
        bridge.shutdown()

        # Verify file exists and is valid JSON
        assert cursor_path.exists()
        data = json.loads(cursor_path.read_text())
        assert task_id in data
        assert isinstance(data[task_id], int)
        assert data[task_id] > 0

    def test_corrupted_cursor_file_is_tolerated(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        task_id = "t_cursor03"
        _seed_events(db_path, task_id, ["created"])

        # Write garbage to cursor file
        cursor_path.write_text("not valid json {{{")

        bridge = bridge_module.KanbanEventBridge(
            task_ids=[task_id], cursor_file=str(cursor_path)
        )
        events = bridge.poll()
        assert len(events) == 1  # starts from cursor 0, gets all events
        bridge.shutdown()

    def test_cursor_advanced_after_each_poll(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        task_id = "t_cursor04"
        _seed_events(db_path, task_id, ["created", "claimed", "completed"])

        bridge = bridge_module.KanbanEventBridge(
            task_ids=[task_id], cursor_file=str(cursor_path)
        )
        try:
            e1 = bridge.poll()
            assert len(e1) == 3

            # Seed more events
            _seed_events(db_path, task_id, ["spawned", "heartbeat"])

            e2 = bridge.poll()
            assert len(e2) == 2
            assert e2[0].kind == "spawned"
        finally:
            bridge.shutdown()


class TestSubscribeUnsubscribe:
    """Dynamic subscription management."""

    def test_subscribe_adds_task(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        task_id = "t_sub01"
        _seed_events(db_path, task_id, ["created"])

        bridge = bridge_module.KanbanEventBridge(cursor_file=str(cursor_path))
        try:
            bridge.subscribe(task_id)
            events = bridge.poll()
            assert len(events) == 1
            assert events[0].task_id == task_id
        finally:
            bridge.shutdown()

    def test_unsubscribe_stops_delivery(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        task_id = "t_sub02"
        _seed_events(db_path, task_id, ["created"])

        bridge = bridge_module.KanbanEventBridge(
            task_ids=[task_id], cursor_file=str(cursor_path)
        )
        bridge.poll()  # consume initial
        bridge.unsubscribe(task_id)

        # Seed more events
        _seed_events(db_path, task_id, ["completed"])

        events = bridge.poll()
        assert len(events) == 0
        bridge.shutdown()


class TestShutdown:
    """Clean shutdown behavior."""

    def test_context_manager_shuts_down(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        task_id = "t_shutdown01"
        _seed_events(db_path, task_id, ["created"])

        with bridge_module.KanbanEventBridge(
            task_ids=[task_id], cursor_file=str(cursor_path)
        ) as bridge:
            events = bridge.poll()
            assert len(events) == 1
            assert not bridge._closed

        # After context exit, should be closed
        assert bridge._closed
        with pytest.raises(RuntimeError, match="bridge is closed"):
            bridge.poll()

    def test_shutdown_idempotent(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        bridge = bridge_module.KanbanEventBridge(cursor_file=str(cursor_path))
        bridge.shutdown()
        bridge.shutdown()  # should not raise
        assert bridge._closed

    def test_poll_after_shutdown_raises(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        bridge = bridge_module.KanbanEventBridge(cursor_file=str(cursor_path))
        bridge.shutdown()
        with pytest.raises(RuntimeError, match="bridge is closed"):
            bridge.poll()

    def test_subscribe_after_shutdown_raises(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        bridge = bridge_module.KanbanEventBridge(cursor_file=str(cursor_path))
        bridge.shutdown()
        with pytest.raises(RuntimeError, match="bridge is closed"):
            bridge.subscribe("t_new")


class TestBridgeEvent:
    """BridgeEvent dataclass properties."""

    def test_event_fields(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        task_id = "t_evt01"
        _seed_events(db_path, task_id, ["completed"])

        bridge = bridge_module.KanbanEventBridge(
            task_ids=[task_id], cursor_file=str(cursor_path)
        )
        events = bridge.poll()
        ev = events[0]

        assert ev.id > 0
        assert ev.task_id == task_id
        assert ev.kind == "completed"
        assert ev.created_at > 0
        assert ev.raw is not None
        # raw is a kanban_db.Event with payload
        assert ev.raw.payload == {"seq": 1}
        bridge.shutdown()


class TestMultiTask:
    """Multiple tasks in one bridge, events sorted by id."""

    def test_multiple_tasks_sorted_by_id(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        _seed_events(db_path, "t_multi_a", ["created"])
        _seed_events(db_path, "t_multi_b", ["created"])

        bridge = bridge_module.KanbanEventBridge(
            task_ids=["t_multi_a", "t_multi_b"],
            cursor_file=str(cursor_path),
        )
        events = bridge.poll()
        assert len(events) == 2
        # Should be sorted by id
        assert events[0].id < events[1].id
        bridge.shutdown()

    def test_per_task_cursors_independent(self, bridge_module, isolated_kanban):
        db_path, cursor_path = isolated_kanban
        _seed_events(db_path, "t_ind_a", ["created", "claimed"])
        _seed_events(db_path, "t_ind_b", ["created"])

        bridge = bridge_module.KanbanEventBridge(
            task_ids=["t_ind_a", "t_ind_b"],
            cursor_file=str(cursor_path),
        )
        events = bridge.poll()
        assert len(events) == 3

        # Add more to only t_ind_b
        _seed_events(db_path, "t_ind_b", ["completed"])

        events2 = bridge.poll()
        assert len(events2) == 1
        assert events2[0].task_id == "t_ind_b"
        bridge.shutdown()