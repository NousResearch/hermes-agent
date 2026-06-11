"""Tests for /queue UX improvements, capacity limits, and SQLite persistence.

Covers:
- /queue with no args returns inventory
- /queue drop removes the last item
- /queue clear wipes the whole queue
- Capacity limits (soft warning + hard rejection)
- /status surfaces queue contents
- /steer fallbacks (SENTINEL, missing agent) use FIFO path — no clobbering
- SQLite crash-recovery persistence (write, promote, rehydrate, clear)
- Slot-occupied-then-new-item goes to overflow (no race)
"""

import asyncio
import sqlite3
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    PlatformConfig,
    Platform,
)
from gateway.run import GatewayRunner


# ---------------------------------------------------------------------------
# Minimal adapter for testing pending message storage
# ---------------------------------------------------------------------------

class _StubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        # The real base class should set this up; ensure dict exists
        if not hasattr(self, "_pending_messages"):
            self._pending_messages = {}

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        from gateway.platforms.base import SendResult
        return SendResult(success=True, message_id="msg-1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _make_runner(tmp_path: Path) -> GatewayRunner:
    """Build a GatewayRunner with just enough wiring to exercise queue code."""
    with patch("gateway.run.get_hermes_home", return_value=tmp_path):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._queued_events = {}
        runner._queue_persistence_conn = None
        runner._draining = False
        runner._restart_requested = False
        runner.adapters = {"telegram": _StubAdapter()}
        return runner


def _make_event(text: str, message_id: str = "m1") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=MagicMock(chat_id="123", platform=Platform.TELEGRAM, user_id="u1"),
        message_id=message_id,
    )


# ---------------------------------------------------------------------------
# Quick-Win 1: /queue with no args → inventory
# ---------------------------------------------------------------------------

class TestQueueInventory:
    def test_empty_queue_message(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        msg = runner._queue_inventory_message("sess:1", adapter=adapter)
        assert "empty" in msg.lower()
        assert "Use" in msg  # tells user how to add one

    def test_inventory_shows_one_item(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        ev = _make_event("deploy to staging")
        adapter._pending_messages["sess:1"] = ev
        msg = runner._queue_inventory_message("sess:1", adapter=adapter)
        assert "1 item" in msg  # singular
        assert "deploy to staging" in msg

    def test_inventory_shows_multiple_items(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        adapter._pending_messages["sess:1"] = _make_event("first")
        runner._queued_events["sess:1"] = [_make_event("second"), _make_event("third")]
        msg = runner._queue_inventory_message("sess:1", adapter=adapter)
        assert "3 items" in msg  # plural
        assert "first" in msg
        assert "second" in msg
        assert "third" in msg
        assert "/queue drop" in msg
        assert "/queue clear" in msg

    def test_inventory_truncates_long_text(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        adapter._pending_messages["sess:1"] = _make_event("a" * 200)
        items = runner._queue_inventory("sess:1", adapter=adapter, max_preview=30)
        assert len(items) == 1
        assert items[0].endswith("...")
        # 1-indexed position + preview
        assert items[0].startswith("  1. ")
        assert "a" * 30 in items[0]


# ---------------------------------------------------------------------------
# Quick-Win 1b: /status surfaces queue contents
# ---------------------------------------------------------------------------

class TestStatusQueueContents:
    def test_status_inventory_method_lists_all(self, tmp_path):
        """The /status command uses _queue_inventory to surface items."""
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        adapter._pending_messages["sess:1"] = _make_event("run tests")
        runner._queued_events["sess:1"] = [_make_event("commit"), _make_event("push")]
        items = runner._queue_inventory("sess:1", adapter=adapter, max_preview=50)
        assert len(items) == 3
        assert "run tests" in items[0]
        assert "commit" in items[1]
        assert "push" in items[2]


# ---------------------------------------------------------------------------
# Quick-Win 2: /queue drop
# ---------------------------------------------------------------------------

class TestQueueDrop:
    def test_drop_from_overflow(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        runner._queued_events["sess:1"] = [
            _make_event("first"),
            _make_event("second"),
            _make_event("third"),
        ]
        msg = runner._queue_drop_command("sess:1", adapter=adapter)
        assert "Dropped" in msg
        assert "third" in msg
        # two items remain in overflow
        assert len(runner._queued_events["sess:1"]) == 2

    def test_drop_from_slot_when_overflow_empty(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        adapter._pending_messages["sess:1"] = _make_event("only one")
        msg = runner._queue_drop_command("sess:1", adapter=adapter)
        assert "Dropped" in msg
        assert "only one" in msg
        assert "sess:1" not in adapter._pending_messages

    def test_drop_when_queue_empty(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        msg = runner._queue_drop_command("sess:1", adapter=adapter)
        assert "nothing to drop" in msg.lower()

    def test_drop_empties_overflow_list(self, tmp_path):
        """When overflow becomes empty, the session_key entry is removed."""
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        runner._queued_events["sess:1"] = [_make_event("only one")]
        runner._queue_drop_command("sess:1", adapter=adapter)
        assert "sess:1" not in runner._queued_events


# ---------------------------------------------------------------------------
# Quick-Win 2b: /queue clear
# ---------------------------------------------------------------------------

class TestQueueClear:
    def test_clear_removes_overflow_and_slot(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        adapter._pending_messages["sess:1"] = _make_event("slot")
        runner._queued_events["sess:1"] = [
            _make_event("a"),
            _make_event("b"),
            _make_event("c"),
        ]
        msg = runner._queue_clear_command("sess:1", adapter=adapter)
        assert "Cleared 4" in msg  # 1 slot + 3 overflow
        assert "sess:1" not in adapter._pending_messages
        assert "sess:1" not in runner._queued_events

    def test_clear_when_empty(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        msg = runner._queue_clear_command("sess:1", adapter=adapter)
        assert "already empty" in msg.lower()

    def test_clear_singular_message(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        adapter._pending_messages["sess:1"] = _make_event("only")
        msg = runner._queue_clear_command("sess:1", adapter=adapter)
        assert "1 queued item" in msg  # singular


# ---------------------------------------------------------------------------
# Quick-Win 3: Capacity limits
# ---------------------------------------------------------------------------

class TestQueueCapacity:
    def test_hard_limit_rejects(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        # Pre-fill to the hard limit
        runner._queued_events["sess:1"] = [
            _make_event(f"item-{i}") for i in range(GatewayRunner._QUEUE_HARD_LIMIT)
        ]
        err = runner._check_queue_capacity("sess:1", adapter=adapter)
        assert err is not None
        assert "Queue full" in err
        assert "/queue clear" in err

    def test_under_limit_no_error(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        runner._queued_events["sess:1"] = [
            _make_event(f"item-{i}") for i in range(5)
        ]
        err = runner._check_queue_capacity("sess:1", adapter=adapter)
        assert err is None

    def test_soft_warning_kicks_in(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        runner._queued_events["sess:1"] = [
            _make_event(f"item-{i}") for i in range(GatewayRunner._QUEUE_SOFT_LIMIT + 2)
        ]
        warn = runner._queue_soft_warning("sess:1", adapter=adapter)
        assert "⚠️" in warn
        assert str(GatewayRunner._QUEUE_SOFT_LIMIT + 2) in warn

    def test_soft_warning_silent_below_threshold(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        runner._queued_events["sess:1"] = [_make_event("a"), _make_event("b")]
        warn = runner._queue_soft_warning("sess:1", adapter=adapter)
        assert warn == ""


# ---------------------------------------------------------------------------
# Race fix: /steer fallbacks use _enqueue_fifo (no slot clobber)
# ---------------------------------------------------------------------------

class TestEnqueueFIFONoClobber:
    def test_enqueue_when_slot_occupied_goes_to_overflow(self, tmp_path):
        """Adding to a non-empty slot MUST land in overflow, not replace."""
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        adapter._pending_messages["sess:1"] = _make_event("original")
        new_event = _make_event("second")
        runner._enqueue_fifo("sess:1", new_event, adapter)
        # Slot still has the original
        assert adapter._pending_messages["sess:1"].text == "original"
        # Overflow has the new one
        assert len(runner._queued_events["sess:1"]) == 1
        assert runner._queued_events["sess:1"][0].text == "second"

    def test_enqueue_into_empty_slot(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        new_event = _make_event("only")
        runner._enqueue_fifo("sess:1", new_event, adapter)
        assert adapter._pending_messages["sess:1"].text == "only"
        assert "sess:1" not in runner._queued_events

    def test_enqueue_with_no_adapter_is_safe(self, tmp_path):
        runner = _make_runner(tmp_path)
        runner._enqueue_fifo("sess:1", _make_event("orphan"), adapter=None)
        # No crash, nothing happened
        assert "sess:1" not in runner._queued_events

    def test_enqueue_persist_false_skips_sqlite(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        # persist=False should NOT open a sqlite connection
        runner._enqueue_fifo("sess:1", _make_event("x"), adapter, persist=False)
        assert runner._queue_persistence_conn is None


# ---------------------------------------------------------------------------
# Strategic A: SQLite crash-recovery persistence
# ---------------------------------------------------------------------------

class TestQueuePersistence:
    def test_persistence_path_under_hermes_home(self, tmp_path):
        runner = _make_runner(tmp_path)
        path = runner._get_queue_persistence_path()
        # get_hermes_home() may append a subdirectory (e.g. profile name);
        # just verify the file is inside the returned home and has the
        # expected name.
        assert path.name == "queue_persistence.sqlite"
        # Parent should be somewhere under our tmp_path (the stub)
        # WindowsPath is non-strict — both forward and back slashes.
        assert str(path.parent).replace("\\", "/").startswith(
            str(tmp_path).replace("\\", "/")
        )

    def test_ensure_db_creates_schema(self, tmp_path):
        runner = _make_runner(tmp_path)
        conn = runner._ensure_queue_persistence_db()
        assert conn is not None
        # Schema check
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='queue_persistence'"
        ).fetchall()
        assert len(rows) == 1

    def test_persist_and_promote_round_trip(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        runner._enqueue_fifo("sess:1", _make_event("first"), adapter)
        runner._enqueue_fifo("sess:1", _make_event("second"), adapter)
        conn = runner._ensure_queue_persistence_db()
        rows = conn.execute(
            "SELECT position, text FROM queue_persistence WHERE session_key = ? ORDER BY position",
            ("sess:1",),
        ).fetchall()
        assert len(rows) == 2
        # Promote head
        runner._promote_persisted_event("sess:1")
        rows = conn.execute(
            "SELECT position, text FROM queue_persistence WHERE session_key = ? ORDER BY position",
            ("sess:1",),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "second"

    def test_clear_persisted_queue(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        runner._enqueue_fifo("sess:1", _make_event("a"), adapter)
        runner._enqueue_fifo("sess:1", _make_event("b"), adapter)
        runner._clear_persisted_queue("sess:1")
        conn = runner._ensure_queue_persistence_db()
        rows = conn.execute(
            "SELECT COUNT(*) FROM queue_persistence WHERE session_key = ?",
            ("sess:1",),
        ).fetchone()
        assert rows[0] == 0

    def test_rehydrate_loads_from_disk(self, tmp_path):
        # Simulate a previous run: write rows to disk in a fresh runner.
        runner_a = _make_runner(tmp_path)
        adapter_a = runner_a.adapters["telegram"]
        runner_a._enqueue_fifo("sess:1", _make_event("survivor-1"), adapter_a)
        runner_a._enqueue_fifo("sess:1", _make_event("survivor-2"), adapter_a)
        # Close the conn so the new runner opens fresh
        runner_a._queue_persistence_conn.close()
        # New runner simulates startup
        runner_b = _make_runner(tmp_path)
        runner_b._queued_events = {}  # fresh state
        runner_b._queue_persistence_conn = None
        count = runner_b._rehydrate_queue_persistence()
        assert count == 1
        assert len(runner_b._queued_events["sess:1"]) == 2
        # First item's text is preserved
        first = runner_b._queued_events["sess:1"][0]
        assert first.text in ("survivor-1", "survivor-2")

    def test_rehydrate_on_empty_db_returns_zero(self, tmp_path):
        runner = _make_runner(tmp_path)
        # Ensure db exists, then no rows
        runner._ensure_queue_persistence_db()
        count = runner._rehydrate_queue_persistence()
        assert count == 0

    def test_persistence_failure_doesnt_break_enqueue(self, tmp_path):
        """If SQLite fails, enqueue still succeeds (best-effort)."""
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        # Force a broken DB path
        with patch.object(runner, "_ensure_queue_persistence_db", return_value=None):
            runner._enqueue_fifo("sess:1", _make_event("survives"), adapter)
        # Event is still in the in-memory queue
        assert adapter._pending_messages["sess:1"].text == "survives"


# ---------------------------------------------------------------------------
# Integration: _promote_queued_event cleans up persistence
# ---------------------------------------------------------------------------

class TestPromoteQueueEvent:
    def test_promote_with_persistence(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        # Pre-populate the slot so the next enqueue MUST go to overflow.
        adapter._pending_messages["sess:1"] = _make_event("slot-filler")
        runner._enqueue_fifo("sess:1", _make_event("head"), adapter)
        # Now promote with empty pending_event — slot-filler isn't really
        # drained, but we test the promote path which returns the
        # overflow head.
        next_ev = runner._promote_queued_event("sess:1", adapter, pending_event=None)
        # The overflow head is returned
        assert next_ev is not None
        assert next_ev.text == "head"
        # And the persisted row is gone
        conn = runner._ensure_queue_persistence_db()
        rows = conn.execute(
            "SELECT COUNT(*) FROM queue_persistence WHERE session_key = ?",
            ("sess:1",),
        ).fetchone()
        assert rows[0] == 0

    def test_promote_when_empty_is_noop(self, tmp_path):
        runner = _make_runner(tmp_path)
        adapter = runner.adapters["telegram"]
        result = runner._promote_queued_event("sess:1", adapter, pending_event=None)
        assert result is None


# ---------------------------------------------------------------------------
# Edge: graceful degradation when hermes_home is unavailable
# ---------------------------------------------------------------------------

class TestRunnerWithoutHome:
    def test_persistence_path_returns_none(self):
        with patch("gateway.run.get_hermes_home", return_value=None):
            runner = GatewayRunner.__new__(GatewayRunner)
            runner._queue_persistence_conn = None
            assert runner._get_queue_persistence_path() is None

    def test_ensure_db_returns_none_without_home(self):
        with patch("gateway.run.get_hermes_home", return_value=None):
            runner = GatewayRunner.__new__(GatewayRunner)
            runner._queue_persistence_conn = None
            assert runner._ensure_queue_persistence_db() is None
