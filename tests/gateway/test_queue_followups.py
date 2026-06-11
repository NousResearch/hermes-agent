"""Tests for the four queue follow-ups ported from PR #44177 onto main.

Quick-Wins:
1. /queue (no args) — numbered inventory preview
2. /queue drop [N] — drop last (or last N) items
3. /queue clear — wipe entire queue
4. /queue <text> — explicit enqueue, useful when busy

Follow-ups:
1. Per-session queue-TTL — auto-expire items after 24h
2. Queue contents in /compress summary
3. Queue-position in /stop ack
4. Bulk-drop via /queue drop N

Plus crash-recovery via SQLite (queue_persistence.sqlite).

Each test exercises real methods on a real GatewayRunner (object.__new__
bypass) so the FIFO / persistence / TTL logic is hit end-to-end.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _StubAdapter(BasePlatformAdapter):
    """Minimal in-memory adapter for queue tests.

    Real adapters back _pending_messages with the platform's API; for
    unit tests we just need a dict so the FIFO + persistence helpers
    can do their bookkeeping.
    """

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _make_runner(tmp_path) -> tuple[GatewayRunner, _StubAdapter, str]:
    """Bypass __init__ and wire up just what the queue helpers need.

    The HERMES_HOME env var is set to tmp_path so the SQLite queue
    persistence DB lands in a per-test temp directory and is cleaned
    up automatically.
    """
    os.environ["HERMES_HOME"] = str(tmp_path)
    # The persistence helper uses get_hermes_home() to resolve the DB
    # path.  In this codebase it's a plain function (not cached), so
    # setting HERMES_HOME before the first call is enough.
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
    )
    runner._running = True
    runner._queued_events = {}
    runner._queue_persistence_conn = None
    runner._queue_ttl_check_counter = 0
    runner._running_agents = {}
    runner._busy_input_mode = "queue"  # the default for the queue helpers
    runner.adapters = {Platform.TELEGRAM: _StubAdapter()}
    session_key = "telegram:user:123"
    return runner, runner.adapters[Platform.TELEGRAM], session_key


def _make_event(text: str, *, source=None) -> MessageEvent:
    if source is None:
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            user_id="u1",
        )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=None,
    )


def _make_event_with_age(text: str, age_seconds: float) -> SimpleNamespace:
    """Like _make_event but with a backdated _queued_at for TTL tests."""
    e = SimpleNamespace(
        text=text,
        source=SimpleNamespace(platform=Platform.TELEGRAM, user_id="u1", chat_id="123"),
        message_id=None,
        channel_prompt=None,
        message_type=MessageType.TEXT,
    )
    e._queued_at = time.time() - age_seconds
    return e


# ---------------------------------------------------------------------------
# Quick-Win 1: /queue (no args) — inventory preview
# ---------------------------------------------------------------------------


class TestQueueInventory:
    def test_empty_inventory_returns_zero_items(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        items = runner._queue_inventory(session_key, adapter=adapter)
        assert items == []

    def test_inventory_messages_empty_queue(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        msg = runner._queue_inventory_message(session_key, adapter)
        assert "empty" in msg.lower()
        assert "/queue <text>" in msg

    def test_inventory_with_slot_only(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        adapter._pending_messages[session_key] = _make_event("first queued message")
        items = runner._queue_inventory(session_key, adapter=adapter)
        assert len(items) == 1
        assert "first queued message" in items[0]
        assert items[0].startswith("  1.")

    def test_inventory_with_overflow_only(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = [
            _make_event("a"),
            _make_event("b"),
            _make_event("c"),
        ]
        items = runner._queue_inventory(session_key, adapter=adapter)
        assert len(items) == 3
        assert "a" in items[0]
        assert "b" in items[1]
        assert "c" in items[2]
        # Positions are 1-based and continue from where the slot would be.
        assert items[0].startswith("  1.")
        assert items[1].startswith("  2.")
        assert items[2].startswith("  3.")

    def test_inventory_with_slot_and_overflow(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        adapter._pending_messages[session_key] = _make_event("slot")
        runner._queued_events[session_key] = [
            _make_event("tail1"),
            _make_event("tail2"),
        ]
        items = runner._queue_inventory(session_key, adapter=adapter)
        assert len(items) == 3
        assert "slot" in items[0]
        assert "tail1" in items[1]
        assert "tail2" in items[2]

    def test_inventory_truncates_long_text(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        long_text = "x" * 200
        runner._queued_events[session_key] = [_make_event(long_text)]
        items = runner._queue_inventory(session_key, adapter=adapter, max_preview=10)
        assert "..." in items[0]
        assert "x" * 10 in items[0]
        assert "x" * 11 not in items[0]


# ---------------------------------------------------------------------------
# Quick-Win 2 + Follow-up 4: /queue drop [N]
# ---------------------------------------------------------------------------


class TestQueueDrop:
    def test_drop_empty_queue_no_op(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        result = runner._queue_drop_command(session_key, adapter, count=1)
        assert "empty" in result.lower()

    def test_drop_drops_last_overflow_item(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = [
            _make_event("a"),
            _make_event("b"),
            _make_event("c"),
        ]
        result = runner._queue_drop_command(session_key, adapter, count=1)
        # 'c' should be gone — only 'a' and 'b' remain in overflow.
        remaining = [e.text for e in runner._queued_events[session_key]]
        assert "c" not in remaining
        assert remaining == ["a", "b"]

    def test_drop_count_zero_rejected(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = [_make_event("a")]
        result = runner._queue_drop_command(session_key, adapter, count=0)
        assert ">= 1" in result

    def test_drop_count_too_large_rejected(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = [_make_event("a")]
        result = runner._queue_drop_command(session_key, adapter, count=51)
        assert "50" in result

    def test_bulk_drop_three(self, tmp_path):
        """Follow-up 4: /queue drop 3 drops the last 3 items."""
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = [
            _make_event("first"),
            _make_event("second"),
            _make_event("third"),
            _make_event("fourth"),
            _make_event("fifth"),
        ]
        result = runner._queue_drop_command(session_key, adapter, count=3)
        # Should keep first 2
        assert len(runner._queued_events.get(session_key, [])) == 2
        assert "Dropped 3" in result
        # Preview should mention the most recent dropped item (fifth)
        assert "fifth" in result

    def test_bulk_drop_caps_at_depth(self, tmp_path):
        """count > depth is a no-op for the excess — drops only what exists."""
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = [_make_event("only")]
        result = runner._queue_drop_command(session_key, adapter, count=5)
        # Single-item drop: result mentions "Dropped from queue" and the
        # remaining depth.  Either the count (1) or "queue empty" must
        # appear so the user knows the result.
        assert "Dropped" in result
        assert "only" in result
        # Queue should now be empty.
        assert runner._queued_events.get(session_key, []) == []


# ---------------------------------------------------------------------------
# Quick-Win 3: /queue clear
# ---------------------------------------------------------------------------


class TestQueueClear:
    def test_clear_empty_queue(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        result = runner._queue_clear_command(session_key, adapter)
        assert "already empty" in result.lower()

    def test_clear_drops_overflow(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = [
            _make_event("a"),
            _make_event("b"),
        ]
        result = runner._queue_clear_command(session_key, adapter)
        assert "Cleared 2" in result
        assert session_key not in runner._queued_events

    def test_clear_drops_slot_too(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        adapter._pending_messages[session_key] = _make_event("slot")
        runner._queued_events[session_key] = [_make_event("a")]
        result = runner._queue_clear_command(session_key, adapter)
        assert "Cleared 2" in result
        assert session_key not in adapter._pending_messages

    def test_clear_also_wipes_persistence(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = [_make_event("a")]
        runner._persist_queued_event(session_key, _make_event("a"))
        # Verify the row landed in SQLite.
        conn = runner._ensure_queue_persistence_db()
        rows = conn.execute(
            "SELECT COUNT(*) FROM queue_persistence WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        assert rows[0] == 1
        # Now clear.
        runner._queue_clear_command(session_key, adapter)
        rows = conn.execute(
            "SELECT COUNT(*) FROM queue_persistence WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        assert rows[0] == 0


# ---------------------------------------------------------------------------
# Quick-Win 4: capacity check + soft warning
# ---------------------------------------------------------------------------


class TestQueueCapacity:
    def test_capacity_returns_none_under_limit(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = [_make_event("a") for _ in range(5)]
        assert runner._check_queue_capacity(session_key, adapter=adapter) is None

    def test_capacity_returns_error_at_hard_limit(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = [
            _make_event(str(i)) for i in range(runner._QUEUE_HARD_LIMIT)
        ]
        err = runner._check_queue_capacity(session_key, adapter=adapter)
        assert err is not None
        assert "Queue full" in err

    def test_soft_warning_only_fires_past_soft_limit(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        # Just under soft limit — no warning.
        runner._queued_events[session_key] = [
            _make_event(str(i)) for i in range(runner._QUEUE_SOFT_LIMIT - 1)
        ]
        assert runner._queue_soft_warning(session_key, adapter=adapter) == ""
        # At/over soft limit — warning fires.
        runner._queued_events[session_key] = [
            _make_event(str(i)) for i in range(runner._QUEUE_SOFT_LIMIT)
        ]
        warn = runner._queue_soft_warning(session_key, adapter=adapter)
        assert "large" in warn.lower()


# ---------------------------------------------------------------------------
# Follow-up 1: Queue TTL
# ---------------------------------------------------------------------------


class TestQueueTTL:
    def test_expire_stale_drops_old_items(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        # 5 items, 3 of them 1 hour old, 2 fresh.
        runner._queued_events[session_key] = [
            _make_event_with_age("old1", 3600),
            _make_event_with_age("old2", 3600),
            _make_event_with_age("old3", 3600),
            _make_event_with_age("fresh1", 60),
            _make_event_with_age("fresh2", 60),
        ]
        dropped = runner._expire_stale_queue_items(
            session_key, adapter=adapter, ttl_seconds=1800
        )
        assert dropped == 3
        survivors = [e.text for e in runner._queued_events[session_key]]
        assert survivors == ["fresh1", "fresh2"]

    def test_expire_stale_drops_old_slot(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        adapter._pending_messages[session_key] = _make_event_with_age("slot-old", 3600)
        dropped = runner._expire_stale_queue_items(
            session_key, adapter=adapter, ttl_seconds=1800
        )
        assert dropped == 1
        assert session_key not in adapter._pending_messages

    def test_maybe_run_ttl_check_is_amortized(self, tmp_path):
        """Counter should gate the TTL sweep — only run every Nth enqueue."""
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = [
            _make_event_with_age("old", 3600),
        ]
        # Force the counter so the *very next* call triggers the sweep.
        # Counter pre-set to INTERVAL-1 (49), then the method increments
        # to 50, then 50 % 50 == 0 — sweep runs.  We pass ttl_seconds=1800
        # explicitly so the 1-hour-old item is stale relative to the
        # 30-minute window we care about for this test.
        runner._queue_ttl_check_counter = (
            runner._QUEUE_TTL_CHECK_INTERVAL - 1
        )
        # Patch _expire_stale_queue_items for this call only by passing
        # the ttl explicitly via the instance attribute.  Easier: just
        # set the items' _queued_at far enough in the past to exceed the
        # default 24h TTL.
        runner._queued_events[session_key] = [
            _make_event_with_age("old", 25 * 3600),  # 25h old
        ]
        runner._maybe_run_ttl_check(session_key, adapter=adapter)
        # The 50th call (post-increment) does the sweep and expires the
        # 25-hour-old item against the 24-hour default TTL.
        assert runner._queued_events.get(session_key, []) == []
        # Subsequent calls under the threshold don't sweep.
        runner._queued_events[session_key] = [
            _make_event_with_age("old2", 25 * 3600),
        ]
        runner._maybe_run_ttl_check(session_key, adapter=adapter)
        # Item should still be there — the counter has incremented past
        # the threshold but we just bumped it from 50 to 51, which doesn't
        # hit the modulo.
        assert len(runner._queued_events.get(session_key, [])) == 1

    def test_expire_stale_no_op_when_all_fresh(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = [
            _make_event_with_age("a", 60),
            _make_event_with_age("b", 60),
        ]
        dropped = runner._expire_stale_queue_items(
            session_key, adapter=adapter, ttl_seconds=1800
        )
        assert dropped == 0
        assert len(runner._queued_events[session_key]) == 2


# ---------------------------------------------------------------------------
# Crash-recovery: SQLite queue_persistence
# ---------------------------------------------------------------------------


class TestQueuePersistence:
    def test_persist_queued_event_writes_row(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._persist_queued_event(session_key, _make_event("persisted text"))
        conn = runner._ensure_queue_persistence_db()
        rows = conn.execute(
            "SELECT text, position FROM queue_persistence "
            "WHERE session_key = ? ORDER BY position",
            (session_key,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "persisted text"
        assert rows[0][1] == 0

    def test_persist_uses_monotonic_position(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        for i in range(3):
            runner._persist_queued_event(session_key, _make_event(f"msg-{i}"))
        conn = runner._ensure_queue_persistence_db()
        positions = [
            row[0] for row in conn.execute(
                "SELECT position FROM queue_persistence "
                "WHERE session_key = ? ORDER BY position",
                (session_key,),
            ).fetchall()
        ]
        assert positions == [0, 1, 2]

    def test_promote_persisted_event_drops_head(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        for i in range(3):
            runner._persist_queued_event(session_key, _make_event(f"msg-{i}"))
        runner._promote_persisted_event(session_key)
        conn = runner._ensure_queue_persistence_db()
        positions = [
            row[0] for row in conn.execute(
                "SELECT position FROM queue_persistence "
                "WHERE session_key = ? ORDER BY position",
                (session_key,),
            ).fetchall()
        ]
        # The position-0 row was dropped.
        assert positions == [1, 2]

    def test_clear_persisted_queue(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        for i in range(3):
            runner._persist_queued_event(session_key, _make_event(f"msg-{i}"))
        runner._clear_persisted_queue(session_key)
        conn = runner._ensure_queue_persistence_db()
        rows = conn.execute(
            "SELECT COUNT(*) FROM queue_persistence WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        assert rows[0] == 0

    def test_rehydrate_restores_after_fake_restart(self, tmp_path):
        """Simulate a hard crash: write rows, throw away the runner,
        build a fresh one, confirm rehydrate pulls the rows back."""
        # Phase 1: write some queued events.
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = []
        for text in ("survives1", "survives2"):
            ev = SimpleNamespace(
                text=text,
                source=SimpleNamespace(
                    platform=Platform.TELEGRAM, user_id="u1", chat_id="123"
                ),
                message_id=None,
                channel_prompt=None,
            )
            ev._queued_at = time.time()
            runner._enqueue_fifo(session_key, ev, adapter)
        # Phase 2: simulate hard crash — drop the in-memory dict.
        runner._queued_events = {}
        # Phase 3: rehydrate.  This is what start() does at boot.
        rehydrated = runner._rehydrate_queue_persistence()
        assert rehydrated == 2
        assert len(runner._queued_events[session_key]) == 2
        # The rehydrated placeholders have _queued_at preserved.
        for ev in runner._queued_events[session_key]:
            assert ev._queued_at > 0
            assert ev.text in ("survives1", "survives2")

    def test_persistence_db_uses_wal_mode(self, tmp_path):
        """WAL + synchronous=NORMAL is the durability/speed tradeoff."""
        runner, adapter, session_key = _make_runner(tmp_path)
        conn = runner._ensure_queue_persistence_db()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


# ---------------------------------------------------------------------------
# Follow-up 2: /compress queue summary
# ---------------------------------------------------------------------------


class TestCompactSummary:
    def test_get_queue_for_compact_empty(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        summary = runner.get_queue_for_compact(session_key, adapter=adapter)
        assert summary == ""

    def test_get_queue_for_compact_with_items(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = [
            _make_event("first"),
            _make_event("second"),
        ]
        summary = runner.get_queue_for_compact(session_key, adapter=adapter)
        assert "2 pending" in summary
        assert "first" in summary
        assert "second" in summary
        # Should warn the user that these won't carry into the compressed
        # context.
        assert "NOT" in summary or "not" in summary.lower()

    def test_get_queue_for_compact_singular(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        runner._queued_events[session_key] = [_make_event("only-one")]
        summary = runner.get_queue_for_compact(session_key, adapter=adapter)
        # Singular form for 1 item — "1 pending item" (no 's').
        assert "1 pending item" in summary
        assert "items" not in summary.splitlines()[0]


# ---------------------------------------------------------------------------
# Follow-up 3: /stop queue position
# ---------------------------------------------------------------------------


class TestStopQueuePosition:
    """The position line is the key thing — it's just a string-building
    helper so we test the underlying _queue_depth + _queue_inventory
    chain rather than mocking out the full stop handler."""

    def test_stop_position_line_empty_queue(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        depth = runner._queue_depth(session_key, adapter=adapter)
        assert depth == 0
        items = runner._queue_inventory(session_key, adapter=adapter)
        assert items == []

    def test_stop_position_line_with_queue(self, tmp_path):
        runner, adapter, session_key = _make_runner(tmp_path)
        adapter._pending_messages[session_key] = _make_event("next-up")
        depth = runner._queue_depth(session_key, adapter=adapter)
        items = runner._queue_inventory(session_key, adapter=adapter)
        assert depth == 1
        assert "next-up" in items[0]


# ---------------------------------------------------------------------------
# Integration: full /queue <text> → /queue drop / /queue clear flow
# ---------------------------------------------------------------------------


class TestQueueEndToEnd:
    def test_enqueue_then_drop_then_clear(self, tmp_path):
        """Drive the FIFO through the public helpers as the slash
        command handler does."""
        runner, adapter, session_key = _make_runner(tmp_path)

        # /queue <text> x3
        for text in ("a", "b", "c"):
            ev = _make_event(text)
            ev._queued_at = time.time()
            runner._enqueue_fifo(session_key, ev, adapter)

        depth = runner._queue_depth(session_key, adapter=adapter)
        assert depth == 3

        # /queue drop (drops the last)
        runner._queue_drop_command(session_key, adapter, count=1)
        assert runner._queue_depth(session_key, adapter=adapter) == 2

        # /queue clear
        result = runner._queue_clear_command(session_key, adapter)
        assert "Cleared" in result
        assert runner._queue_depth(session_key, adapter=adapter) == 0

    def test_persistence_tracks_through_full_lifecycle(self, tmp_path):
        """Write 3, drop 1, clear — verify SQLite ends up empty."""
        runner, adapter, session_key = _make_runner(tmp_path)
        for text in ("x", "y", "z"):
            ev = _make_event(text)
            ev._queued_at = time.time()
            runner._enqueue_fifo(session_key, ev, adapter)

        conn = runner._ensure_queue_persistence_db()
        rows = conn.execute(
            "SELECT COUNT(*) FROM queue_persistence WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        assert rows[0] == 3

        runner._queue_drop_command(session_key, adapter, count=1)
        rows = conn.execute(
            "SELECT COUNT(*) FROM queue_persistence WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        # The promote hook should have removed the head row.
        assert rows[0] == 2

        runner._queue_clear_command(session_key, adapter)
        rows = conn.execute(
            "SELECT COUNT(*) FROM queue_persistence WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        assert rows[0] == 0
