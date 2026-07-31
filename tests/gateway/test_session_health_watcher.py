"""Tests for the session-health watcher that detects and recovers wedged
sessions (#58891).

A session is **wedged** when its last persisted message is an
``assistant(tool_calls)`` with no matching ``tool`` result and no subsequent
``user`` message.  The gateway process is still alive (so ``resume_pending``
was never set by the restart watchdog), yet the session is silently stuck
because nothing triggers a new turn.

The fix has three parts, each exercised here:

1. ``SessionDB.has_dangling_tool_call_tail()`` — DB-level detection.
2. ``GatewayRunner._session_health_watcher()`` — periodic background probe
   that marks wedged sessions ``resume_pending`` with reason
   ``"orphaned_tool_call"`` and schedules a synthetic recovery turn.
3. The ``_is_resume_pending`` branch in ``_handle_message_with_agent`` —
   injects a recovery note that does NOT claim a restart happened (the
   gateway was never down).
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import (
    _AGENT_PENDING_SENTINEL,
    _is_fresh_gateway_interruption,
    _auto_continue_freshness_window,
)
from gateway.session import SessionEntry, SessionSource, SessionStore
from hermes_state import SessionDB
from tests.gateway.restart_test_helpers import (
    make_restart_runner,
    make_restart_source,
)


# ---------------------------------------------------------------------------
# SessionDB.has_dangling_tool_call_tail — unit tests
# ---------------------------------------------------------------------------


def _make_db(tmp_path: Path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


def _make_session(db: SessionDB, sid: str = "20260705_114503_d11bcb") -> str:
    db.create_session(sid, source="telegram")
    return sid


@pytest.fixture
def db(tmp_path):
    d = _make_db(tmp_path)
    _make_session(d)
    return d


class TestHasDanglingToolCallTail:
    """Unit tests for the DB-level wedge detection query."""

    def test_query_uses_insertion_order_index(self, db):
        columns = [
            row[2]
            for row in db._conn.execute(
                "PRAGMA index_info(idx_messages_session_active_id)"
            ).fetchall()
        ]
        assert columns == ["session_id", "active", "id"]

        plan = db._conn.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT role, tool_calls FROM messages "
            "WHERE session_id = ? AND active = 1 "
            "ORDER BY id DESC LIMIT 1",
            ("20260705_114503_d11bcb",),
        ).fetchall()
        detail = " ".join(row[-1] for row in plan)
        assert "idx_messages_session_active_id" in detail, detail
        assert "USE TEMP B-TREE" not in detail, detail

    def test_empty_session_returns_false(self, db):
        assert db.has_dangling_tool_call_tail("20260705_114503_d11bcb") is False

    def test_dangling_assistant_tool_calls_returns_true(self, db):
        """Last message is assistant(tool_calls) with no tool result → wedged."""
        db.append_message("20260705_114503_d11bcb", "user", "hello")
        db.append_message(
            "20260705_114503_d11bcb",
            "assistant",
            "Let me check",
            tool_calls=[{"id": "call_abc", "function": {"name": "patch", "arguments": "{}"}}],
        )
        assert db.has_dangling_tool_call_tail("20260705_114503_d11bcb") is True

    def test_matched_tool_result_returns_false(self, db):
        """assistant(tool_calls) + tool result → not wedged."""
        db.append_message("20260705_114503_d11bcb", "user", "hello")
        db.append_message(
            "20260705_114503_d11bcb",
            "assistant",
            "Let me check",
            tool_calls=[{"id": "call_abc", "function": {"name": "patch", "arguments": "{}"}}],
        )
        db.append_message(
            "20260705_114503_d11bcb",
            "tool",
            "patched successfully",
            tool_call_id="call_abc",
        )
        assert db.has_dangling_tool_call_tail("20260705_114503_d11bcb") is False

    def test_user_message_after_dangling_returns_false(self, db):
        """User message after the dangling tool_call → not wedged (user will trigger turn)."""
        db.append_message("20260705_114503_d11bcb", "user", "hello")
        db.append_message(
            "20260705_114503_d11bcb",
            "assistant",
            "Let me check",
            tool_calls=[{"id": "call_abc", "function": {"name": "patch", "arguments": "{}"}}],
        )
        db.append_message("20260705_114503_d11bcb", "user", "any update?")
        assert db.has_dangling_tool_call_tail("20260705_114503_d11bcb") is False

    def test_assistant_without_tool_calls_returns_false(self, db):
        """Last message is assistant without tool_calls → not wedged."""
        db.append_message("20260705_114503_d11bcb", "user", "hello")
        db.append_message("20260705_114503_d11bcb", "assistant", "Hi there!")
        assert db.has_dangling_tool_call_tail("20260705_114503_d11bcb") is False

    def test_tool_result_as_last_message_returns_false(self, db):
        """Last message is a tool result → not wedged (turn completed)."""
        db.append_message(
            "20260705_114503_d11bcb",
            "assistant",
            "Checking",
            tool_calls=[{"id": "call_abc", "function": {"name": "read_file", "arguments": "{}"}}],
        )
        db.append_message(
            "20260705_114503_d11bcb",
            "tool",
            "file contents",
            tool_call_id="call_abc",
        )
        assert db.has_dangling_tool_call_tail("20260705_114503_d11bcb") is False

    def test_inactive_dangling_tail_returns_false(self, db):
        """A soft-deleted (active=0) dangling tail must not count as wedged."""
        db.append_message("20260705_114503_d11bcb", "user", "hello")
        msg_id = db.append_message(
            "20260705_114503_d11bcb",
            "assistant",
            "Let me check",
            tool_calls=[{"id": "call_abc", "function": {"name": "patch", "arguments": "{}"}}],
        )
        # Soft-delete the dangling assistant message
        with db._lock:
            db._conn.execute(
                "UPDATE messages SET active = 0 WHERE id = ?", (msg_id,)
            )
            db._conn.commit()
        assert db.has_dangling_tool_call_tail("20260705_114503_d11bcb") is False

    def test_nonexistent_session_returns_false(self, db):
        assert db.has_dangling_tool_call_tail("nonexistent_session") is False

    def test_multiple_tool_calls_all_unanswered_returns_true(self, db):
        """Multiple tool_calls in one assistant message, none answered → wedged."""
        db.append_message("20260705_114503_d11bcb", "user", "do two things")
        db.append_message(
            "20260705_114503_d11bcb",
            "assistant",
            "On it",
            tool_calls=[
                {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "call_2", "function": {"name": "patch", "arguments": "{}"}},
            ],
        )
        assert db.has_dangling_tool_call_tail("20260705_114503_d11bcb") is True

    def test_empty_tool_calls_string_returns_false(self, db):
        """tool_calls column is empty string (not NULL) → not dangling."""
        db.append_message("20260705_114503_d11bcb", "user", "hello")
        msg_id = db.append_message("20260705_114503_d11bcb", "assistant", "Hi")
        # The append_message above stores tool_calls as NULL when not passed.
        # Manually set it to empty string to test the edge case.
        with db._lock:
            db._conn.execute(
                "UPDATE messages SET tool_calls = '' WHERE id = ?",
                (msg_id,),
            )
            db._conn.commit()
        assert db.has_dangling_tool_call_tail("20260705_114503_d11bcb") is False

    def test_assistant_tool_calls_followed_by_assistant_no_tool_calls(self, db):
        """assistant(tool_calls) → assistant(text) → not wedged (turn continued)."""
        db.append_message(
            "20260705_114503_d11bcb",
            "assistant",
            "Checking",
            tool_calls=[{"id": "call_abc", "function": {"name": "read_file", "arguments": "{}"}}],
        )
        db.append_message(
            "20260705_114503_d11bcb",
            "tool",
            "contents",
            tool_call_id="call_abc",
        )
        db.append_message("20260705_114503_d11bcb", "assistant", "Here is the result")
        assert db.has_dangling_tool_call_tail("20260705_114503_d11bcb") is False


# ---------------------------------------------------------------------------
# _session_health_watcher — integration tests
# ---------------------------------------------------------------------------


def _make_store(tmp_path) -> SessionStore:
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _add_wedged_session(store: SessionStore, db: SessionDB, session_key: str, sid: str):
    """Create a session entry + DB state that simulates a wedged session."""
    source = make_restart_source(chat_id=session_key)
    entry = store.get_or_create_session(source)
    entry.session_id = sid
    store._save()
    db.create_session(sid, source="telegram")
    db.append_message(sid, "user", "hello")
    db.append_message(
        sid,
        "assistant",
        "Let me check",
        tool_calls=[{"id": "call_abc", "function": {"name": "patch", "arguments": "{}"}}],
    )
    return entry


class TestSessionHealthWatcher:
    """Integration tests for the _session_health_probe (one pass of the
    _session_health_watcher).  Tests call _session_health_probe directly to
    avoid the 90-second initial delay and the sleep loop."""

    def _setup_runner(self, tmp_path, chat_id="wedged_chat", platform=Platform.TELEGRAM):
        runner, adapter = make_restart_runner()
        store = _make_store(tmp_path)
        db = _make_db(tmp_path)
        runner.session_store = store
        store._db = db
        store._ensure_loaded()

        source = SessionSource(
            platform=platform,
            chat_id=chat_id,
            chat_type="dm",
            user_id="u1",
        )
        entry = store.get_or_create_session(source)
        entry.session_id = f"session_{chat_id}"
        store._save()
        return runner, adapter, store, db, source, entry

    def _add_wedged_messages(self, db, sid):
        db.create_session(sid, source="telegram")
        db.append_message(sid, "user", "hello")
        db.append_message(
            sid,
            "assistant",
            "Let me check",
            tool_calls=[{"id": "call_abc", "function": {"name": "patch", "arguments": "{}"}}],
        )

    @pytest.mark.asyncio
    async def test_probe_detects_and_marks_wedged_session(self, tmp_path):
        """A wedged session is detected, marked resume_pending, and scheduled for recovery."""
        runner, adapter, store, db, source, entry = self._setup_runner(tmp_path)
        self._add_wedged_messages(db, entry.session_id)

        resume_called = asyncio.Event()
        resume_args = {}

        async def _fake_resume(adapter_arg, event_arg, key_arg):
            resume_args["adapter"] = adapter_arg
            resume_args["event"] = event_arg
            resume_args["key"] = key_arg
            resume_called.set()
            runner._release_running_agent_state(key_arg)

        runner._run_startup_resume_event = _fake_resume
        runner._persist_active_agents = MagicMock()

        scheduled = await runner._session_health_probe()
        # Yield control so the scheduled asyncio.create_task can run
        await asyncio.sleep(0.1)

        assert scheduled == 1
        session_key = runner._session_key_for_source(source)
        updated = store._entries.get(session_key)
        assert updated.resume_pending is True
        assert updated.resume_reason == "orphaned_tool_call"
        assert resume_called.is_set()
        assert resume_args["key"] == session_key
        assert resume_args["event"].text == ""
        assert resume_args["event"].internal is True

    @pytest.mark.asyncio
    async def test_probe_uses_async_store_before_claiming_runner_slot(self, tmp_path):
        runner, adapter, store, db, source, entry = self._setup_runner(tmp_path)
        self._add_wedged_messages(db, entry.session_id)
        session_key = runner._session_key_for_source(source)
        real_facade = runner.async_session_store

        async def mark_resume_pending(key, reason):
            assert key not in runner._running_agents
            return await real_facade.mark_resume_pending(key, reason)

        facade = MagicMock()
        facade._store = store
        facade.list_session_items = AsyncMock(
            side_effect=real_facade.list_session_items
        )
        facade.has_dangling_tool_call_tail = AsyncMock(
            side_effect=real_facade.has_dangling_tool_call_tail
        )
        facade.mark_resume_pending = AsyncMock(side_effect=mark_resume_pending)
        runner._async_session_store = facade
        runner._persist_active_agents = MagicMock()

        async def release_slot(_adapter, _event, key):
            runner._release_running_agent_state(key)

        runner._run_startup_resume_event = release_slot

        assert await runner._session_health_probe() == 1
        await asyncio.sleep(0)

        facade.list_session_items.assert_awaited_once_with()
        facade.has_dangling_tool_call_tail.assert_awaited_once_with(entry.session_id)
        facade.mark_resume_pending.assert_awaited_once_with(
            session_key, reason="orphaned_tool_call"
        )

    @pytest.mark.asyncio
    async def test_probe_does_not_replace_runner_claimed_during_persistence(
        self, tmp_path
    ):
        runner, adapter, store, db, source, entry = self._setup_runner(tmp_path)
        self._add_wedged_messages(db, entry.session_id)
        session_key = runner._session_key_for_source(source)
        active_runner = asyncio.create_task(asyncio.sleep(0))
        real_facade = runner.async_session_store

        async def mark_and_claim(key, reason):
            marked = await real_facade.mark_resume_pending(key, reason)
            runner._running_agents[key] = active_runner
            return marked

        facade = MagicMock()
        facade._store = store
        facade.list_session_items = AsyncMock(
            side_effect=real_facade.list_session_items
        )
        facade.has_dangling_tool_call_tail = AsyncMock(
            side_effect=real_facade.has_dangling_tool_call_tail
        )
        facade.mark_resume_pending = AsyncMock(side_effect=mark_and_claim)
        runner._async_session_store = facade
        runner._run_startup_resume_event = AsyncMock()
        runner._persist_active_agents = MagicMock()

        assert await runner._session_health_probe() == 0
        assert runner._running_agents[session_key] is active_runner
        runner._run_startup_resume_event.assert_not_awaited()
        await active_runner

    @pytest.mark.asyncio
    async def test_probe_skips_running_sessions(self, tmp_path):
        """A session with an active agent must not be flagged as wedged."""
        runner, adapter, store, db, source, entry = self._setup_runner(tmp_path, "active_chat")
        self._add_wedged_messages(db, "session_active_chat")

        session_key = runner._session_key_for_source(source)
        runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL

        resume_called = AsyncMock()
        runner._run_startup_resume_event = resume_called
        runner._persist_active_agents = MagicMock()

        scheduled = await runner._session_health_probe()

        assert scheduled == 0
        resume_called.assert_not_called()
        updated = store._entries.get(session_key)
        assert updated.resume_pending is False

    @pytest.mark.asyncio
    async def test_probe_skips_suspended_sessions(self, tmp_path):
        """A suspended session must not be flagged as wedged."""
        runner, adapter, store, db, source, entry = self._setup_runner(tmp_path, "suspended_chat")
        entry.suspended = True
        store._save()
        self._add_wedged_messages(db, "session_suspended_chat")

        resume_called = AsyncMock()
        runner._run_startup_resume_event = resume_called
        runner._persist_active_agents = MagicMock()

        scheduled = await runner._session_health_probe()

        assert scheduled == 0
        resume_called.assert_not_called()
        session_key = runner._session_key_for_source(source)
        updated = store._entries.get(session_key)
        assert updated.resume_pending is False

    @pytest.mark.asyncio
    async def test_probe_skips_already_resume_pending(self, tmp_path):
        """A session already marked resume_pending must not be re-detected."""
        runner, adapter, store, db, source, entry = self._setup_runner(tmp_path, "already_pending")
        entry.resume_pending = True
        entry.resume_reason = "restart_timeout"
        store._save()
        self._add_wedged_messages(db, "session_already_pending")

        resume_called = AsyncMock()
        runner._run_startup_resume_event = resume_called
        runner._persist_active_agents = MagicMock()

        scheduled = await runner._session_health_probe()

        assert scheduled == 0
        resume_called.assert_not_called()
        session_key = runner._session_key_for_source(source)
        updated = store._entries.get(session_key)
        assert updated.resume_reason == "restart_timeout"

    @pytest.mark.asyncio
    async def test_probe_skips_healthy_session(self, tmp_path):
        """A session with a normal last message must not be flagged."""
        runner, adapter, store, db, source, entry = self._setup_runner(tmp_path, "healthy_chat")
        db.create_session("session_healthy_chat", source="telegram")
        db.append_message("session_healthy_chat", "user", "hello")
        db.append_message("session_healthy_chat", "assistant", "Hi there!")

        resume_called = AsyncMock()
        runner._run_startup_resume_event = resume_called
        runner._persist_active_agents = MagicMock()

        scheduled = await runner._session_health_probe()

        assert scheduled == 0
        resume_called.assert_not_called()
        session_key = runner._session_key_for_source(source)
        updated = store._entries.get(session_key)
        assert updated.resume_pending is False

    @pytest.mark.asyncio
    async def test_probe_skips_session_without_adapter(self, tmp_path):
        """A wedged session whose adapter is unavailable is skipped."""
        runner, adapter, store, db, source, entry = self._setup_runner(
            tmp_path, "no_adapter_chat", platform=Platform.DISCORD
        )
        self._add_wedged_messages(db, "session_no_adapter_chat")

        resume_called = AsyncMock()
        runner._run_startup_resume_event = resume_called
        runner._persist_active_agents = MagicMock()

        scheduled = await runner._session_health_probe()

        assert scheduled == 0
        resume_called.assert_not_called()
        session_key = runner._session_key_for_source(source)
        updated = store._entries.get(session_key)
        assert updated.resume_pending is False

    @pytest.mark.asyncio
    async def test_probe_skips_expired_sessions(self, tmp_path):
        """An expiry-finalized session must not be flagged as wedged."""
        runner, adapter, store, db, source, entry = self._setup_runner(tmp_path, "expired_chat")
        entry.expiry_finalized = True
        store._save()
        self._add_wedged_messages(db, "session_expired_chat")

        resume_called = AsyncMock()
        runner._run_startup_resume_event = resume_called
        runner._persist_active_agents = MagicMock()

        scheduled = await runner._session_health_probe()

        assert scheduled == 0
        resume_called.assert_not_called()

    @pytest.mark.asyncio
    async def test_probe_returns_count_of_scheduled_recoveries(self, tmp_path):
        """Multiple wedged sessions are all detected and scheduled in one pass."""
        runner, adapter, store, db, _, _ = self._setup_runner(tmp_path, "chat_1")
        runner._run_startup_resume_event = AsyncMock()
        runner._persist_active_agents = MagicMock()

        # Add three wedged sessions
        for i, chat_id in enumerate(["chat_1", "chat_2", "chat_3"]):
            source = make_restart_source(chat_id=chat_id)
            entry = store.get_or_create_session(source)
            entry.session_id = f"session_{chat_id}"
            store._save()
            sid = f"session_{chat_id}"
            db.create_session(sid, source="telegram")
            db.append_message(sid, "user", "hello")
            db.append_message(
                sid,
                "assistant",
                "Working",
                tool_calls=[{"id": f"call_{i}", "function": {"name": "patch", "arguments": "{}"}}],
            )

        scheduled = await runner._session_health_probe()
        assert scheduled == 3

    @pytest.mark.asyncio
    async def test_probe_recovers_shared_session_id_only_once(self, tmp_path):
        """Routing-key aliases for one durable session must not start duplicate recoveries."""
        runner, adapter, store, db, source, entry = self._setup_runner(
            tmp_path, "alias_chat_1"
        )
        self._add_wedged_messages(db, entry.session_id)

        alias_source = make_restart_source(chat_id="alias_chat_2")
        alias_entry = store.get_or_create_session(alias_source)
        alias_entry.session_id = entry.session_id
        store._save()

        release_resume = asyncio.Event()

        async def _hold_resume(_adapter, _event, key):
            await release_resume.wait()
            runner._release_running_agent_state(key)

        runner._run_startup_resume_event = _hold_resume
        runner._persist_active_agents = MagicMock()
        facade = runner.async_session_store
        facade.has_dangling_tool_call_tail = AsyncMock(
            side_effect=facade.has_dangling_tool_call_tail
        )

        try:
            assert await runner._session_health_probe() == 1
            await asyncio.sleep(0)
            facade.has_dangling_tool_call_tail.assert_awaited_once_with(
                entry.session_id
            )

            # While recovery is active under either routing key, the shared
            # durable session remains ineligible through every alias.
            assert await runner._session_health_probe() == 0
        finally:
            release_resume.set()
            await asyncio.gather(*runner._background_tasks)

    @pytest.mark.asyncio
    async def test_probe_does_not_re_detect_in_second_pass(self, tmp_path):
        """After a session is marked resume_pending, a second probe pass skips it."""
        runner, adapter, store, db, source, entry = self._setup_runner(tmp_path)
        self._add_wedged_messages(db, entry.session_id)

        runner._run_startup_resume_event = AsyncMock(
            side_effect=lambda a, e, k: runner._release_running_agent_state(k)
        )
        runner._persist_active_agents = MagicMock()

        first = await runner._session_health_probe()
        assert first == 1

        # Second pass — session is now resume_pending, should be skipped
        second = await runner._session_health_probe()
        assert second == 0

    @pytest.mark.asyncio
    async def test_probe_skips_unauthorized_session(self, tmp_path):
        """A wedged session whose owner is no longer authorized must not be
        auto-resumed (#23778 parity with _schedule_resume_pending_sessions)."""
        runner, adapter, store, db, source, entry = self._setup_runner(tmp_path)
        self._add_wedged_messages(db, entry.session_id)

        resume_called = AsyncMock()
        runner._run_startup_resume_event = resume_called
        runner._persist_active_agents = MagicMock()
        # Override the allow-all default from make_restart_runner
        runner._is_user_authorized = lambda _source: False

        scheduled = await runner._session_health_probe()

        assert scheduled == 0
        resume_called.assert_not_called()
        session_key = runner._session_key_for_source(source)
        updated = store._entries.get(session_key)
        assert updated.resume_pending is False

    @pytest.mark.asyncio
    async def test_probe_skips_during_drain(self, tmp_path):
        """The probe must not schedule recovery turns while the gateway is
        draining — they would be immediately interrupted by shutdown."""
        runner, adapter, store, db, source, entry = self._setup_runner(tmp_path)
        self._add_wedged_messages(db, entry.session_id)

        resume_called = AsyncMock()
        runner._run_startup_resume_event = resume_called
        runner._persist_active_agents = MagicMock()
        runner._draining = True

        scheduled = await runner._session_health_probe()

        assert scheduled == 0
        resume_called.assert_not_called()
        session_key = runner._session_key_for_source(source)
        updated = store._entries.get(session_key)
        assert updated.resume_pending is False

    @pytest.mark.asyncio
    async def test_probe_skips_session_with_blocking_approval(self, tmp_path):
        """A wedged session with a pending blocking approval must not be
        auto-resumed — the agent is waiting for /approve or /deny, not a
        recovery turn.  A synthetic empty-text turn would spin up the agent
        only to block again on the same approval gate."""
        runner, adapter, store, db, source, entry = self._setup_runner(tmp_path)
        self._add_wedged_messages(db, entry.session_id)

        resume_called = AsyncMock()
        runner._run_startup_resume_event = resume_called
        runner._persist_active_agents = MagicMock()

        session_key = runner._session_key_for_source(source)
        with patch(
            "tools.approval.has_blocking_approval",
            return_value=True,
        ):
            scheduled = await runner._session_health_probe()

        assert scheduled == 0
        resume_called.assert_not_called()
        updated = store._entries.get(session_key)
        assert updated.resume_pending is False


# ---------------------------------------------------------------------------
# Recovery note for orphaned_tool_call reason
# ---------------------------------------------------------------------------


class TestOrphanedToolCallRecoveryNote:
    """Verify the recovery note injected for the orphaned_tool_call reason
    does NOT claim a restart happened (the gateway was never down)."""

    def test_note_for_orphaned_tool_call_does_not_mention_restart(self):
        """The orphaned_tool_call note must not say 'gateway restart' or 'back online'."""
        from tests.gateway.test_restart_resume_pending import _simulate_note_injection

        history = [
            {"role": "user", "content": "hello", "timestamp": time.time()},
            {
                "role": "assistant",
                "content": "Let me check",
                "tool_calls": [{"id": "call_abc", "function": {"name": "patch", "arguments": "{}"}}],
                "timestamp": time.time(),
            },
        ]

        now = datetime.now()
        entry = SessionEntry(
            session_key="agent:main:telegram:dm:1",
            session_id="test_session",
            created_at=now,
            updated_at=now,
            resume_pending=True,
            resume_reason="orphaned_tool_call",
            last_resume_marked_at=now,
        )

        result = _simulate_note_injection(history, "", entry)

        assert "never completed" in result
        assert "automatically recovered" in result
        # Must NOT claim a restart happened
        assert "gateway restart" not in result
        assert "back online" not in result
        assert "shutdown" not in result

    def test_note_for_orphaned_tool_call_with_user_message(self):
        """When a real user message is present, the note addresses it first."""
        from tests.gateway.test_restart_resume_pending import _simulate_note_injection

        history = [
            {"role": "user", "content": "hello", "timestamp": time.time()},
            {
                "role": "assistant",
                "content": "Let me check",
                "tool_calls": [{"id": "call_abc", "function": {"name": "patch", "arguments": "{}"}}],
                "timestamp": time.time(),
            },
        ]

        now = datetime.now()
        entry = SessionEntry(
            session_key="agent:main:telegram:dm:1",
            session_id="test_session",
            created_at=now,
            updated_at=now,
            resume_pending=True,
            resume_reason="orphaned_tool_call",
            last_resume_marked_at=now,
        )

        result = _simulate_note_injection(history, "what happened?", entry)

        assert "never completed" in result
        assert "what happened?" in result
        assert "NEW message" in result

    def test_standard_restart_note_unchanged(self):
        """The existing restart_timeout note must not be affected by the new branch."""
        from tests.gateway.test_restart_resume_pending import _simulate_note_injection

        history = [
            {"role": "user", "content": "hello", "timestamp": time.time()},
        ]

        now = datetime.now()
        entry = SessionEntry(
            session_key="agent:main:telegram:dm:1",
            session_id="test_session",
            created_at=now,
            updated_at=now,
            resume_pending=True,
            resume_reason="restart_timeout",
            last_resume_marked_at=now,
        )

        result = _simulate_note_injection(history, "", entry)

        assert "gateway restart" in result
        assert "back online" in result
        # Must NOT contain the orphaned_tool_call wording
        assert "never completed" not in result


# ---------------------------------------------------------------------------
# _AUTO_RESUME_REASONS includes orphaned_tool_call
# ---------------------------------------------------------------------------


class TestAutoResumeReasons:
    """Verify orphaned_tool_call is in _AUTO_RESUME_REASONS so
    _schedule_resume_pending_sessions picks it up on platform reconnect."""

    def test_orphaned_tool_call_in_auto_resume_reasons(self):
        from gateway.run import GatewayRunner

        assert "orphaned_tool_call" in GatewayRunner._AUTO_RESUME_REASONS

    def test_standard_reasons_still_present(self):
        from gateway.run import GatewayRunner

        reasons = GatewayRunner._AUTO_RESUME_REASONS
        assert "restart_timeout" in reasons
        assert "shutdown_timeout" in reasons
        assert "restart_interrupted" in reasons
