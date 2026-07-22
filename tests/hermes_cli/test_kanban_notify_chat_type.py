"""Kanban notify-subscription chat_type persistence + wake scoping (#68874).

A task created from a DM via ``/kanban create`` subscribes the conversation
for terminal-state notifications. When the task later reaches a wake kind
(completed / gave_up / crashed / timed_out / blocked), the notifier rebuilds
a synthetic MessageEvent to wake the agent. Previously it hardcoded
``chat_type="group"``, and since DM and group conversations key to different
session ids, the wake resumed a fresh group session instead of the
originating DM. These tests pin the fix: chat_type is persisted on the
subscription and used to rebuild the correct session scope, with a "group"
fallback for legacy rows that never captured it.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ── DB layer: persistence, normalization, self-heal, back-compat ──────────


class TestChatTypePersistence:
    def test_chat_type_round_trips(self, kanban_home):
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="dm task", assignee="w1")
            kb.add_notify_sub(
                conn, task_id=tid, platform="telegram",
                chat_id="c1", chat_type="dm",
            )
            subs = kb.list_notify_subs(conn, tid)
        finally:
            conn.close()
        assert len(subs) == 1
        assert subs[0]["chat_type"] == "dm"

    def test_chat_type_normalized_lowercase_and_trimmed(self, kanban_home):
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="w1")
            kb.add_notify_sub(
                conn, task_id=tid, platform="telegram",
                chat_id="c1", chat_type="  Group ",
            )
            subs = kb.list_notify_subs(conn, tid)
        finally:
            conn.close()
        assert subs[0]["chat_type"] == "group"

    def test_absent_chat_type_stored_as_null(self, kanban_home):
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="w1")
            kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="c1")
            subs = kb.list_notify_subs(conn, tid)
        finally:
            conn.close()
        # Back-compat: existing callers that don't pass chat_type keep working.
        assert subs[0]["chat_type"] is None

    def test_empty_chat_type_is_null_not_empty_string(self, kanban_home):
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="w1")
            kb.add_notify_sub(
                conn, task_id=tid, platform="telegram",
                chat_id="c1", chat_type="   ",
            )
            subs = kb.list_notify_subs(conn, tid)
        finally:
            conn.close()
        assert subs[0]["chat_type"] is None

    def test_resubscribe_backfills_missing_chat_type(self, kanban_home):
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="w1")
            # First subscribe without a chat_type (legacy-shaped row).
            kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="c1")
            # A later subscribe from a known DM scope self-heals the row.
            kb.add_notify_sub(
                conn, task_id=tid, platform="telegram",
                chat_id="c1", chat_type="dm",
            )
            subs = kb.list_notify_subs(conn, tid)
        finally:
            conn.close()
        assert len(subs) == 1  # still idempotent on (task, platform, chat, thread)
        assert subs[0]["chat_type"] == "dm"

    def test_resubscribe_does_not_overwrite_existing_chat_type(self, kanban_home):
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="w1")
            kb.add_notify_sub(
                conn, task_id=tid, platform="telegram",
                chat_id="c1", chat_type="dm",
            )
            # A spurious later subscribe with a different scope must not clobber
            # the captured DM scope.
            kb.add_notify_sub(
                conn, task_id=tid, platform="telegram",
                chat_id="c1", chat_type="group",
            )
            subs = kb.list_notify_subs(conn, tid)
        finally:
            conn.close()
        assert subs[0]["chat_type"] == "dm"


class TestChatTypeMigration:
    def test_legacy_table_without_chat_type_gets_column(self, kanban_home):
        """A DB whose kanban_notify_subs predates the chat_type column must
        gain it on init, with old rows readable as NULL."""
        conn = kb.connect()
        try:
            # Simulate a legacy schema: drop and recreate without chat_type.
            with kb.write_txn(conn):
                conn.execute("DROP TABLE kanban_notify_subs")
                conn.execute(
                    "CREATE TABLE kanban_notify_subs ("
                    " task_id TEXT NOT NULL, platform TEXT NOT NULL,"
                    " chat_id TEXT NOT NULL, thread_id TEXT NOT NULL DEFAULT '',"
                    " user_id TEXT, notifier_profile TEXT,"
                    " created_at INTEGER NOT NULL,"
                    " last_event_id INTEGER NOT NULL DEFAULT 0,"
                    " PRIMARY KEY (task_id, platform, chat_id, thread_id))"
                )
                conn.execute(
                    "INSERT INTO kanban_notify_subs"
                    " (task_id, platform, chat_id, created_at)"
                    " VALUES ('t_legacy', 'telegram', 'c1', 0)"
                )
            cols_before = {
                r["name"] for r in conn.execute("PRAGMA table_info(kanban_notify_subs)")
            }
            assert "chat_type" not in cols_before
        finally:
            conn.close()

        # Re-init runs the additive migration.
        kb.init_db()

        conn = kb.connect()
        try:
            cols_after = {
                r["name"] for r in conn.execute("PRAGMA table_info(kanban_notify_subs)")
            }
            assert "chat_type" in cols_after
            legacy = kb.list_notify_subs(conn, "t_legacy")
            assert legacy and legacy[0]["chat_type"] is None
        finally:
            conn.close()


# ── Notifier wake: rebuild the correct session scope ──────────────────────


async def _run_notifier_capturing_wake(chat_type):
    """Create a completed, subscribed task with the given sub chat_type, drive
    one notifier tick, and return the SessionSource the wake injected (or None
    if handle_message was never called)."""
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="dm-created task", assignee="w1",
            session_id="sess-abc",
        )
        kwargs = {"task_id": tid, "platform": "telegram", "chat_id": "c1"}
        if chat_type is not None:
            kwargs["chat_type"] = chat_type
        kb.add_notify_sub(conn, **kwargs)
        kb.complete_task(conn, tid, result="done")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}

    captured = {}
    fake_adapter = MagicMock()
    fake_adapter.send = AsyncMock()

    async def _handle(event):
        captured["source"] = event.source
        runner._running = False

    fake_adapter.handle_message = AsyncMock(side_effect=_handle)
    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1), timeout=10.0,
        )

    return captured.get("source")


class TestWakeSessionScope:
    @pytest.mark.asyncio
    async def test_dm_subscription_wakes_dm_scope(self, kanban_home):
        source = await _run_notifier_capturing_wake(chat_type="dm")
        assert source is not None, "wake should have injected a message event"
        assert source.chat_type == "dm"

    @pytest.mark.asyncio
    async def test_legacy_null_chat_type_falls_back_to_group(self, kanban_home):
        source = await _run_notifier_capturing_wake(chat_type=None)
        assert source is not None
        assert source.chat_type == "group"
