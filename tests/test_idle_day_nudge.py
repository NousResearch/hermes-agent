"""Tests for issue #66: Add idle-day nudge scheduler to Discord adapter.

Each test is anchored to a specific Acceptance Criterion from the issue.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _run(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter():
    from plugins.life_ops.discord_adapter import LifeOpsDiscordAdapter

    with patch("plugins.platforms.discord.adapter.DiscordAdapter.__init__", lambda self, cfg: None):
        adapter = LifeOpsDiscordAdapter.__new__(LifeOpsDiscordAdapter)
        adapter._client = MagicMock()
        adapter._bedtime_task = None
        adapter._approvals_task = None
        adapter._todo_closure_task = None
        adapter._stale_nudge_task = None
        adapter._idle_nudge_task = None
        adapter._allowed_user_ids = set()
        adapter._allowed_role_ids = set()
        platform_mock = MagicMock()
        platform_mock.value = "test"
        adapter.__dict__["platform"] = platform_mock
    return adapter


def _make_db_with_closures(closed_at_values: list[str]) -> Path:
    """Create a temp SQLite todos.db with closure records at the given ISO timestamps."""
    tmp = tempfile.mkdtemp()
    db_path = Path(tmp) / "todos.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE todos (
            key TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            category TEXT,
            priority TEXT,
            source_dates TEXT,
            recurring INTEGER NOT NULL DEFAULT 0,
            confidence REAL,
            status TEXT NOT NULL DEFAULT 'open',
            snoozed_until TEXT,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            closed_at TEXT,
            origin TEXT NOT NULL DEFAULT 'journal'
        )
    """)
    today = "2026-07-17"
    for i, closed_at in enumerate(closed_at_values):
        status = "done" if closed_at else "open"
        conn.execute(
            "INSERT INTO todos VALUES (?,?,NULL,NULL,NULL,0,NULL,?,NULL,?,?,?,?)",
            (f"key-{i}", f"todo {i}", status, today, today, closed_at, "journal"),
        )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# AC1+AC2: count_todos_closed_today() in todo_store
# ---------------------------------------------------------------------------


class TestCountTodosClosedToday:
    def test_returns_zero_on_fresh_db(self):
        """AC1: Returns 0 when no todos have been closed today."""
        from plugins.life_ops import todo_store

        db_path = _make_db_with_closures([])
        with patch.object(todo_store, "todos_db_path", return_value=db_path):
            result = todo_store.count_todos_closed_today()
        assert result == 0

    def test_returns_correct_count_after_closures(self):
        """AC2: Returns the correct positive integer after closures today."""
        import datetime

        from plugins.life_ops import todo_store

        today_iso = datetime.date.today().isoformat()
        closed_timestamps = [
            f"{today_iso}T10:00:00+00:00",
            f"{today_iso}T12:00:00+00:00",
        ]
        db_path = _make_db_with_closures(closed_timestamps)
        with patch.object(todo_store, "todos_db_path", return_value=db_path):
            result = todo_store.count_todos_closed_today()
        assert result == 2

    def test_returns_zero_for_past_closures(self):
        """AC2: Returns 0 when closure records exist but with a past date."""
        from plugins.life_ops import todo_store

        past_timestamps = [
            "2026-07-16T10:00:00+00:00",
            "2026-07-01T08:00:00+00:00",
        ]
        db_path = _make_db_with_closures(past_timestamps)
        with patch.object(todo_store, "todos_db_path", return_value=db_path):
            result = todo_store.count_todos_closed_today()
        assert result == 0

    def test_does_not_count_open_todos(self):
        """AC2: Open todos (closed_at=NULL) are not counted."""
        from plugins.life_ops import todo_store

        db_path = _make_db_with_closures([None, None, None])
        with patch.object(todo_store, "todos_db_path", return_value=db_path):
            result = todo_store.count_todos_closed_today()
        assert result == 0

    def test_does_not_alter_close_todo_signature(self):
        """AC8: close_todo() signature is unchanged."""
        import inspect

        from plugins.life_ops import todo_store

        sig = inspect.signature(todo_store.close_todo)
        params = list(sig.parameters.keys())
        assert params == ["key", "action", "source", "snooze_until"]

    def test_no_discord_dependency(self):
        """AC1: count_todos_closed_today() importable without discord."""
        import sys
        saved = sys.modules.pop("discord", None)
        try:
            from plugins.life_ops import todo_store
            result = todo_store.count_todos_closed_today
            assert callable(result)
        finally:
            if saved is not None:
                sys.modules["discord"] = saved


# ---------------------------------------------------------------------------
# AC3: env vars control fire time; missing either → scheduler disabled
# ---------------------------------------------------------------------------


class TestReadIdleNudgeConfig:
    def test_disabled_when_hour_not_set(self):
        """AC3: Scheduler does not start when DISCORD_NUDGE_IDLE_HOUR is unset."""
        from plugins.life_ops.discord_adapter import _read_idle_nudge_config

        env = {k: v for k, v in os.environ.items()
               if k not in ("DISCORD_NUDGE_IDLE_HOUR", "DISCORD_NUDGE_IDLE_MINUTE")}
        with patch.dict(os.environ, env, clear=True):
            cfg = _read_idle_nudge_config()
        assert cfg["enabled"] is False

    def test_disabled_when_minute_not_set(self):
        """AC3: Scheduler does not start when DISCORD_NUDGE_IDLE_MINUTE is unset."""
        from plugins.life_ops.discord_adapter import _read_idle_nudge_config

        env = {k: v for k, v in os.environ.items()
               if k not in ("DISCORD_NUDGE_IDLE_HOUR", "DISCORD_NUDGE_IDLE_MINUTE")}
        env["DISCORD_NUDGE_IDLE_HOUR"] = "9"
        with patch.dict(os.environ, env, clear=True):
            cfg = _read_idle_nudge_config()
        assert cfg["enabled"] is False

    def test_enabled_when_both_set(self):
        """AC3: Enabled when both DISCORD_NUDGE_IDLE_HOUR and MINUTE are set."""
        from plugins.life_ops.discord_adapter import _read_idle_nudge_config

        with patch.dict(os.environ, {
            "DISCORD_NUDGE_IDLE_HOUR": "21",
            "DISCORD_NUDGE_IDLE_MINUTE": "30",
        }):
            cfg = _read_idle_nudge_config()
        assert cfg["enabled"] is True
        assert cfg["hour"] == 21
        assert cfg["minute"] == 30

    def test_disabled_on_invalid_hour(self):
        """AC3: Invalid DISCORD_NUDGE_IDLE_HOUR yields disabled config."""
        from plugins.life_ops.discord_adapter import _read_idle_nudge_config

        with patch.dict(os.environ, {
            "DISCORD_NUDGE_IDLE_HOUR": "not-a-number",
            "DISCORD_NUDGE_IDLE_MINUTE": "0",
        }):
            cfg = _read_idle_nudge_config()
        assert cfg["enabled"] is False


# ---------------------------------------------------------------------------
# AC4: Scheduler registered in _run_post_connect_initialization and cancel
# ---------------------------------------------------------------------------


class TestLifecycleRegistration:
    def test_post_connect_starts_idle_nudge_scheduler(self):
        """AC4: _run_post_connect_initialization calls _start_idle_day_nudge_scheduler."""
        adapter = _make_adapter()

        async def _check():
            with patch.object(adapter, "_start_bedtime_scheduler"), \
                 patch.object(adapter, "_start_approvals_scheduler"), \
                 patch.object(adapter, "_start_todo_closure_scheduler"), \
                 patch.object(adapter, "_start_stale_todo_nudge_scheduler"), \
                 patch.object(adapter, "_start_idle_day_nudge_scheduler") as mock_start, \
                 patch("plugins.platforms.discord.adapter.DiscordAdapter._run_post_connect_initialization",
                       new_callable=AsyncMock):
                await adapter._run_post_connect_initialization()
            mock_start.assert_called_once()

        _run(_check())

    def test_cancel_background_tasks_includes_idle_nudge(self):
        """AC4: cancel_background_tasks() also cancels the idle-day nudge task."""
        adapter = _make_adapter()

        async def _check():
            with patch.dict(os.environ, {
                "DISCORD_NUDGE_IDLE_HOUR": "21",
                "DISCORD_NUDGE_IDLE_MINUTE": "30",
            }):
                adapter._start_idle_day_nudge_scheduler()
                assert adapter._idle_nudge_task is not None

            with patch("plugins.platforms.discord.adapter.DiscordAdapter.cancel_background_tasks",
                       new_callable=AsyncMock):
                await adapter.cancel_background_tasks()

            assert adapter._idle_nudge_task is None

        _run(_check())


# ---------------------------------------------------------------------------
# AC5: Nudge fires only when all three conditions hold
# ---------------------------------------------------------------------------


class TestIdleNudgeFireConditions:
    def _setup_channel(self, adapter):
        channel = MagicMock()
        channel.send = AsyncMock(return_value=MagicMock(id=42))
        adapter._client.get_channel = MagicMock(return_value=channel)
        return channel

    def test_fires_when_all_conditions_met(self):
        """AC5: Posts when away=False, closed_today=0, open_todos non-empty."""
        adapter = _make_adapter()
        channel = self._setup_channel(adapter)
        open_todos = [{"key": "T-1", "text": "do something", "priority": "normal"}]

        async def _check():
            with patch("plugins.life_ops.discord_adapter._read_morning_brief_channel_id",
                       return_value="123456"), \
                 patch("plugins.life_ops.discord_adapter._ensure_view_classes", return_value=True), \
                 patch("plugins.life_ops.away_mode.is_away", return_value=False), \
                 patch("plugins.life_ops.todo_store.count_todos_closed_today", return_value=0), \
                 patch("plugins.life_ops.todo_store.get_open_todos", return_value=open_todos), \
                 patch("plugins.life_ops.discord_adapter.TodoClosureView", return_value=MagicMock()):
                await adapter._fire_idle_day_nudge()

        _run(_check())
        channel.send.assert_called_once()

    def test_skips_when_away_mode_active(self):
        """AC5+AC7: No message posted when away mode is active."""
        adapter = _make_adapter()
        channel = self._setup_channel(adapter)
        open_todos = [{"key": "T-1", "text": "do something", "priority": "normal"}]

        async def _check():
            with patch("plugins.life_ops.discord_adapter._read_morning_brief_channel_id",
                       return_value="123456"), \
                 patch("plugins.life_ops.discord_adapter._ensure_view_classes", return_value=True), \
                 patch("plugins.life_ops.away_mode.is_away", return_value=True), \
                 patch("plugins.life_ops.todo_store.count_todos_closed_today", return_value=0), \
                 patch("plugins.life_ops.todo_store.get_open_todos", return_value=open_todos):
                await adapter._fire_idle_day_nudge()

        _run(_check())
        channel.send.assert_not_called()

    def test_skips_when_todo_closed_today(self):
        """AC5+AC7: No message posted when at least one todo was closed today."""
        adapter = _make_adapter()
        channel = self._setup_channel(adapter)
        open_todos = [{"key": "T-1", "text": "do something", "priority": "normal"}]

        async def _check():
            with patch("plugins.life_ops.discord_adapter._read_morning_brief_channel_id",
                       return_value="123456"), \
                 patch("plugins.life_ops.discord_adapter._ensure_view_classes", return_value=True), \
                 patch("plugins.life_ops.away_mode.is_away", return_value=False), \
                 patch("plugins.life_ops.todo_store.count_todos_closed_today", return_value=1), \
                 patch("plugins.life_ops.todo_store.get_open_todos", return_value=open_todos):
                await adapter._fire_idle_day_nudge()

        _run(_check())
        channel.send.assert_not_called()

    def test_skips_when_open_todo_list_empty(self):
        """AC5+AC7: No message posted when open todo list is empty."""
        adapter = _make_adapter()
        channel = self._setup_channel(adapter)

        async def _check():
            with patch("plugins.life_ops.discord_adapter._read_morning_brief_channel_id",
                       return_value="123456"), \
                 patch("plugins.life_ops.discord_adapter._ensure_view_classes", return_value=True), \
                 patch("plugins.life_ops.away_mode.is_away", return_value=False), \
                 patch("plugins.life_ops.todo_store.count_todos_closed_today", return_value=0), \
                 patch("plugins.life_ops.todo_store.get_open_todos", return_value=[]):
                await adapter._fire_idle_day_nudge()

        _run(_check())
        channel.send.assert_not_called()


# ---------------------------------------------------------------------------
# AC6: Nudge message text and view
# ---------------------------------------------------------------------------


class TestIdleNudgeMessageContent:
    def test_message_text_exact(self):
        """AC6: Nudge message is exactly "Haven't touched your list today — want to review it?"."""
        adapter = _make_adapter()
        channel = MagicMock()
        captured = []

        async def capture_send(*args, **kwargs):
            content = kwargs.get("content") or (args[0] if args else "")
            captured.append({"content": content, "kwargs": kwargs})
            return MagicMock(id=1)

        channel.send = AsyncMock(side_effect=capture_send)
        adapter._client.get_channel = MagicMock(return_value=channel)
        open_todos = [{"key": "T-1", "text": "do something", "priority": "normal"}]

        async def _check():
            with patch("plugins.life_ops.discord_adapter._read_morning_brief_channel_id",
                       return_value="123456"), \
                 patch("plugins.life_ops.discord_adapter._ensure_view_classes", return_value=True), \
                 patch("plugins.life_ops.away_mode.is_away", return_value=False), \
                 patch("plugins.life_ops.todo_store.count_todos_closed_today", return_value=0), \
                 patch("plugins.life_ops.todo_store.get_open_todos", return_value=open_todos), \
                 patch("plugins.life_ops.discord_adapter.TodoClosureView", return_value=MagicMock()):
                await adapter._fire_idle_day_nudge()

        _run(_check())

        assert len(captured) == 1
        assert captured[0]["content"] == "Haven't touched your list today — want to review it?"

    def test_message_includes_todo_closure_view(self):
        """AC6: Message is accompanied by a TodoClosureView."""
        adapter = _make_adapter()
        channel = MagicMock()
        channel.send = AsyncMock(return_value=MagicMock(id=1))
        adapter._client.get_channel = MagicMock(return_value=channel)
        open_todos = [{"key": "T-1", "text": "do something", "priority": "normal"}]

        async def _check():
            with patch("plugins.life_ops.discord_adapter._read_morning_brief_channel_id",
                       return_value="123456"), \
                 patch("plugins.life_ops.discord_adapter._ensure_view_classes", return_value=True), \
                 patch("plugins.life_ops.away_mode.is_away", return_value=False), \
                 patch("plugins.life_ops.todo_store.count_todos_closed_today", return_value=0), \
                 patch("plugins.life_ops.todo_store.get_open_todos", return_value=open_todos), \
                 patch("plugins.life_ops.discord_adapter.TodoClosureView") as mock_view_cls:
                mock_view_cls.return_value = MagicMock()
                await adapter._fire_idle_day_nudge()

        _run(_check())

        channel.send.assert_called_once()
        call_kwargs = channel.send.call_args[1]
        assert "view" in call_kwargs, "Message must include view= kwarg"


# ---------------------------------------------------------------------------
# AC3: no-start when env vars absent — scheduler task
# ---------------------------------------------------------------------------


class TestSchedulerDisabledByDefault:
    def test_no_task_created_when_both_unset(self):
        """AC3: No task created when DISCORD_NUDGE_IDLE_HOUR and MINUTE are both unset."""
        adapter = _make_adapter()
        env = {k: v for k, v in os.environ.items()
               if k not in ("DISCORD_NUDGE_IDLE_HOUR", "DISCORD_NUDGE_IDLE_MINUTE")}
        with patch.dict(os.environ, env, clear=True):
            adapter._start_idle_day_nudge_scheduler()
        assert adapter._idle_nudge_task is None

    def test_task_created_when_both_set(self):
        """AC3: Task is created when both env vars are set."""
        adapter = _make_adapter()

        async def _check():
            with patch.dict(os.environ, {
                "DISCORD_NUDGE_IDLE_HOUR": "21",
                "DISCORD_NUDGE_IDLE_MINUTE": "30",
            }):
                adapter._start_idle_day_nudge_scheduler()
                assert adapter._idle_nudge_task is not None
                assert not adapter._idle_nudge_task.done()
                adapter._idle_nudge_task.cancel()
                try:
                    await adapter._idle_nudge_task
                except asyncio.CancelledError:
                    pass

        _run(_check())

    def test_start_is_idempotent(self):
        """AC3: A second _start call while running is a no-op (same task)."""
        adapter = _make_adapter()

        async def _check():
            with patch.dict(os.environ, {
                "DISCORD_NUDGE_IDLE_HOUR": "21",
                "DISCORD_NUDGE_IDLE_MINUTE": "30",
            }):
                adapter._start_idle_day_nudge_scheduler()
                first = adapter._idle_nudge_task
                adapter._start_idle_day_nudge_scheduler()
                second = adapter._idle_nudge_task
                assert first is second
                first.cancel()
                try:
                    await first
                except asyncio.CancelledError:
                    pass

        _run(_check())


# ---------------------------------------------------------------------------
# Cancel task
# ---------------------------------------------------------------------------


class TestCancelIdleNudgeTask:
    def test_cancel_cleans_up(self):
        """AC4: _cancel_idle_day_nudge_task cancels without raising."""
        adapter = _make_adapter()

        async def _check():
            with patch.dict(os.environ, {
                "DISCORD_NUDGE_IDLE_HOUR": "21",
                "DISCORD_NUDGE_IDLE_MINUTE": "30",
            }):
                adapter._start_idle_day_nudge_scheduler()
            await adapter._cancel_idle_day_nudge_task()
            assert adapter._idle_nudge_task is None

        _run(_check())

    def test_cancel_noop_when_no_task(self):
        """AC4: _cancel_idle_day_nudge_task safe to call when no task exists."""
        adapter = _make_adapter()
        adapter._idle_nudge_task = None

        async def _check():
            await adapter._cancel_idle_day_nudge_task()

        _run(_check())
        assert adapter._idle_nudge_task is None


# ---------------------------------------------------------------------------
# Scheduler loop fires once then sleeps
# ---------------------------------------------------------------------------


def test_scheduler_loop_fires_once_then_sleeps():
    """AC4: The loop fires the nudge once then sleeps until the next day."""
    adapter = _make_adapter()
    fired_count = [0]

    async def fake_fire():
        fired_count[0] += 1

    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)
        if len(sleep_calls) >= 2:
            raise asyncio.CancelledError()

    async def _check():
        with patch.object(adapter, "_fire_idle_day_nudge", side_effect=fake_fire), \
             patch("asyncio.sleep", side_effect=fake_sleep):
            cfg = {"enabled": True, "hour": 0, "minute": 0}
            try:
                await adapter._idle_day_nudge_scheduler_loop(cfg)
            except asyncio.CancelledError:
                pass

    _run(_check())

    assert fired_count[0] >= 1
    assert len(sleep_calls) >= 2
    assert sleep_calls[1] > 3600, f"Expected long inter-fire sleep, got {sleep_calls[1]}"
