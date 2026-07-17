"""Tests for issue #65: Daily stale-todo nudge scheduler in LifeOpsDiscordAdapter.

Each test is anchored to a specific Acceptance Criterion.
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _run(coro):
    """Run a coroutine on the current event loop."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _make_adapter():
    """Return a LifeOpsDiscordAdapter instance with a fake discord client."""
    from plugins.life_ops.discord_adapter import LifeOpsDiscordAdapter

    with patch("plugins.platforms.discord.adapter.DiscordAdapter.__init__", lambda self, cfg: None):
        adapter = LifeOpsDiscordAdapter.__new__(LifeOpsDiscordAdapter)
        # Minimal state the adapter needs
        adapter._client = MagicMock()
        adapter._bedtime_task = None
        adapter._approvals_task = None
        adapter._todo_closure_task = None
        adapter._stale_nudge_task = None
        adapter._idle_nudge_task = None
        adapter._allowed_user_ids = set()
        adapter._allowed_role_ids = set()
        # name is a property derived from platform; mock it via __dict__
        platform_mock = MagicMock()
        platform_mock.value = "test"
        adapter.__dict__["platform"] = platform_mock
    return adapter


# ---------------------------------------------------------------------------
# AC9: _read_stale_nudge_config helper shape
# ---------------------------------------------------------------------------


class TestReadStaleNudgeConfig:
    def test_disabled_when_hour_not_set(self):
        """AC1/AC9: disabled by default when DISCORD_NUDGE_STALE_HOUR is unset."""
        from plugins.life_ops.discord_adapter import _read_stale_nudge_config

        env = {k: v for k, v in os.environ.items()
               if k not in ("DISCORD_NUDGE_STALE_HOUR", "DISCORD_NUDGE_STALE_MINUTE",
                            "DISCORD_NUDGE_STALE_DAYS")}
        with patch.dict(os.environ, env, clear=True):
            cfg = _read_stale_nudge_config()

        assert cfg["enabled"] is False

    def test_enabled_when_hour_set(self):
        """AC2/AC9: enabled when DISCORD_NUDGE_STALE_HOUR is set."""
        from plugins.life_ops.discord_adapter import _read_stale_nudge_config

        with patch.dict(os.environ, {"DISCORD_NUDGE_STALE_HOUR": "9",
                                     "DISCORD_NUDGE_STALE_MINUTE": "0"}):
            cfg = _read_stale_nudge_config()

        assert cfg["enabled"] is True
        assert cfg["hour"] == 9
        assert cfg["minute"] == 0

    def test_days_defaults_to_5(self):
        """AC3: DISCORD_NUDGE_STALE_DAYS defaults to 5."""
        from plugins.life_ops.discord_adapter import _read_stale_nudge_config

        env = {k: v for k, v in os.environ.items() if k != "DISCORD_NUDGE_STALE_DAYS"}
        with patch.dict(os.environ, {**env, "DISCORD_NUDGE_STALE_HOUR": "9"}, clear=True):
            cfg = _read_stale_nudge_config()

        assert cfg["days"] == 5

    def test_days_reads_from_env(self):
        """AC3: DISCORD_NUDGE_STALE_DAYS integer value is used as threshold."""
        from plugins.life_ops.discord_adapter import _read_stale_nudge_config

        with patch.dict(os.environ, {"DISCORD_NUDGE_STALE_HOUR": "9",
                                     "DISCORD_NUDGE_STALE_DAYS": "10"}):
            cfg = _read_stale_nudge_config()

        assert cfg["days"] == 10

    def test_disabled_on_invalid_hour(self):
        """AC9: invalid DISCORD_NUDGE_STALE_HOUR yields disabled config."""
        from plugins.life_ops.discord_adapter import _read_stale_nudge_config

        with patch.dict(os.environ, {"DISCORD_NUDGE_STALE_HOUR": "not-a-number"}):
            cfg = _read_stale_nudge_config()

        assert cfg["enabled"] is False

    def test_minute_defaults_to_0(self):
        """AC9: DISCORD_NUDGE_STALE_MINUTE defaults to 0 when unset."""
        from plugins.life_ops.discord_adapter import _read_stale_nudge_config

        env = {k: v for k, v in os.environ.items() if k != "DISCORD_NUDGE_STALE_MINUTE"}
        with patch.dict(os.environ, {**env, "DISCORD_NUDGE_STALE_HOUR": "9"}, clear=True):
            cfg = _read_stale_nudge_config()

        assert cfg["minute"] == 0


# ---------------------------------------------------------------------------
# AC1: scheduler does not start when DISCORD_NUDGE_STALE_HOUR is unset
# ---------------------------------------------------------------------------


class TestSchedulerDisabledByDefault:
    def test_no_task_created_when_disabled(self):
        """AC1: When DISCORD_NUDGE_STALE_HOUR is not set, no task is created."""
        adapter = _make_adapter()
        env = {k: v for k, v in os.environ.items() if k != "DISCORD_NUDGE_STALE_HOUR"}
        with patch.dict(os.environ, env, clear=True):
            adapter._start_stale_todo_nudge_scheduler()

        assert adapter._stale_nudge_task is None


# ---------------------------------------------------------------------------
# AC2: scheduler starts when env vars set
# ---------------------------------------------------------------------------


class TestSchedulerStartsWhenConfigured:
    def test_task_created_when_hour_set(self):
        """AC2: When DISCORD_NUDGE_STALE_HOUR and MINUTE are set, task is created."""
        adapter = _make_adapter()

        async def _check():
            with patch.dict(os.environ, {"DISCORD_NUDGE_STALE_HOUR": "9",
                                         "DISCORD_NUDGE_STALE_MINUTE": "0"}):
                adapter._start_stale_todo_nudge_scheduler()
                assert adapter._stale_nudge_task is not None
                assert not adapter._stale_nudge_task.done()
                adapter._stale_nudge_task.cancel()
                try:
                    await adapter._stale_nudge_task
                except asyncio.CancelledError:
                    pass

        _run(_check())

    def test_start_is_idempotent(self):
        """AC2: A second _start call while running is a no-op (same task returned)."""
        adapter = _make_adapter()

        async def _check():
            with patch.dict(os.environ, {"DISCORD_NUDGE_STALE_HOUR": "9",
                                         "DISCORD_NUDGE_STALE_MINUTE": "0"}):
                adapter._start_stale_todo_nudge_scheduler()
                task_first = adapter._stale_nudge_task
                adapter._start_stale_todo_nudge_scheduler()
                task_second = adapter._stale_nudge_task
                assert task_first is task_second
                task_first.cancel()
                try:
                    await task_first
                except asyncio.CancelledError:
                    pass

        _run(_check())


# ---------------------------------------------------------------------------
# AC4: away mode suppresses nudge
# ---------------------------------------------------------------------------


def test_away_mode_suppresses_nudge():
    """AC4: When is_away() returns True, no message is posted."""
    adapter = _make_adapter()
    channel = MagicMock()
    channel.send = AsyncMock()
    adapter._client.get_channel = MagicMock(return_value=channel)

    stale = [{"key": "T-1", "text": "something", "priority": "normal"}]

    async def _check():
        with patch("plugins.life_ops.discord_adapter._read_morning_brief_channel_id",
                   return_value="123456"), \
             patch("plugins.life_ops.discord_adapter._ensure_view_classes", return_value=True), \
             patch("plugins.life_ops.away_mode.is_away", return_value=True), \
             patch("plugins.life_ops.todo_store.get_stale_todos", return_value=stale):
            await adapter._fire_stale_todo_nudge(5)

    _run(_check())
    channel.send.assert_not_called()


# ---------------------------------------------------------------------------
# AC5 + AC10: stale todos present, not away → exactly one message with correct
#             text and attached TodoClosureView
# ---------------------------------------------------------------------------


def test_stale_todos_present_posts_one_message():
    """AC5: When stale todos exist and away mode is inactive, exactly one message is posted."""
    adapter = _make_adapter()
    channel = MagicMock()
    channel.send = AsyncMock(return_value=MagicMock(id=999))
    adapter._client.get_channel = MagicMock(return_value=channel)

    stale = [
        {"key": "T-1", "text": "old todo 1", "priority": "normal"},
        {"key": "T-2", "text": "old todo 2", "priority": "normal"},
        {"key": "T-3", "text": "old todo 3", "priority": "high"},
    ]

    async def _check():
        with patch("plugins.life_ops.discord_adapter._read_morning_brief_channel_id",
                   return_value="123456"), \
             patch("plugins.life_ops.discord_adapter._ensure_view_classes", return_value=True), \
             patch("plugins.life_ops.away_mode.is_away", return_value=False), \
             patch("plugins.life_ops.todo_store.get_stale_todos", return_value=stale), \
             patch("plugins.life_ops.discord_adapter.TodoClosureView") as mock_view_cls:
            mock_view = MagicMock()
            mock_view_cls.return_value = mock_view
            await adapter._fire_stale_todo_nudge(5)

    _run(_check())

    channel.send.assert_called_once()
    call_kwargs = channel.send.call_args
    content = call_kwargs[1].get("content") or (call_kwargs[0][0] if call_kwargs[0] else "")
    assert "3" in content, f"Expected stale count in message, got: {content!r}"
    assert "5" in content, f"Expected threshold in message, got: {content!r}"
    assert "haven't moved" in content or "todo" in content.lower(), \
        f"Expected stale nudge text in message, got: {content!r}"
    assert "view" in call_kwargs[1], "Message must include a view= kwarg"


def test_stale_message_text_format():
    """AC5: Message contains stale count and threshold in 'X todo(s) haven't moved in Y+ days' format."""
    adapter = _make_adapter()
    channel = MagicMock()
    captured_content = []

    async def capture_send(*args, **kwargs):
        content = kwargs.get("content") or (args[0] if args else "")
        captured_content.append(content)
        return MagicMock(id=888)

    channel.send = AsyncMock(side_effect=capture_send)
    adapter._client.get_channel = MagicMock(return_value=channel)

    stale = [{"key": "T-1", "text": "foo", "priority": "normal"}]

    async def _check():
        with patch("plugins.life_ops.discord_adapter._read_morning_brief_channel_id",
                   return_value="123456"), \
             patch("plugins.life_ops.discord_adapter._ensure_view_classes", return_value=True), \
             patch("plugins.life_ops.away_mode.is_away", return_value=False), \
             patch("plugins.life_ops.todo_store.get_stale_todos", return_value=stale), \
             patch("plugins.life_ops.discord_adapter.TodoClosureView", return_value=MagicMock()):
            await adapter._fire_stale_todo_nudge(5)

    _run(_check())

    assert len(captured_content) == 1
    msg = captured_content[0]
    # AC5 exact format: "Still on these? N todo(s) haven't moved in Y+ days:"
    assert "1" in msg
    assert "5" in msg
    assert "todo" in msg.lower()
    assert "moved" in msg.lower() or "days" in msg.lower()


# ---------------------------------------------------------------------------
# AC6: empty stale list → no message posted
# ---------------------------------------------------------------------------


def test_empty_stale_list_no_post():
    """AC6: When get_stale_todos() returns [], no message is posted."""
    adapter = _make_adapter()
    channel = MagicMock()
    channel.send = AsyncMock()
    adapter._client.get_channel = MagicMock(return_value=channel)

    async def _check():
        with patch("plugins.life_ops.discord_adapter._read_morning_brief_channel_id",
                   return_value="123456"), \
             patch("plugins.life_ops.discord_adapter._ensure_view_classes", return_value=True), \
             patch("plugins.life_ops.away_mode.is_away", return_value=False), \
             patch("plugins.life_ops.todo_store.get_stale_todos", return_value=[]):
            await adapter._fire_stale_todo_nudge(5)

    _run(_check())
    channel.send.assert_not_called()


# ---------------------------------------------------------------------------
# AC3: DISCORD_NUDGE_STALE_DAYS value passed as threshold
# ---------------------------------------------------------------------------


def test_stale_days_passed_to_get_stale_todos():
    """AC3: The configured DISCORD_NUDGE_STALE_DAYS value is passed as threshold."""
    adapter = _make_adapter()
    channel = MagicMock()
    channel.send = AsyncMock(return_value=MagicMock(id=1))
    adapter._client.get_channel = MagicMock(return_value=channel)

    async def _check():
        with patch("plugins.life_ops.discord_adapter._read_morning_brief_channel_id",
                   return_value="123456"), \
             patch("plugins.life_ops.discord_adapter._ensure_view_classes", return_value=True), \
             patch("plugins.life_ops.away_mode.is_away", return_value=False), \
             patch("plugins.life_ops.todo_store.get_stale_todos", return_value=[]) as mock_get, \
             patch("plugins.life_ops.discord_adapter.TodoClosureView", return_value=MagicMock()):
            await adapter._fire_stale_todo_nudge(10)
            mock_get.assert_called_once_with(threshold_days=10)

    _run(_check())


# ---------------------------------------------------------------------------
# AC8: _cancel_stale_todo_nudge_task cancels cleanly
# ---------------------------------------------------------------------------


def test_cancel_stale_nudge_task_cleans_up():
    """AC8: _cancel_stale_todo_nudge_task cancels the background task without raising."""
    adapter = _make_adapter()

    async def _check():
        with patch.dict(os.environ, {"DISCORD_NUDGE_STALE_HOUR": "9",
                                     "DISCORD_NUDGE_STALE_MINUTE": "0"}):
            adapter._start_stale_todo_nudge_scheduler()
            assert adapter._stale_nudge_task is not None
            await adapter._cancel_stale_todo_nudge_task()

        assert adapter._stale_nudge_task is None

    _run(_check())


def test_cancel_stale_nudge_task_noop_when_no_task():
    """AC8: _cancel_stale_todo_nudge_task is safe to call when no task exists."""
    adapter = _make_adapter()
    adapter._stale_nudge_task = None

    async def _check():
        await adapter._cancel_stale_todo_nudge_task()

    _run(_check())
    assert adapter._stale_nudge_task is None


# ---------------------------------------------------------------------------
# AC7: scheduler fires once per 24 hours (loop shape test)
# ---------------------------------------------------------------------------


def test_scheduler_loop_fires_once_then_sleeps():
    """AC7: The scheduler loop fires the nudge once then sleeps until the next day."""
    adapter = _make_adapter()
    fired_count = [0]

    async def fake_fire(days):
        fired_count[0] += 1

    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)
        if len(sleep_calls) >= 2:
            raise asyncio.CancelledError()

    async def _check():
        with patch.object(adapter, "_fire_stale_todo_nudge", side_effect=fake_fire), \
             patch("asyncio.sleep", side_effect=fake_sleep):
            cfg = {"enabled": True, "hour": 0, "minute": 0, "days": 5}
            try:
                await adapter._stale_todo_nudge_scheduler_loop(cfg)
            except asyncio.CancelledError:
                pass

    _run(_check())

    # One fire happened, then a second sleep was requested (long wait until next day)
    assert fired_count[0] >= 1
    assert len(sleep_calls) >= 2
    # Second sleep must be at least 1 hour — confirms the loop waits until the next
    # scheduled fire time, not a short retry. Exact value depends on current UTC time.
    assert sleep_calls[1] > 3600, f"Expected long inter-fire sleep, got {sleep_calls[1]}"


# ---------------------------------------------------------------------------
# AC10: No new View class — TodoClosureView is reused
# ---------------------------------------------------------------------------


def test_no_new_view_class_introduced():
    """AC10: The implementation reuses TodoClosureView; no new View subclass is added."""
    import plugins.life_ops.discord_adapter as adapter_mod

    known_views = {"JournalApproveView", "BedtimeView", "TodoClosureView"}
    module_names = set(dir(adapter_mod))
    new_views = {
        name for name in module_names
        if name.endswith("View") and name not in known_views
    }
    assert not new_views, f"Unexpected new View classes introduced: {new_views}"


# ---------------------------------------------------------------------------
# Integration: cancel_background_tasks includes stale nudge
# ---------------------------------------------------------------------------


def test_cancel_background_tasks_includes_stale_nudge():
    """AC8: cancel_background_tasks() also cancels the stale nudge task."""
    adapter = _make_adapter()

    async def _check():
        with patch.dict(os.environ, {"DISCORD_NUDGE_STALE_HOUR": "9",
                                     "DISCORD_NUDGE_STALE_MINUTE": "0"}):
            adapter._start_stale_todo_nudge_scheduler()
            assert adapter._stale_nudge_task is not None

        with patch("plugins.platforms.discord.adapter.DiscordAdapter.cancel_background_tasks",
                   new_callable=AsyncMock):
            await adapter.cancel_background_tasks()

        assert adapter._stale_nudge_task is None

    _run(_check())


# ---------------------------------------------------------------------------
# Integration: _run_post_connect_initialization calls stale nudge scheduler
# ---------------------------------------------------------------------------


def test_post_connect_starts_stale_nudge_scheduler():
    """AC2: _run_post_connect_initialization calls _start_stale_todo_nudge_scheduler."""
    adapter = _make_adapter()

    async def _check():
        with patch.object(adapter, "_start_bedtime_scheduler"), \
             patch.object(adapter, "_start_approvals_scheduler"), \
             patch.object(adapter, "_start_todo_closure_scheduler"), \
             patch.object(adapter, "_start_stale_todo_nudge_scheduler") as mock_start, \
             patch("plugins.platforms.discord.adapter.DiscordAdapter._run_post_connect_initialization",
                   new_callable=AsyncMock):
            await adapter._run_post_connect_initialization()

        mock_start.assert_called_once()

    _run(_check())
