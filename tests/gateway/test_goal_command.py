import asyncio
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/goal", platform=Platform.SLACK, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._background_tasks = set()
    return runner


class TestHandleGoalCommand:
    @pytest.mark.asyncio
    async def test_no_prompt_shows_usage(self):
        runner = _make_runner()
        result = await runner._handle_goal_command(_make_event("/goal"))
        assert "Usage:" in result
        assert "/goal" in result

    @pytest.mark.asyncio
    async def test_valid_prompt_starts_goal_task(self, monkeypatch):
        runner = _make_runner()
        monkeypatch.setenv("HERMES_GOAL_MAX_LOOPS", "2")
        created_tasks = []

        def capture_task(coro, *args, **kwargs):
            coro.close()
            mock_task = MagicMock()
            created_tasks.append(mock_task)
            return mock_task

        with patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            result = await runner._handle_goal_command(_make_event("/goal finish AutomateSTIG batch"))

        assert "🎯" in result
        assert "Goal loop started" in result
        assert "goal_" in result
        assert "Max supervisor loops: 2" in result
        assert "finish AutomateSTIG batch" in result
        assert len(created_tasks) == 1

    @pytest.mark.asyncio
    async def test_long_prompt_is_truncated(self):
        runner = _make_runner()
        long_prompt = "A" * 100
        with patch("gateway.run.asyncio.create_task", side_effect=lambda c, **kw: (c.close(), MagicMock())[1]):
            result = await runner._handle_goal_command(_make_event(f"/goal {long_prompt}"))
        assert "..." in result
        assert long_prompt not in result
