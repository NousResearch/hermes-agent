"""Gateway tests for the /task command."""

from __future__ import annotations

import json

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _Source(SessionSource):
    pass


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", thread_id="7"),
        message_id="42",
    )


@pytest.mark.asyncio
async def test_task_command_renders_index(monkeypatch, tmp_path):
    path = tmp_path / "todo-state.json"
    path.write_text(
        json.dumps(
            {
                "updated_at": "2026-05-11T10:00:00+08:00",
                "active": [{"id": "p1", "title": "Main task", "status": "active"}],
                "pending": [],
                "resolved_recent": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("task_tree.get_task_tree_path", lambda: path)
    runner = object.__new__(GatewayRunner)

    result = await runner._handle_task_command(_make_event("/task"))

    assert "Tasks" in result
    assert "Main task" in result


@pytest.mark.asyncio
async def test_task_command_renders_query(monkeypatch, tmp_path):
    path = tmp_path / "todo-state.json"
    path.write_text(
        json.dumps(
            {
                "updated_at": "2026-05-11T10:00:00+08:00",
                "active": [
                    {
                        "id": "p1",
                        "title": "Main task",
                        "status": "active",
                        "subtasks": [
                            {"id": "s1", "title": "Sub task", "status": "pending"}
                        ],
                    }
                ],
                "pending": [],
                "resolved_recent": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("task_tree.get_task_tree_path", lambda: path)
    runner = object.__new__(GatewayRunner)

    result = await runner._handle_task_command(_make_event("/task main"))

    assert "Task 1" in result
    assert "Sub task" in result


@pytest.mark.asyncio
async def test_task_command_uses_platform_inline_sender_when_available(monkeypatch, tmp_path):
    path = tmp_path / "todo-state.json"
    path.write_text(
        json.dumps(
            {
                "updated_at": "2026-05-11T10:00:00+08:00",
                "active": [{"id": "p1", "title": "Main task", "status": "active"}],
                "pending": [],
                "resolved_recent": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("task_tree.get_task_tree_path", lambda: path)
    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(send_task_browser_view=AsyncMock(return_value=SimpleNamespace(success=True)))
    runner.adapters = {Platform.TELEGRAM: adapter}

    result = await runner._handle_task_command(_make_event("/task"))

    assert result == ""
    adapter.send_task_browser_view.assert_awaited_once()
    _, kwargs = adapter.send_task_browser_view.call_args
    assert kwargs["reply_to"] == "42"
    assert kwargs["thread_id"] == "7"
