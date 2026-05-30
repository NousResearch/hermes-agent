"""Tests for /background slash dispatch on the API server."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def api_server():
    from gateway.platforms.api_server import APIServerAdapter

    server = object.__new__(APIServerAdapter)
    server._active_run_agents = {}
    server._background_tasks = set()
    server._persist_subagent_event = MagicMock(side_effect=lambda event, session_id=None: event)
    return server


def test_background_without_prompt_returns_usage(api_server):
    result = api_server._dispatch_slash_command("/background", "session-1")
    assert result["type"] == "error"
    assert "usage" in result["message"].lower()


def test_background_starts_task_and_returns_confirmation(api_server):
    created = []

    def capture_task(coro):
        created.append(coro)
        coro.close()
        return MagicMock()

    with patch("gateway.platforms.api_server.asyncio.get_running_loop") as get_loop:
        loop = MagicMock()
        loop.create_task.side_effect = capture_task
        get_loop.return_value = loop

        result = api_server._dispatch_slash_command(
            "/background summarize the repo",
            "session-abc",
        )

    assert result["type"] == "text"
    assert "Background task started" in result["content"]
    assert "bg_" in result["content"]
    assert len(created) == 1
    api_server._persist_subagent_event.assert_called_once()
    start_event = api_server._persist_subagent_event.call_args.args[0]
    assert start_event["event"] == "subagent.start"
    assert start_event["runtime"] == "background"


@pytest.mark.asyncio
async def test_run_api_background_task_persists_completion(api_server):
    api_server._run_agent = AsyncMock(
        return_value=({"final_response": "done"}, {"input_tokens": 1, "output_tokens": 2})
    )

    await api_server._run_api_background_task(
        parent_session_id="session-abc",
        prompt="check tests",
        task_id="bg_test123",
    )

    assert api_server._persist_subagent_event.call_count == 1
    complete_event = api_server._persist_subagent_event.call_args.args[0]
    assert complete_event["event"] == "subagent.complete"
    assert complete_event["status"] == "completed"
    assert complete_event["summary"] == "done"
