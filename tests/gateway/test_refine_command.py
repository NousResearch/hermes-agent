"""Behavior tests for the gateway /refine command."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.slash_commands import GatewaySlashCommandsMixin


@pytest.mark.asyncio
@pytest.mark.parametrize("args", ["", "review the deploy workflow"])
async def test_refine_marks_bare_and_focused_requests_explicit(args):
    agent = SimpleNamespace(
        _session_messages=[{"role": "user", "content": "hello"}],
        valid_tool_names={"skill_manage"},
        _spawn_background_review=MagicMock(),
    )
    host = SimpleNamespace(
        _agent_cache={"session-key": (agent, 0.0)},
        _agent_cache_lock=threading.Lock(),
        _running_agents={},
        _session_key_for_source=lambda _source: "session-key",
    )
    event = SimpleNamespace(
        source=object(),
        get_command_args=lambda: args,
    )

    response = await GatewaySlashCommandsMixin._handle_refine_command(host, event)

    assert response.startswith("⚗ Reviewing this conversation in the background")
    agent._spawn_background_review.assert_called_once_with(
        messages_snapshot=agent._session_messages,
        review_memory=True,
        review_skills=True,
        focus=args or None,
        explicit=True,
    )