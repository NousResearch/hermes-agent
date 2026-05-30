"""Tests for POST /v1/runs/{run_id}/steer on the API server."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def api_server():
    from gateway.platforms.api_server import APIServerAdapter

    server = object.__new__(APIServerAdapter)
    server._active_run_agents = {}
    server._check_auth = lambda _request: None
    return server


@pytest.mark.asyncio
async def test_steer_run_calls_agent_steer(api_server):
    agent = MagicMock()
    agent.steer.return_value = True
    api_server._active_run_agents["chatcmpl-test123"] = agent

    request = MagicMock()
    request.match_info = {"run_id": "chatcmpl-test123"}
    request.json = AsyncMock(return_value={"text": "focus on tests"})

    response = await api_server._handle_steer_run(request)
    payload = json.loads(response.body.decode())

    assert response.status == 200
    assert payload["status"] == "queued"
    assert payload["run_id"] == "chatcmpl-test123"
    agent.steer.assert_called_once_with("focus on tests")


@pytest.mark.asyncio
async def test_steer_run_missing_agent_returns_404(api_server):
    request = MagicMock()
    request.match_info = {"run_id": "chatcmpl-missing"}
    request.json = AsyncMock(return_value={"text": "hello"})

    response = await api_server._handle_steer_run(request)
    assert response.status == 404
