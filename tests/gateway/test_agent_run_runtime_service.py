"""Tests for agent_run_runtime_service thin entry (proxy short-circuit)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.agent_run_runtime_service import run_gateway_agent_inner


@pytest.mark.asyncio
async def test_run_gateway_agent_inner_proxy_short_circuit():
    proxy_result = {"final_response": "via-proxy", "completed": True}
    runner = SimpleNamespace(
        _get_proxy_url=MagicMock(return_value="http://proxy.example"),
        _run_agent_via_proxy=AsyncMock(return_value=proxy_result),
    )
    source = SimpleNamespace(platform="telegram")
    out = await run_gateway_agent_inner(
        runner=runner,
        message="hi",
        context_prompt="",
        history=[],
        source=source,
        session_id="sid",
        session_key="sk",
        run_generation=1,
        event_message_id="mid",
        logger=MagicMock(),
    )
    assert out is proxy_result
    runner._run_agent_via_proxy.assert_awaited_once()


@pytest.mark.asyncio
async def test_runner_inner_delegates_to_service(monkeypatch):
    """GatewayRunner._run_agent_inner stays a thin delegate."""
    from gateway.run import GatewayRunner

    called = {}

    async def fake_run(**kwargs):
        called.update(kwargs)
        return {"final_response": "ok", "completed": True}

    monkeypatch.setattr(
        "gateway.run.run_gateway_agent_inner",
        fake_run,
    )
    runner = GatewayRunner.__new__(GatewayRunner)
    source = SimpleNamespace(platform="qq")
    result = await GatewayRunner._run_agent_inner(
        runner,
        "msg",
        "ctx",
        [],
        source,
        "sid",
        session_key="sk",
    )
    assert result["final_response"] == "ok"
    assert called["runner"] is runner
    assert called["message"] == "msg"
    assert called["session_id"] == "sid"
