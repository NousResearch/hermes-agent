"""Regression: api_server agent entry never registered MCP tools.

``/v1/runs`` and ``_run_agent`` construct an ``AIAgent`` directly, and an agent freezes its
tool registry at construction. Every other agent entry point registers MCP servers before
that happens — ``start_gateway()`` at gateway startup, cron's ``run_job()`` (#4219), the
CLI/TUI/ACP surfaces via ``hermes_cli.mcp_startup`` — and each of them covers exactly ONE
profile: the one its process launched on.

A multiplexing gateway builds agents for ``/p/<profile>/`` requests whose config it never
read, so those profiles' ``mcp_servers`` were never dialled and the model was handed built-in
tools only, however healthy the servers were. Both api_server agent-entry paths now call
``_ensure_mcp_tools_registered`` under the request's own profile scope first.
"""

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


def _runs_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


@pytest.mark.asyncio
async def test_run_registers_mcp_tools_before_building_the_agent(adapter):
    order = []

    def _discover():
        order.append("discover")
        return []

    def _create_agent(**kwargs):
        order.append("create_agent")
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "done"}
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0
        return agent

    app = _runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch("tools.mcp_tool_discovery.discover_mcp_tools", side_effect=_discover), \
             patch.object(adapter, "_create_agent", side_effect=_create_agent):
            resp = await cli.post("/v1/runs", json={"input": "hello"})
            assert resp.status == 202
            run_id = (await resp.json())["run_id"]

            for _ in range(300):
                status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                if status["status"] in {"completed", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.01)

    assert order[:2] == ["discover", "create_agent"], (
        "MCP discovery must run before the agent freezes its tool registry; "
        f"observed {order}"
    )


def test_discovery_runs_inside_the_requested_profile_scope(adapter):
    """A multiplexed run must discover ITS profile, not the one the process launched on."""
    scoped = []
    seen_while_scoped = []

    @contextmanager
    def _fake_scope(profile):
        scoped.append(profile)
        try:
            yield
        finally:
            scoped.pop()

    with patch.object(adapter, "_profile_scope", _fake_scope), \
         patch("tools.mcp_tool_discovery.discover_mcp_tools",
               side_effect=lambda: seen_while_scoped.append(scoped[-1])):
        adapter._ensure_mcp_tools_registered("worker")

    assert seen_while_scoped == ["worker"]
    assert scoped == []


def test_a_broken_mcp_server_does_not_fail_the_run(adapter):
    """Same contract cron gained in #4219: MCP failure is never fatal to the turn."""
    with patch("tools.mcp_tool_discovery.discover_mcp_tools",
               side_effect=RuntimeError("server refused the handshake")):
        adapter._ensure_mcp_tools_registered(None)
