"""ACP MCP tools must be directly visible in agent tool definitions (#101289).

Spins a REAL local Streamable-HTTP MCP server (lowlevel ``Server`` +
uvicorn on 127.0.0.1 — no external network) exposing one tool
``probe_tool`` and drives both MCP configuration paths with REAL
discovery / registry / tool-defs (only the LLM provider resolution is
stubbed so no credentials are needed):

- config ``mcp_servers.probe`` -> ``SessionManager.create_session``
- ``session/new`` ``mcpServers`` -> ``HermesACPAgent.new_session``

Both must yield ``mcp__probe__probe_tool`` as a DIRECT entry of
``agent.tools`` / ``agent.valid_tool_names`` — not merely deferred
behind the ``tool_search`` bridge. The #101289 probe (a fake Responses
endpoint inspecting wire ``tools[]``) never observed the MCP entry.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest
import uvicorn

import tools.mcp_tool as mcp_tool_origin
import tools.mcp_tool_loop as mcp_loop
from tools.registry import registry

PROBE_TOOL = "mcp__probe__probe_tool"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port_open(port: int, timeout: float = 25.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return
        except OSError:
            time.sleep(0.2)
    raise AssertionError(f"probe MCP server did not bind 127.0.0.1:{port}")


@pytest.fixture()
def probe_mcp_url():
    """URL of a live local MCP server exposing ``probe_tool``; torn down after."""
    from mcp.server.lowlevel import Server
    from mcp.types import (
        CallToolRequest,
        CallToolResult,
        ListToolsRequest,
        ListToolsResult,
        TextContent,
        Tool,
    )

    srv = Server("probe")

    async def _list(*_args):
        return ListToolsResult(tools=[Tool(
            name="probe_tool", description="Probe tool for 101289",
            inputSchema={"type": "object", "properties": {}})])

    async def _call(*_args):
        return CallToolResult(content=[TextContent(type="text", text="probe-ok")])

    srv.add_request_handler("tools/list", ListToolsRequest, _list)
    srv.add_request_handler("tools/call", CallToolRequest, _call)

    port = _free_port()
    app = srv.streamable_http_app()
    holder: dict = {}

    def _run() -> None:
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        holder["server"] = server
        asyncio.run(server.serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    _wait_port_open(port)
    time.sleep(0.5)  # let the ASGI app finish warming up
    yield f"http://127.0.0.1:{port}/mcp"
    server = holder.get("server")
    if server is not None:
        server.should_exit = True
        thread.join(timeout=15)


@pytest.fixture()
def _clean_mcp_state():
    """Snapshot/restore process-global MCP + registry state around each test."""
    saved = {
        "servers": dict(mcp_tool_origin._servers),
        "connecting": set(mcp_tool_origin._server_connecting),
        "errors": dict(mcp_tool_origin._server_connect_errors),
        "tool_servers": dict(mcp_tool_origin._mcp_tool_server_names),
        "tool_scopes": {k: set(v) for k, v in mcp_tool_origin._server_tool_scopes.items()},
        "scope_keys": dict(mcp_tool_origin._server_scope_keys),
        "parallel": set(mcp_tool_origin._parallel_safe_servers),
        "lazy_cfg": dict(mcp_tool_origin._lazy_server_configs),
        "aliases": dict(registry._toolset_aliases),
    }
    yield
    # Shut down the live client connection off the MCP loop, then restore.
    try:
        async def _down():
            with mcp_tool_origin._lock:
                srv = mcp_tool_origin._servers.pop("probe", None)
            if srv is not None:
                try:
                    await srv.shutdown()
                except Exception:
                    pass
        mcp_loop._run_on_mcp_loop(_down, timeout=30)
    except Exception:
        mcp_tool_origin._servers.pop("probe", None)
    registry.deregister(PROBE_TOOL)
    mcp_tool_origin._servers.clear()
    mcp_tool_origin._servers.update(saved["servers"])
    mcp_tool_origin._server_connecting.clear()
    mcp_tool_origin._server_connecting.update(saved["connecting"])
    mcp_tool_origin._server_connect_errors.clear()
    mcp_tool_origin._server_connect_errors.update(saved["errors"])
    mcp_tool_origin._mcp_tool_server_names.clear()
    mcp_tool_origin._mcp_tool_server_names.update(saved["tool_servers"])
    mcp_tool_origin._server_tool_scopes.clear()
    mcp_tool_origin._server_tool_scopes.update(saved["tool_scopes"])
    mcp_tool_origin._server_scope_keys.clear()
    mcp_tool_origin._server_scope_keys.update(saved["scope_keys"])
    mcp_tool_origin._parallel_safe_servers.clear()
    mcp_tool_origin._parallel_safe_servers.update(saved["parallel"])
    mcp_tool_origin._lazy_server_configs.clear()
    mcp_tool_origin._lazy_server_configs.update(saved["lazy_cfg"])
    registry._toolset_aliases.clear()
    registry._toolset_aliases.update(saved["aliases"])


@pytest.fixture()
def _stub_provider(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "openai-api", "api_mode": "responses",
            "base_url": "http://127.0.0.1:1", "api_key": "test-key",
            "command": None, "args": [],
        },
    )


def _direct_names(agent) -> set:
    return {(t.get("function") or {}).get("name", "") for t in (agent.tools or [])}


@pytest.mark.asyncio
async def test_session_new_mcp_server_is_directly_visible(
    tmp_path, monkeypatch, probe_mcp_url, _clean_mcp_state, _stub_provider,
):
    """session/new mcpServers: discovered tool must be a direct agent tool."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: openai-api\n  default: probe-model\n", encoding="utf-8")

    from acp.schema import McpServerHttp
    from acp_adapter.server import HermesACPAgent
    from acp_adapter.session import SessionManager

    mgr = SessionManager()
    agent = HermesACPAgent(session_manager=mgr)
    resp = await agent.new_session(
        cwd="/tmp", mcp_servers=[McpServerHttp(name="probe", url=probe_mcp_url, headers=[])])
    state = mgr.get_session(resp.session_id)

    assert "mcp-probe" in (state.agent.enabled_toolsets or [])
    assert PROBE_TOOL in _direct_names(state.agent), (
        f"discovered MCP tool missing from direct agent.tools: "
        f"{sorted(_direct_names(state.agent))}")
    assert PROBE_TOOL in (state.agent.valid_tool_names or set())


@pytest.mark.asyncio
async def test_config_mcp_server_is_directly_visible(
    tmp_path, monkeypatch, probe_mcp_url, _clean_mcp_state, _stub_provider,
):
    """config mcp_servers: background-discovered tool must be a direct agent tool."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: openai-api\n  default: probe-model\n"
        f"mcp_servers:\n  probe:\n    url: {probe_mcp_url}\n",
        encoding="utf-8")

    from acp_adapter.session import SessionManager

    mgr = SessionManager()
    state = mgr.create_session(cwd="/tmp")

    assert "mcp-probe" in (state.agent.enabled_toolsets or [])
    assert PROBE_TOOL in _direct_names(state.agent), (
        f"discovered MCP tool missing from direct agent.tools: "
        f"{sorted(_direct_names(state.agent))}")
    assert PROBE_TOOL in (state.agent.valid_tool_names or set())
