"""``initialize()`` must advertise ``mcpCapabilities`` for HTTP/SSE session MCP (#124910),
and ``_mcp_server_config`` must keep the transport kind the client declared.

The adapter registers client-provided ``McpServerHttp``/``McpServerSse`` servers
(``_register_session_mcp_servers`` → ``_mcp_server_config`` builds ``{"url", "headers"}``
configs), but its ``initialize()`` response never declared the capability, so
capability-checking clients (e.g. qwen-audio-agent) refuse the session with 422 and drop
Hermes as a backend. ``McpServerSse`` additionally needs ``transport: sse`` kept in the
config, or the dispatcher tries Streamable HTTP first and an SSE-only server that rejects
the chunked initialize POST outside the 400-family set fails the connect with zero tools.
"""

from __future__ import annotations

import asyncio

import pytest

acp_schema = pytest.importorskip("acp.schema")


def _initialize_response():
    from acp_adapter.server import HermesACPAgent

    agent = HermesACPAgent.__new__(HermesACPAgent)  # no __init__: network/model state not needed
    return asyncio.run(agent.initialize())


def test_initialize_advertises_http_and_sse_mcp():
    response = _initialize_response()
    caps = response.agent_capabilities
    assert caps.mcp_capabilities is not None, (
        "initialize() must declare mcpCapabilities: clients gate MCP-over-HTTP on it (#124910)"
    )
    assert caps.mcp_capabilities.http is True
    assert caps.mcp_capabilities.sse is True


def test_initialize_serializes_mcp_capabilities_alias():
    """The wire format uses the camelCase ``mcpCapabilities`` key."""
    response = _initialize_response()
    payload = response.model_dump(by_alias=True, exclude_none=True)
    assert payload["agentCapabilities"]["mcpCapabilities"] == {"http": True, "sse": True}


def _mk(kind: str, **kw):
    cls = {"stdio": acp_schema.McpServerStdio, "http": acp_schema.McpServerHttp, "sse": acp_schema.McpServerSse}[kind]
    if kind == "stdio":
        return cls(name="s", command="x", args=[], env=[])
    return cls(name="s", url="http://example.invalid/mcp", headers=[])


def test_mcp_server_config_keeps_sse_transport_kind():
    from acp_adapter.server import _mcp_server_config

    cfg = _mcp_server_config(_mk("sse"))
    assert cfg.get("transport") == "sse", (
        "McpServerSse must carry transport=sse or the dispatcher connects over Streamable "
        "HTTP first and SSE-only servers outside the 400-family rejection set fail outright"
    )
    assert cfg["url"] == "http://example.invalid/mcp"


def test_mcp_server_config_http_has_no_transport_override():
    from acp_adapter.server import _mcp_server_config

    cfg = _mcp_server_config(_mk("http"))
    assert "transport" not in cfg, "McpServerHttp must keep the Streamable HTTP default path"
    assert cfg["url"] == "http://example.invalid/mcp"


def test_mcp_server_config_stdio_shape_unchanged():
    from acp_adapter.server import _mcp_server_config

    cfg = _mcp_server_config(_mk("stdio"))
    assert cfg == {"command": "x", "args": [], "env": {}}
