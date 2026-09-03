"""Wire-level regression tests for request-local ACP MCP validation."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


class _DrainProtocol(asyncio.Protocol):
    async def _drain_helper(self) -> None:
        return None

    def _get_close_waiter(self, _stream: asyncio.StreamWriter) -> asyncio.Future[None]:
        waiter = asyncio.get_running_loop().create_future()
        waiter.set_result(None)
        return waiter


class _CapturingTransport(asyncio.WriteTransport):
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.response_ready = asyncio.Event()
        self._closing = False

    def write(self, data: bytes | bytearray | memoryview[Any]) -> None:
        self.buffer.extend(data)
        if b"\n" in data:
            self.response_ready.set()

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        self._closing = True

    def abort(self) -> None:
        self.close()

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        return default


async def _send_wire_request(agent: HermesACPAgent, request: dict) -> bytes:
    """Send one JSON-RPC request through the real ACP connection and serializer."""
    loop = asyncio.get_running_loop()
    request_reader = asyncio.StreamReader()
    response_transport = _CapturingTransport()
    response_writer = asyncio.StreamWriter(
        response_transport,
        _DrainProtocol(),
        request_reader,
        loop,
    )
    agent_task = asyncio.create_task(
        acp.run_agent(
            agent,
            input_stream=response_writer,
            output_stream=request_reader,
            use_unstable_protocol=True,
        )
    )

    request_reader.feed_data((json.dumps(request) + "\n").encode())
    try:
        await asyncio.wait_for(response_transport.response_ready.wait(), timeout=5.0)
        return bytes(response_transport.buffer)
    finally:
        request_reader.feed_eof()
        await asyncio.wait_for(agent_task, timeout=2.0)
        response_writer.close()
        await response_writer.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "extra_params"),
    [
        pytest.param("session/new", {}, id="new"),
        pytest.param("session/load", {"sessionId": "existing"}, id="load"),
        pytest.param("session/resume", {"sessionId": "existing"}, id="resume"),
        pytest.param("session/fork", {"sessionId": "existing"}, id="fork"),
    ],
)
async def test_duplicate_mcp_names_are_invalid_params_without_side_effects(
    tmp_path, method, extra_params
):
    manager = SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))
    agent = HermesACPAgent(session_manager=manager)
    mcp_servers = [
        {
            "name": "duplicate",
            "command": "/private/mcp-command",
            "args": ["--token", "private-argument"],
            "env": [{"name": "PRIVATE_CONFIG", "value": "private-config-value"}],
        },
        {
            "type": "http",
            "name": "duplicate",
            "url": "https://credential.example/mcp?token=private-url-value",
            "headers": [
                {"name": "Authorization", "value": "Bearer private-header-value"}
            ],
        },
    ]
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": {
            "cwd": str(tmp_path),
            "mcpServers": mcp_servers,
            **extra_params,
        },
    }

    with (
        patch.object(manager, "create_session", wraps=manager.create_session) as create,
        patch.object(manager, "update_cwd", wraps=manager.update_cwd) as update_cwd,
        patch.object(manager, "fork_session", wraps=manager.fork_session) as fork,
        patch.object(
            agent, "_register_session_mcp_servers", new_callable=AsyncMock
        ) as register,
    ):
        raw_response = await _send_wire_request(agent, request)

    assert json.loads(raw_response) == {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32602,
            "message": "Invalid params",
            "data": {"serverName": "duplicate"},
        },
    }
    for private_value in (
        "/private/mcp-command",
        "private-argument",
        "PRIVATE_CONFIG",
        "private-config-value",
        "credential.example",
        "private-url-value",
        "Authorization",
        "private-header-value",
    ):
        assert private_value.encode() not in raw_response
    create.assert_not_called()
    update_cwd.assert_not_called()
    fork.assert_not_called()
    register.assert_not_awaited()
    assert manager._sessions == {}
