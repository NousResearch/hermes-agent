from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

from anyio import ClosedResourceError
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .core import ToolExecutionError, normalize_json


class RemoteMcpBridge:
    def __init__(self, *, command: str, args: list[str], env: dict[str, str] | None = None) -> None:
        self._command = command
        self._args = args
        self._env = env or {}
        self._lock = asyncio.Lock()
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._tool_cache: list[str] | None = None

    async def _reset(self) -> None:
        async with self._lock:
            stack = self._stack
            self._stack = None
            self._session = None
            self._tool_cache = None
        if stack is not None:
            await stack.aclose()

    async def ensure_started(self) -> None:
        async with self._lock:
            if self._session is not None:
                return
            stack = AsyncExitStack()
            params = StdioServerParameters(command=self._command, args=self._args, env=self._env)
            read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            self._stack = stack
            self._session = session
            self._tool_cache = None

    async def list_tools(self) -> list[str]:
        for _ in range(2):
            await self.ensure_started()
            assert self._session is not None
            try:
                tools = await self._session.list_tools()
            except ClosedResourceError:
                await self._reset()
                continue
            self._tool_cache = [tool.name for tool in tools.tools]
            return list(self._tool_cache)
        raise ToolExecutionError("Remote MCP session closed while listing tools.", category="provider")

    async def resolve_tool(self, explicit: str | None, candidates: list[str]) -> str:
        tools = self._tool_cache or await self.list_tools()
        if explicit:
            if explicit not in tools:
                raise ToolExecutionError(
                    f"Configured remote MCP tool '{explicit}' was not found. Available tools: {', '.join(sorted(tools))}",
                    category="validation",
                )
            return explicit
        for candidate in candidates:
            if candidate in tools:
                return candidate
        raise ToolExecutionError(
            f"No compatible remote MCP tool found. Expected one of {candidates}; available tools: {', '.join(sorted(tools))}",
            category="validation",
        )

    async def call_tool(self, remote_name: str, arguments: dict[str, Any]) -> Any:
        for _ in range(2):
            await self.ensure_started()
            assert self._session is not None
            try:
                result = await self._session.call_tool(remote_name, arguments)
            except ClosedResourceError:
                await self._reset()
                continue
            return normalize_json(result)
        raise ToolExecutionError(f"Remote MCP session closed while calling '{remote_name}'.", category="provider")
