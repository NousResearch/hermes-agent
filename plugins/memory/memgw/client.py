"""Thin MCP client for the Memory Gateway, with a dedicated event loop.

The Memory Gateway exposes its rich tool surface (recall, write, reflect,
profile, search) over a Streamable-HTTP MCP endpoint. Hermes runs synchronously
on the turn path, so we drive the async MCP client from a single background
event loop and expose blocking helpers — the same pattern the Hindsight plugin
uses for its embedded server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 20.0


class MemGatewayClient:
    """Blocking facade over the gateway's Streamable-HTTP MCP endpoint."""

    def __init__(self, api_url: str, api_key: str | None = None, timeout: float = _DEFAULT_TIMEOUT):
        # Normalize: callers may pass the base host or the full /mcp path.
        self._api_url = api_url.rstrip('/')
        if not self._api_url.endswith('/mcp'):
            self._api_url = self._api_url + '/mcp'
        self._api_key = api_key or ''
        self._timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()

    # -- event loop plumbing -------------------------------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._loop and self._loop.is_running():
                return self._loop
            loop = asyncio.new_event_loop()

            def _run() -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            thread = threading.Thread(target=_run, daemon=True, name='memgw-loop')
            thread.start()
            self._loop = loop
            self._loop_thread = thread
            return loop

    def _run_sync(self, coro: Any) -> Any:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=self._timeout + 5.0)

    # -- MCP call ------------------------------------------------------------

    async def _call_tool_async(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        headers = {'Authorization': f'Bearer {self._api_key}'} if self._api_key else None
        async with streamablehttp_client(self._api_url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments=arguments)
                return self._unwrap(result)

    @staticmethod
    def _unwrap(result: Any) -> dict[str, Any]:
        """Extract a JSON dict from an MCP CallToolResult."""
        if getattr(result, 'isError', False):
            content = getattr(result, 'content', None) or []
            msg = next(
                (getattr(b, 'text', None) for b in content if getattr(b, 'text', None)),
                'tool error',
            )
            raise RuntimeError(f'MCP tool error: {msg}')
        # Prefer structured content when the server provides it.
        structured = getattr(result, 'structuredContent', None)
        if isinstance(structured, dict):
            return structured
        content = getattr(result, 'content', None) or []
        for block in content:
            text = getattr(block, 'text', None)
            if text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                    return {'result': parsed}
                except (ValueError, TypeError):
                    return {'result': text}
        return {}

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Blocking tool call. Raises on transport failure."""
        return self._run_sync(self._call_tool_async(name, arguments))

    def close(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop = None
        self._loop_thread = None
