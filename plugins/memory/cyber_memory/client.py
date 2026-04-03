"""Provider-local stdio MCP client for Cyber Memory.

This client is intentionally narrow in scope: it launches the ``cyber-memory``
binary as a stdio MCP server, maintains a single MCP session, exposes a sync
``call_tool()`` API for the memory provider, and shuts everything down cleanly.

It does NOT participate in Hermes's global MCP registry/runtime. The Cyber
Memory provider is the public interface; MCP is only the transport layer.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List, Optional


def cyber_memory_mcp_available() -> bool:
    """Return True when the Python MCP client dependency is importable."""
    try:
        from mcp import ClientSession, StdioServerParameters  # noqa: F401
        from mcp.client.stdio import stdio_client  # noqa: F401
        return True
    except Exception:
        return False


class CyberMemoryClient:
    """Small persistent stdio MCP client for the Cyber Memory provider."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session = None
        self._tool_names: List[str] = []
        self._task = None
        self._ready = None
        self._shutdown_event = None
        self._error: Optional[Exception] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        command: str,
        args: List[str] | None = None,
        env: Optional[Dict[str, str]] = None,
        *,
        timeout: float = 25.0,
    ) -> None:
        """Start the Cyber Memory MCP subprocess and initialize a session."""
        if not cyber_memory_mcp_available():
            raise RuntimeError(
                "mcp package not installed. Install Hermes with MCP support "
                "(e.g. hermes-agent[mcp])."
            )

        with self._lock:
            if self._loop is not None:
                return

            self._loop = asyncio.new_event_loop()
            self._ready = asyncio.Event()
            self._shutdown_event = asyncio.Event()
            self._error = None
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="cyber-memory-mcp",
            )
            self._thread.start()

        future = asyncio.run_coroutine_threadsafe(
            self._async_boot(command, args or [], env or {}),
            self._loop,
        )
        try:
            future.result(timeout=timeout)
        except Exception:
            self.close()
            raise

    def close(self, *, timeout: float = 10.0) -> None:
        """Close the MCP session and stop the loop thread."""
        with self._lock:
            loop = self._loop
            thread = self._thread
            if loop is None:
                return

        try:
            future = asyncio.run_coroutine_threadsafe(self._async_shutdown(), loop)
            future.result(timeout=timeout)
        except Exception:
            pass
        finally:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
            if thread and thread.is_alive():
                thread.join(timeout=timeout)
            with self._lock:
                self._loop = None
                self._thread = None
                self._session = None
                self._tool_names = []
                self._task = None
                self._ready = None
                self._shutdown_event = None
                self._error = None

    def list_tools(self) -> List[str]:
        """Return the most recently discovered backend MCP tool names."""
        with self._lock:
            return list(self._tool_names)

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------

    def call_tool(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None, *,
        timeout: float = 45.0,
    ) -> Dict[str, Any]:
        """Call a backend MCP tool and return a JSON-compatible result."""
        with self._lock:
            loop = self._loop
        if loop is None:
            return {"error": "Cyber Memory client is not running"}

        future = asyncio.run_coroutine_threadsafe(
            self._async_call_tool(tool_name, arguments or {}),
            loop,
        )
        try:
            return future.result(timeout=timeout)
        except Exception as exc:
            return {"error": f"Cyber Memory MCP call failed: {type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    async def _async_boot(
        self, command: str, args: List[str], env: Dict[str, str]
    ) -> None:
        assert self._ready is not None
        self._task = asyncio.ensure_future(self._run(command, args, env))
        await self._ready.wait()
        if self._error:
            raise self._error

    async def _run(
        self, command: str, args: List[str], env: Dict[str, str]
    ) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=command,
            args=args,
            env=env or None,
        )

        try:
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    self._session = session
                    await self._session.initialize()
                    await self._refresh_tools()
                    if self._ready is not None:
                        self._ready.set()
                    if self._shutdown_event is not None:
                        await self._shutdown_event.wait()
        except Exception as exc:
            self._session = None
            if self._ready is not None and not self._ready.is_set():
                self._error = exc
                self._ready.set()
            raise
        finally:
            self._session = None

    async def _refresh_tools(self) -> None:
        if self._session is None:
            return
        tools_result = await self._session.list_tools()
        tools = getattr(tools_result, "tools", []) or []
        with self._lock:
            self._tool_names = [
                getattr(tool, "name", "")
                for tool in tools
                if getattr(tool, "name", "")
            ]

    async def _async_call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self._session is None:
            return {"error": "Cyber Memory session is not initialized"}

        result = await self._session.call_tool(tool_name, arguments=arguments)
        blocks = getattr(result, "content", []) or []
        text = self._extract_text(blocks).strip()

        if getattr(result, "isError", False):
            return {"error": text or f"Cyber Memory MCP tool '{tool_name}' returned an error"}

        if not text:
            return {"result": ""}

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        except Exception:
            return {"result": text}

    async def _async_shutdown(self) -> None:
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

    @staticmethod
    def _extract_text(blocks: List[Any]) -> str:
        parts: List[str] = []
        for block in blocks:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)
