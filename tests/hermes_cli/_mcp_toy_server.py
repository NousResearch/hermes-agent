"""Tiny real MCP stdio server used ONLY as a test fixture.

Exposes two tools:
- ``echo`` returns its ``text`` argument verbatim (or raises when called with
  ``{"fail": true}``).
- ``attachment`` returns a mix of text and image content, to exercise
  non-text call results.

Not part of the CLI — a real process the record/replay tests spawn and
record against, per the "no mocked transport" goal of ``hermes mcp
fixtures``.

Run standalone: ``python -m tests.hermes_cli._mcp_toy_server``
"""

from __future__ import annotations

import asyncio
from typing import List


async def _run() -> None:
    from mcp import types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    server: Server = Server("hermes-mcp-toy-server")

    @server.list_tools()
    async def _list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="echo",
                description="Echo the given text back",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "fail": {"type": "boolean"},
                    },
                },
            ),
            types.Tool(
                name="attachment",
                description="Return a mix of text and image content",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict):
        if name == "echo":
            if arguments.get("fail"):
                raise RuntimeError("toy server: intentional failure")
            return [types.TextContent(type="text", text=str(arguments.get("text", "")))]
        if name == "attachment":
            return [
                types.TextContent(type="text", text="caption"),
                types.ImageContent(type="image", data="Zm9v", mimeType="image/png"),
            ]
        raise ValueError(f"unknown tool {name!r}")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(_run())
