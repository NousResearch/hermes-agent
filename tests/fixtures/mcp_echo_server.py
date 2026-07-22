#!/usr/bin/env python3
"""Minimal MCP server that exposes a single 'echo' tool.

Used by integration tests to verify MCP server connection and scoping
without depending on external packages (npx/uvx).

Spawn via stdio: python3 mcp_echo_server.py
"""
import asyncio
import json
import sys

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions, ServerCapabilities
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ToolsCapability


def _make_echo_tool() -> Tool:
    return Tool(
        name="echo",
        description="Echo back the provided message",
        inputSchema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo back",
                },
            },
            "required": ["message"],
        },
    )


async def _serve() -> None:
    server = Server("mcp-echo-test")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [_make_echo_tool()]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "echo":
            msg = arguments.get("message", "")
            return [TextContent(type="text", text=f"ECHO: {msg}")]
        raise ValueError(f"Unknown tool: {name}")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-echo-test",
                server_version="1.0.0",
                capabilities=ServerCapabilities(
                    tools=ToolsCapability(listChanged=True),
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(_serve())
