#!/usr/bin/env python3
"""MCP stdio server: EVM + Solana RPC tools (optional ``hermes-agent[web3-mcp]``)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import anyio
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from tool_handlers import handle_tool
from tools_schema import build_tool_list

logger = logging.getLogger("web3_mcp_server")


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")


def build_server() -> Server:
    app = Server("hermes-web3-chain-tools")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return build_tool_list()

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> Any:
        try:
            return await handle_tool(name, arguments or {})
        except Exception as exc:  # noqa: BLE001
            logger.exception("tool error")
            return [types.TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    return app


def main() -> None:
    _configure_logging()
    app = build_server()

    async def arun() -> None:
        async with stdio_server() as streams:
            await app.run(
                streams[0],
                streams[1],
                app.create_initialization_options(),
            )

    anyio.run(arun)


if __name__ == "__main__":
    main()
