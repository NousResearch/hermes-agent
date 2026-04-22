#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def _call(url: str, tool_name: str) -> None:
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(10.0, read=60.0),
        trust_env=False,
    ) as client:
        async with streamable_http_client(url, http_client=client) as (
            read_stream,
            write_stream,
            _session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, {})
                print(json.dumps(result.model_dump(mode="json"), indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trigger a Granola broker call so Spark-side OAuth can complete."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8767/mcp/granola",
        help="Granola MCP broker URL",
    )
    parser.add_argument(
        "--tool",
        default="granola.folders.list",
        help="Granola tool to call",
    )
    args = parser.parse_args()
    anyio.run(_call, args.url, args.tool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
