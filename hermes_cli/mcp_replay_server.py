"""Deterministic MCP replay server — stdio only.

Serves the tools/calls recorded in a fixture produced by
``hermes mcp fixtures record`` over a REAL ``mcp`` stdio server, so a real
``ClientSession`` gets real protocol responses without touching the
original backend, network, or credentials.

Standalone entry point (usable directly as an ``mcp_servers.<name>.command``
in a test's config, or driven by ``hermes mcp fixtures replay`` for a
self-check round-trip)::

    python -m hermes_cli.mcp_replay_server <fixture.json>
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["load_fixture", "run_replay_server", "main"]


def load_fixture(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_call(
    fixture: Dict[str, Any], name: str, arguments: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    for call in fixture.get("calls", []):
        if call.get("name") == name and (call.get("arguments") or {}) == (
            arguments or {}
        ):
            return call
    return None


async def run_replay_server(fixture_path: Path) -> None:
    """Serve ``fixture_path`` over stdio until the client disconnects."""
    from mcp import types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    fixture = load_fixture(fixture_path)
    server: Server = Server(f"hermes-mcp-replay:{fixture.get('server_name', 'unknown')}")

    @server.list_tools()
    async def _list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name=t["name"],
                description=t.get("description"),
                inputSchema=t.get("input_schema") or {"type": "object"},
            )
            for t in fixture.get("tools", [])
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> List[types.TextContent]:
        recorded = _find_call(fixture, name, arguments or {})
        if recorded is None:
            raise ValueError(
                f"no recorded call for {name}({arguments!r}) in fixture "
                f"{fixture_path.name} — replay is deterministic and only "
                "answers exactly what was recorded"
            )
        if "error" in recorded:
            raise RuntimeError(recorded["error"])
        return [
            types.TextContent(type="text", text=c.get("text") or "")
            for c in recorded.get("content", [])
        ]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(
            "usage: python -m hermes_cli.mcp_replay_server <fixture.json>",
            file=sys.stderr,
        )
        return 1
    asyncio.run(run_replay_server(Path(argv[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
