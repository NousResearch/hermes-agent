"""Stdio MCP server that advertises Hermes' tools to the official Claude Code
client (`claude -p`), so the model can emit `tool_use` blocks for them.

This is the tool half of the ``claude-code`` provider (see
``agent/claude_code_client.py`` and ``docs/claude-code-provider.md``). It is
launched by the client shim via ``--mcp-config`` as:

    python -m agent.claude_code_bridge <tools.json>

``<tools.json>`` is an Anthropic-format tool list (``[{name, description,
input_schema}, ...]``). Tools advertised here appear to the model as
``mcp__hermes__<name>`` (the ``hermes`` prefix comes from the MCP server key in
the ``--mcp-config`` file written by the shim).

Tool *execution* never actually happens through this bridge: the shim runs
``claude -p`` with ``--max-turns 1``, so the official client stops at the first
``tool_use`` (``stop_reason=tool_use``) without invoking the tool. Hermes
executes the tool itself. ``tools/call`` is implemented defensively only so a
stray invocation returns an inert sentinel instead of hanging.

Speaks the minimal subset of MCP (JSON-RPC 2.0 over stdio) that Claude Code
needs: ``initialize``, ``notifications/initialized``, ``tools/list``,
``tools/call``. No third-party dependency required.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

# MCP protocol version Claude Code negotiates against. Matches the value the
# official client sends in `initialize`; we echo a compatible one.
_PROTOCOL_VERSION = "2024-11-05"
_SERVER_NAME = "hermes"

# Returned by tools/call if the official client ever executes a tool despite
# --max-turns 1. Hermes owns execution, so this should never be the real path.
_INERT_SENTINEL = (
    "This tool is executed by the Hermes agent, not by Claude Code. "
    "No action was taken here."
)


def _load_tools(path: str) -> List[Dict[str, Any]]:
    """Load Anthropic-format tools and convert to MCP tool descriptors."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:
        return []
    tools: List[Dict[str, Any]] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        # Anthropic tools use `input_schema`; MCP uses `inputSchema`.
        schema = entry.get("input_schema") or entry.get("inputSchema") or {
            "type": "object",
            "properties": {},
        }
        tools.append({
            "name": name,
            "description": entry.get("description") or "",
            "inputSchema": schema,
        })
    return tools


def _send(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def serve(tools: List[Dict[str, Any]]) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if not isinstance(msg, dict):
            continue
        method = msg.get("method")
        mid = msg.get("id")
        if method == "initialize":
            _send({
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": _SERVER_NAME, "version": "1.0.0"},
                },
            })
        elif method == "notifications/initialized":
            # Notification — no response.
            continue
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": mid, "result": {"tools": tools}})
        elif method == "tools/call":
            _send({
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "content": [{"type": "text", "text": _INERT_SENTINEL}],
                    "isError": False,
                },
            })
        elif mid is not None:
            # Unknown request that expects a reply.
            _send({
                "jsonrpc": "2.0",
                "id": mid,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            })
        # Unknown notifications (no id) are ignored.


def main(argv: List[str]) -> int:
    tools = _load_tools(argv[1]) if len(argv) > 1 else []
    try:
        serve(tools)
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
