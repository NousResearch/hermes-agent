#!/usr/bin/env python3
"""Standalone stdio MCP server that bridges `claude -p` tool calls back into
the parent Hermes process's real tool dispatcher.

Architecture (IPC decision — see module docstring in claude_live_client.py):

    claude -p  ──stdio JSON-RPC──▶  THIS bridge (separate process, spawned by
                                    claude via --mcp-config)
                                        │
                                        │  AF_UNIX stream, newline-delimited JSON
                                        ▼
                             parent Hermes process (LiveToolServer)
                                        │
                                        ▼
                             agent._invoke_tool(name, args)  ← the REAL dispatcher

The bridge itself executes NOTHING. It cannot: it is a distinct process without
the parent's live ``AIAgent`` (conversation state, guardrails, session db,
todo/memory stores). So every ``tools/call`` is forwarded over a unix-domain
socket to the parent, which owns tool execution and conversation state, and the
result is streamed back. Tool *schemas* are handed to the bridge as a JSON file
(byte-stable within a session for prompt-cache stability) so ``tools/list`` never
needs the socket.

Configured entirely via env (set by the parent in the --mcp-config server entry):
  * ``HERMES_MCP_BRIDGE_SOCKET`` — AF_UNIX path of the parent's LiveToolServer.
  * ``HERMES_MCP_BRIDGE_TOOLS``  — path to the JSON tool-defs file.
  * ``HERMES_MCP_BRIDGE_TOKEN``  — shared secret; first line sent on the socket
    so the parent rejects any stray local connection.

MCP surface implemented: initialize, notifications/initialized, tools/list,
tools/call (matching bench/mcp_server.py, proven against the real CLI).
"""

from __future__ import annotations

import json
import os
import socket
import sys
from typing import Any, Optional

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "hermes"
_SOCKET_TIMEOUT_S = 1800.0


# ---------------------------------------------------------------------------
# Schema translation (pure — imported by tests)
# ---------------------------------------------------------------------------


def translate_openai_tools_to_mcp(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """OpenAI ``{"function": {name, description, parameters}}`` → MCP tool defs.

    Names are emitted bare; Claude Code namespaces them as ``mcp__hermes__<name>``
    on its side. Output is deterministically ordered so the JSON file stays
    byte-stable across turns (prompt-cache stability)."""
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        out.append(
            {
                "name": name.strip(),
                "description": str(fn.get("description") or ""),
                "inputSchema": params,
            }
        )
    out.sort(key=lambda t: t["name"])
    return out


def load_tool_defs(path: Optional[str]) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    if isinstance(data, list):
        return [t for t in data if isinstance(t, dict) and t.get("name")]
    return []


# ---------------------------------------------------------------------------
# Parent socket client
# ---------------------------------------------------------------------------


class ParentToolClient:
    """Newline-delimited JSON-RPC-ish client to the parent LiveToolServer."""

    def __init__(self, socket_path: str, token: str):
        self._path = socket_path
        self._token = token

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute one tool via the parent. Returns ``{content, is_error}``."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(_SOCKET_TIMEOUT_S)
        try:
            sock.connect(self._path)
            request = {
                "token": self._token,
                "type": "call",
                "tool": name,
                "arguments": arguments,
            }
            sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
            payload = _recv_line(sock)
        finally:
            try:
                sock.close()
            except Exception:
                pass
        if not payload:
            return {"content": "tool bridge: empty response from parent", "is_error": True}
        try:
            response = json.loads(payload)
        except Exception:
            return {"content": "tool bridge: malformed parent response", "is_error": True}
        return {
            "content": str(response.get("content", "")),
            "is_error": bool(response.get("is_error", False)),
        }


def _recv_line(sock: socket.socket) -> str:
    chunks: list[bytes] = []
    while True:
        data = sock.recv(65536)
        if not data:
            break
        chunks.append(data)
        if b"\n" in data:
            break
    return b"".join(chunks).split(b"\n", 1)[0].decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# JSON-RPC request handling
# ---------------------------------------------------------------------------


def handle_request(
    req: dict[str, Any],
    *,
    tools: list[dict[str, Any]],
    parent: Optional[ParentToolClient],
) -> Optional[dict[str, Any]]:
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params") or {}

    if method == "initialize":
        return _ok(
            req_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": "0.1.0"},
            },
        )
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return _ok(req_id, {"tools": tools})
    if method == "tools/call":
        return _handle_tools_call(req_id, params, parent)
    if req_id is not None:
        return _err(req_id, -32601, f"Method not found: {method}")
    return None


def _handle_tools_call(
    req_id: Any, params: dict[str, Any], parent: Optional[ParentToolClient]
) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if not isinstance(name, str) or not name.strip():
        return _err(req_id, -32602, "tools/call missing tool name")
    if parent is None:
        return _tool_error_result(req_id, "tool bridge not connected to parent")
    try:
        outcome = parent.call(name.strip(), arguments if isinstance(arguments, dict) else {})
    except Exception as exc:  # socket failure — surface as a tool error, not a crash
        return _tool_error_result(req_id, f"tool bridge error: {exc}")
    return _ok(
        req_id,
        {
            "content": [{"type": "text", "text": outcome.get("content", "")}],
            "isError": bool(outcome.get("is_error", False)),
        },
    )


def _ok(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _tool_error_result(req_id: Any, message: str) -> dict[str, Any]:
    # A failed tool is reported as a successful JSON-RPC result carrying an MCP
    # tool error, so the model can react instead of the turn hard-failing.
    return _ok(req_id, {"content": [{"type": "text", "text": message}], "isError": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    tools = load_tool_defs(os.getenv("HERMES_MCP_BRIDGE_TOOLS"))
    socket_path = os.getenv("HERMES_MCP_BRIDGE_SOCKET", "").strip()
    token = os.getenv("HERMES_MCP_BRIDGE_TOKEN", "").strip()
    parent = ParentToolClient(socket_path, token) if socket_path else None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            resp = handle_request(req, tools=tools, parent=parent)
        except Exception as exc:
            resp = _err(req.get("id"), -32000, str(exc))
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
