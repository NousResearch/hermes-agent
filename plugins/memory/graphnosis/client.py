"""Minimal MCP client over Graphnosis's Unix-domain socket.

Wire format matches the sidecar socket transport: newline-delimited JSON-RPC.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_SOCKET = "~/.graphnosis/mcp.sock"
MCP_PROTOCOL_VERSION = "2024-11-05"


class GraphnosisMcpError(Exception):
    """Raised when the Graphnosis MCP socket is unreachable or returns an error."""


class GraphnosisMcpClient:
    """Thread-safe MCP client for recall / remember / stats tool calls."""

    def __init__(self, socket_path: str = DEFAULT_SOCKET):
        self._socket_path = str(Path(socket_path).expanduser())
        self._sock: Optional[socket.socket] = None
        self._buf = ""
        self._lock = threading.Lock()
        self._req_id = 0
        self._ready = False

    @property
    def socket_path(self) -> str:
        return self._socket_path

    @staticmethod
    def default_socket_exists() -> bool:
        return Path(DEFAULT_SOCKET).expanduser().exists()

    def connect(self) -> None:
        with self._lock:
            if self._ready:
                return
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(30.0)
                sock.connect(self._socket_path)
            except OSError as exc:
                raise GraphnosisMcpError(
                    "Graphnosis is not running or your cortex is locked. "
                    "Open the Graphnosis app and unlock your cortex, then retry."
                ) from exc
            self._sock = sock
            self._buf = ""
            self._initialize_locked()
            self._ready = True

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
            self._sock = None
            self._ready = False
            self._buf = ""

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Invoke an MCP tool and return the text content payload."""
        with self._lock:
            if not self._ready:
                self.connect()
            assert self._sock is not None
            req_id = self._next_id_locked()
            self._send_locked(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments or {}},
                }
            )
            result = self._read_until_id_locked(req_id)
        return _extract_tool_text(result)

    def _initialize_locked(self) -> None:
        assert self._sock is not None
        req_id = self._next_id_locked()
        self._send_locked(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "hermes-graphnosis", "version": "1.0.0"},
                },
            }
        )
        self._read_until_id_locked(req_id)
        self._send_locked({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _next_id_locked(self) -> int:
        self._req_id += 1
        return self._req_id

    def _send_locked(self, msg: Dict[str, Any]) -> None:
        assert self._sock is not None
        payload = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        self._sock.sendall(payload)

    def _read_until_id_locked(self, req_id: int) -> Dict[str, Any]:
        assert self._sock is not None
        while True:
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == req_id:
                    if "error" in msg:
                        err = msg["error"]
                        message = err.get("message", str(err))
                        raise GraphnosisMcpError(message)
                    return msg.get("result") or {}
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout as exc:
                raise GraphnosisMcpError("Graphnosis MCP request timed out") from exc
            if not chunk:
                raise GraphnosisMcpError("Graphnosis MCP connection closed unexpectedly")
            self._buf += chunk.decode("utf-8", errors="replace")


def _extract_tool_text(result: Dict[str, Any]) -> str:
    """Normalize MCP tools/call result to plain text."""
    if not result:
        return ""
    content = result.get("content")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    if isinstance(result.get("text"), str):
        return result["text"]
    return json.dumps(result)
