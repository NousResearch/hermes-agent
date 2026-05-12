"""Minimal LSP JSON-RPC 2.0 framer over async streams.

LSP wire format:

    Content-Length: <bytes>\\r\\n
    \\r\\n
    <utf-8 JSON body>

The body is a JSON-RPC 2.0 envelope: request, response, or notification.

This module replaces what ``vscode-jsonrpc/node`` would do in a
TypeScript implementation.  We keep it deliberately small — just the
framer + envelope helpers — so :class:`agent.lsp.client.LSPClient` can
focus on protocol semantics.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger("agent.lsp.protocol")

# LSP error codes we care about.  Full list in
# https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#errorCodes
ERROR_CONTENT_MODIFIED = -32801
ERROR_REQUEST_CANCELLED = -32800
ERROR_METHOD_NOT_FOUND = -32601


class LSPProtocolError(Exception):
    """Raised when the wire protocol is violated.

    Distinct from :class:`LSPRequestError` which represents a server
    returning a JSON-RPC error response — that's protocol-conformant.
    This exception means the framing or envelope itself is broken.
    """


class LSPRequestError(Exception):
    """Raised when an LSP request returns an error response.

    Carries the JSON-RPC ``code``, ``message``, and optional ``data``.
    """

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"LSP error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


def encode_message(obj: dict) -> bytes:
    """Encode a JSON-RPC envelope as a Content-Length framed byte string.

    The body is encoded as compact UTF-8 JSON (no spaces between
    separators) — matches what ``vscode-jsonrpc`` emits and keeps the
    Content-Length count exact.
    """
    body = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


async def read_message(reader: asyncio.StreamReader) -> Optional[dict]:
    """Read one Content-Length framed JSON-RPC message from the stream.

    Returns ``None`` on clean EOF (server closed stdout cleanly between
    messages — typical shutdown).  Raises :class:`LSPProtocolError` on
    malformed framing.

    The reader is advanced to just past the JSON body on success.
    """
    headers: dict = {}
    header_bytes = 0
    while True:
        try:
            line = await reader.readuntil(b"\r\n")
        except asyncio.IncompleteReadError as e:
            # EOF while reading headers.  If we hadn't started a header
            # block, treat as clean EOF; otherwise the framing is bad.
            if not e.partial and not headers:
                return None
            raise LSPProtocolError(
                f"unexpected EOF while reading LSP headers (partial={e.partial!r})"
            ) from e
        # Defensive cap against a server streaming headers without ever
        # emitting CRLF-CRLF.  Caps total header bytes at 8 KiB — a
        # well-behaved server fits in well under 200 bytes.
        header_bytes += len(line)
        if header_bytes > 8192:
            raise LSPProtocolError(
                f"LSP header block exceeded 8 KiB without terminator"
            )
        line = line[:-2]  # strip CRLF
        if not line:
            break  # blank line ends header block
        try:
            key, _, value = line.decode("ascii").partition(":")
        except UnicodeDecodeError as e:
            raise LSPProtocolError(f"non-ASCII LSP header: {line!r}") from e
        if not key:
            raise LSPProtocolError(f"malformed LSP header line: {line!r}")
        headers[key.strip().lower()] = value.strip()

    cl = headers.get("content-length")
    if cl is None:
        raise LSPProtocolError(f"LSP message missing Content-Length: {headers!r}")
    try:
        n = int(cl)
    except ValueError as e:
        raise LSPProtocolError(f"non-integer Content-Length: {cl!r}") from e
    if n < 0 or n > 64 * 1024 * 1024:  # 64 MiB sanity cap
        raise LSPProtocolError(f"unreasonable Content-Length: {n}")

    try:
        body = await reader.readexactly(n)
    except asyncio.IncompleteReadError as e:
        raise LSPProtocolError(
            f"truncated LSP body: expected {n} bytes, got {len(e.partial)}"
        ) from e

    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise LSPProtocolError(f"invalid JSON in LSP body: {e}") from e
    except UnicodeDecodeError as e:
        raise LSPProtocolError(f"non-UTF-8 LSP body: {e}") from e


def make_request(req_id: int, method: str, params: Any) -> dict:
    """Build a JSON-RPC 2.0 request envelope."""
    msg: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def make_notification(method: str, params: Any) -> dict:
    """Build a JSON-RPC 2.0 notification envelope (no ``id``)."""
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def make_response(req_id: Any, result: Any) -> dict:
    """Build a JSON-RPC 2.0 success response envelope."""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def make_error_response(req_id: Any, code: int, message: str, data: Any = None) -> dict:
    """Build a JSON-RPC 2.0 error response envelope."""
    err: dict = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def classify_message(msg: dict) -> Tuple[str, Any]:
    """Return ``(kind, key)`` where kind is one of ``request``,
    ``response``, ``notification``, ``invalid``.

    The key is the request id for request/response, the method name
    for notifications, and ``None`` for invalid messages.
    """
    if not isinstance(msg, dict):
        return "invalid", None
    if msg.get("jsonrpc") != "2.0":
        return "invalid", None
    has_id = "id" in msg
    has_method = "method" in msg
    if has_id and has_method:
        return "request", msg["id"]
    if has_id and ("result" in msg or "error" in msg):
        return "response", msg["id"]
    if has_method and not has_id:
        return "notification", msg["method"]
    return "invalid", None


__all__ = [
    "ERROR_CONTENT_MODIFIED",
    "ERROR_REQUEST_CANCELLED",
    "ERROR_METHOD_NOT_FOUND",
    "LSPProtocolError",
    "LSPRequestError",
    "encode_message",
    "read_message",
    "make_request",
    "make_notification",
    "make_response",
    "make_error_response",
    "classify_message",
]
