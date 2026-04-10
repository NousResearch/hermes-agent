"""Asyncio HTTP proxy server for credential placeholder substitution.

Phase 1: plain HTTP only.  CONNECT (HTTPS) tunnelling returns 501 — that
requires the local-CA MITM layer in Phase 2.

The proxy listens on a Unix domain socket and rewrites ``hermes-proxy://<name>``
placeholders in request headers and bodies before forwarding to the real
upstream server.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import socket
from http.client import HTTPConnection
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple
from urllib.parse import urlparse

if TYPE_CHECKING:
    from proxy.store import CredentialStore

logger = logging.getLogger(__name__)

# Matches hermes-proxy://<credential-name> where name is alphanumeric + _ + -
_PLACEHOLDER_RE = re.compile(r"hermes-proxy://([A-Za-z0-9_-]+)")


def _substitute_placeholders(text: str, store: "CredentialStore") -> str:
    """Replace all ``hermes-proxy://<name>`` occurrences with real values."""
    def _replace(match: re.Match) -> str:
        name = match.group(1)
        value = store._resolve(name)
        if value is None:
            logger.warning("unresolved credential placeholder: %s", name)
            return match.group(0)  # leave as-is
        return value

    return _PLACEHOLDER_RE.sub(_replace, text)


class _ProxyProtocol(asyncio.Protocol):
    """Handle one inbound HTTP request from a tool subprocess."""

    def __init__(self, store: "CredentialStore") -> None:
        self._store = store
        self._transport: Optional[asyncio.Transport] = None
        self._buffer = b""

    def connection_made(self, transport: asyncio.Transport) -> None:
        self._transport = transport

    def data_received(self, data: bytes) -> None:
        self._buffer += data
        # Wait until we have the full header block
        if b"\r\n\r\n" not in self._buffer:
            return
        asyncio.ensure_future(self._handle_request())

    def connection_lost(self, exc: Optional[Exception]) -> None:
        self._transport = None

    async def _handle_request(self) -> None:
        """Parse, rewrite, forward, and relay the response."""
        try:
            raw = self._buffer
            self._buffer = b""

            # Split headers and body
            header_end = raw.index(b"\r\n\r\n")
            header_block = raw[:header_end].decode("utf-8", errors="replace")
            body = raw[header_end + 4:]

            lines = header_block.split("\r\n")
            request_line = lines[0]
            parts = request_line.split(" ", 2)
            if len(parts) < 3:
                self._send_error(400, "Bad Request")
                return

            method, url, version = parts

            # CONNECT = HTTPS tunnel — not supported in Phase 1
            if method.upper() == "CONNECT":
                self._send_error(
                    501,
                    "Not Implemented",
                    "HTTPS CONNECT tunnelling requires Phase 2 (local CA MITM).\n",
                )
                return

            # Parse the target URL
            parsed = urlparse(url)
            host = parsed.hostname or ""
            port = parsed.port or 80
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query

            # Rewrite headers — substitute placeholders
            rewritten_headers: list[Tuple[str, str]] = []
            content_length: Optional[int] = None
            for line in lines[1:]:
                if ":" not in line:
                    continue
                key, _, val = line.partition(":")
                val = val.strip()
                val = _substitute_placeholders(val, self._store)
                rewritten_headers.append((key.strip(), val))
                if key.strip().lower() == "content-length":
                    try:
                        content_length = int(val)
                    except ValueError:
                        pass

            # If there's a body with a Content-Length, read remaining data
            if content_length is not None and len(body) < content_length:
                # For Phase 1, we handle small bodies only; large streaming
                # bodies are a Phase 2 concern.
                remaining = content_length - len(body)
                # We already have everything in the buffer for typical API calls
                if remaining > 0:
                    logger.debug("body incomplete: have %d, need %d more", len(body), remaining)

            # Substitute placeholders in body
            body_str = body.decode("utf-8", errors="replace")
            body_str = _substitute_placeholders(body_str, self._store)
            body = body_str.encode("utf-8")

            # Update Content-Length if body changed
            new_content_length = len(body)

            # Forward to upstream via blocking HTTPConnection in executor
            loop = asyncio.get_event_loop()
            response_data = await loop.run_in_executor(
                None,
                self._forward_request,
                host, port, method, path, rewritten_headers, body, new_content_length,
            )

            if self._transport and not self._transport.is_closing():
                self._transport.write(response_data)
                self._transport.close()

        except Exception as exc:
            logger.exception("proxy request failed: %s", exc)
            self._send_error(502, "Bad Gateway", str(exc))

    def _forward_request(
        self,
        host: str,
        port: int,
        method: str,
        path: str,
        headers: list[Tuple[str, str]],
        body: bytes,
        content_length: int,
    ) -> bytes:
        """Synchronously forward the request and return the raw HTTP response."""
        conn = HTTPConnection(host, port, timeout=30)
        try:
            conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            host_written = False
            cl_written = False
            for key, val in headers:
                if key.lower() == "content-length":
                    conn.putheader("Content-Length", str(content_length))
                    cl_written = True
                elif key.lower() == "host":
                    conn.putheader("Host", val)
                    host_written = True
                else:
                    conn.putheader(key, val)
            if not host_written:
                conn.putheader("Host", host if port == 80 else f"{host}:{port}")
            if body and not cl_written:
                conn.putheader("Content-Length", str(content_length))
            conn.endheaders(body if body else None)

            resp = conn.getresponse()
            resp_body = resp.read()

            # Build raw HTTP response
            status_line = f"HTTP/1.1 {resp.status} {resp.reason}\r\n"
            resp_headers = "".join(
                f"{k}: {v}\r\n" for k, v in resp.getheaders()
            )
            return (status_line + resp_headers + "\r\n").encode() + resp_body
        finally:
            conn.close()

    def _send_error(self, code: int, reason: str, body: str = "") -> None:
        """Send an HTTP error response and close."""
        if not body:
            body = f"{code} {reason}\n"
        resp = (
            f"HTTP/1.1 {code} {reason}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Content-Type: text/plain\r\n"
            f"\r\n"
            f"{body}"
        )
        if self._transport and not self._transport.is_closing():
            self._transport.write(resp.encode())
            self._transport.close()


async def run_proxy(
    socket_path: Path,
    store: "CredentialStore",
    ready_event: Optional[asyncio.Event] = None,
) -> None:
    """Start the proxy server on a Unix domain socket.

    Args:
        socket_path: Path to the Unix socket.
        store: The credential store instance.
        ready_event: Optional event to set once the server is listening.
    """
    # Clean up stale socket
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    loop = asyncio.get_event_loop()
    server = await loop.create_unix_server(
        lambda: _ProxyProtocol(store),
        path=str(socket_path),
    )

    # Make socket accessible
    os.chmod(str(socket_path), 0o600)

    logger.info("credential proxy listening on %s", socket_path)
    if ready_event:
        ready_event.set()

    stop_event = asyncio.Event()

    def _shutdown(sig: int, frame) -> None:
        logger.info("credential proxy received signal %d, shutting down", sig)
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    await stop_event.wait()

    server.close()
    await server.wait_closed()
    if socket_path.exists():
        socket_path.unlink()
    logger.info("credential proxy shut down")
