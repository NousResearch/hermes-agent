"""asyncio HTTP proxy for credential placeholder substitution (Phase 1 — HTTP only).

For plain HTTP requests, the proxy substitutes ``hermes-proxy://<name>`` tokens
in request headers and the request body before forwarding to the upstream server.

For CONNECT tunnels (HTTPS), the proxy establishes a blind TCP relay without
interception.  Credential substitution inside HTTPS traffic is Phase 2 scope
and requires MITM CA generation — not implemented here.

No external dependencies — stdlib asyncio only.

``CredentialProxyAddon`` is retained as a thin, mitmproxy-free substitution
helper so that existing unit tests can exercise header/body rewriting without
spinning up a proxy server.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.parse

from .store import CredStore
from .substitutor import substitute

logger = logging.getLogger(__name__)

_MAX_HEADER_SIZE = 65_536    # 64 KiB — refuse oversized headers
_READ_CHUNK = 65_536
_CONNECT_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# CredentialProxyAddon — pure-Python substitution hook (no mitmproxy dep)
# ---------------------------------------------------------------------------

class CredentialProxyAddon:
    """Substitutes ``hermes-proxy://`` credential placeholders in HTTP flows.

    Accepts any duck-typed flow object with:
      - ``flow.request.headers``  — dict-like mapping of header name → value
      - ``flow.request.content``  — bytes or None

    This interface is intentionally kept compatible with mitmproxy's Flow so
    that existing tests work unchanged.  The proxy itself (``run_proxy``) uses
    this class internally for substitution; it does not require mitmproxy at
    runtime.
    """

    def __init__(self, store: CredStore | None = None) -> None:
        self._store = store if store is not None else CredStore()

    def request(self, flow) -> None:
        """Substitute credential placeholders in request headers and body."""
        # Headers
        for key in list(flow.request.headers.keys()):
            val = flow.request.headers[key]
            new_val = substitute(val, self._store)
            if new_val != val:
                logger.debug("Substituted credential in request header %r", key)
            flow.request.headers[key] = new_val

        # Body — caller is responsible for Content-Length recalculation when
        # using this method directly; the asyncio proxy handles it automatically.
        if flow.request.content:
            text = flow.request.content.decode("utf-8", errors="replace")
            new_text = substitute(text, self._store)
            if new_text != text:
                logger.debug("Substituted credential in request body")
                flow.request.content = new_text.encode("utf-8")


# ---------------------------------------------------------------------------
# asyncio proxy internals
# ---------------------------------------------------------------------------

async def _pipe(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Copy bytes from *reader* to *writer* until EOF or connection reset."""
    try:
        while True:
            data = await reader.read(_READ_CHUNK)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


def _parse_request_line(line: bytes) -> tuple[str, str, str]:
    """Parse ``b'METHOD URL HTTP/1.x'`` → ``(method, url, version)``."""
    parts = line.decode("latin-1").strip().split(" ", 2)
    if len(parts) != 3:
        raise ValueError(f"Malformed request line: {line!r}")
    return parts[0], parts[1], parts[2]


def _parse_headers(raw: bytes) -> list[tuple[str, str]]:
    """Parse raw header block into list of ``(name, value)`` tuples."""
    headers: list[tuple[str, str]] = []
    for line in raw.split(b"\r\n"):
        if b":" not in line:
            continue
        name, _, value = line.partition(b":")
        headers.append((
            name.decode("latin-1").strip(),
            value.decode("latin-1").strip(),
        ))
    return headers


def _headers_to_bytes(headers: list[tuple[str, str]]) -> bytes:
    return (
        b"\r\n".join(
            f"{name}: {value}".encode("latin-1") for name, value in headers
        )
        + b"\r\n"
    )


async def _read_request(
    reader: asyncio.StreamReader,
) -> tuple[bytes, bytes, bytes] | None:
    """Read a complete HTTP request from *reader*.

    Returns ``(request_line, raw_headers, body)`` bytes or ``None`` on EOF.
    Raises ``ValueError`` for requests with oversized headers.
    """
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = await reader.read(_READ_CHUNK)
        if not chunk:
            return None
        buf += chunk
        if len(buf) > _MAX_HEADER_SIZE:
            raise ValueError("Request headers too large")

    split = buf.index(b"\r\n\r\n")
    header_block = buf[:split]
    leftover = buf[split + 4:]

    first_line_end = header_block.index(b"\r\n")
    request_line = header_block[:first_line_end]
    raw_headers = header_block[first_line_end + 2:]

    # Read body based on Content-Length (if present)
    content_length = 0
    for line in raw_headers.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            try:
                content_length = int(line.split(b":", 1)[1].strip())
            except ValueError:
                pass
            break

    body = leftover
    remaining = content_length - len(leftover)
    while remaining > 0:
        chunk = await reader.read(min(remaining, _READ_CHUNK))
        if not chunk:
            break
        body += chunk
        remaining -= len(chunk)

    return request_line, raw_headers, body[:content_length] if content_length else body


async def _handle_connect(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    host: str,
    port: int,
) -> None:
    """Handle a CONNECT tunnel (HTTPS pass-through — no MITM in Phase 1)."""
    try:
        up_reader, up_writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=_CONNECT_TIMEOUT,
        )
    except (OSError, asyncio.TimeoutError) as exc:
        logger.warning("CONNECT %s:%d failed: %s", host, port, exc)
        client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        await client_writer.drain()
        return

    client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await client_writer.drain()

    # Blind bidirectional relay — no inspection
    await asyncio.gather(
        _pipe(client_reader, up_writer),
        _pipe(up_reader, client_writer),
        return_exceptions=True,
    )
    try:
        up_writer.close()
    except Exception:
        pass


async def _handle_http(
    client_writer: asyncio.StreamWriter,
    request_line: bytes,
    raw_headers: bytes,
    body: bytes,
    store: CredStore,
) -> None:
    """Handle a plain HTTP proxy request with credential substitution."""
    try:
        method, url, version = _parse_request_line(request_line)
    except ValueError as exc:
        logger.warning("Bad request line: %s", exc)
        client_writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        await client_writer.drain()
        return

    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or 80
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    # Substitute credentials in headers
    headers = _parse_headers(raw_headers)
    new_headers: list[tuple[str, str]] = []
    for name, value in headers:
        new_val = substitute(value, store)
        if new_val != value:
            logger.debug("Substituted credential in header %r", name)
        new_headers.append((name, new_val))

    # Strip proxy-specific hop-by-hop headers before forwarding
    new_headers = [
        (n, v) for n, v in new_headers
        if n.lower() not in ("proxy-connection", "proxy-authorization")
    ]

    # Substitute credentials in body
    new_body = body
    if body:
        try:
            text = body.decode("utf-8", errors="replace")
            new_text = substitute(text, store)
            if new_text != text:
                logger.debug("Substituted credential in request body")
                new_body = new_text.encode("utf-8")
        except Exception:
            pass

    # Recalculate Content-Length if body was rewritten
    if new_body is not body:
        new_headers = [
            (n, v) for n, v in new_headers if n.lower() != "content-length"
        ]
        new_headers.append(("Content-Length", str(len(new_body))))

    new_request_line = f"{method} {path} {version}".encode("latin-1")
    outgoing = (
        new_request_line + b"\r\n"
        + _headers_to_bytes(new_headers) + b"\r\n"
        + new_body
    )

    try:
        up_reader, up_writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=_CONNECT_TIMEOUT,
        )
    except (OSError, asyncio.TimeoutError) as exc:
        logger.warning("Connection to %s:%d failed: %s", host, port, exc)
        client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        await client_writer.drain()
        return

    up_writer.write(outgoing)
    await up_writer.drain()

    try:
        while True:
            data = await up_reader.read(_READ_CHUNK)
            if not data:
                break
            client_writer.write(data)
            await client_writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            up_writer.close()
        except Exception:
            pass


async def _handle_management(
    writer: asyncio.StreamWriter,
    method: str,
    path: str,
    body: bytes,
    store: CredStore,
) -> None:
    """Handle management API requests on the ``/_cred/`` path prefix.

    Endpoints:
      POST /_cred/add    {"name": "...", "value": "..."}  → store a credential
      GET  /_cred/list                                    → list credential names
      POST /_cred/delete  {"name": "..."}                 → delete a credential
    """
    import json

    def _respond(status: str, body_dict: dict) -> bytes:
        payload = json.dumps(body_dict).encode("utf-8")
        return (
            f"HTTP/1.1 {status}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"\r\n"
        ).encode("latin-1") + payload

    if path == "/_cred/add" and method == "POST":
        try:
            data = json.loads(body)
            name, value = data["name"], data["value"]
        except (json.JSONDecodeError, KeyError):
            writer.write(_respond("400 Bad Request", {"error": "need name and value"}))
            await writer.drain()
            return
        store.set(name, value)
        writer.write(_respond("200 OK", {"stored": True, "name": name}))
        await writer.drain()

    elif path == "/_cred/list" and method == "GET":
        writer.write(_respond("200 OK", {"names": store.list()}))
        await writer.drain()

    elif path == "/_cred/delete" and method == "POST":
        try:
            data = json.loads(body)
            name = data["name"]
        except (json.JSONDecodeError, KeyError):
            writer.write(_respond("400 Bad Request", {"error": "need name"}))
            await writer.drain()
            return
        try:
            store.delete(name)
            writer.write(_respond("200 OK", {"deleted": True, "name": name}))
        except KeyError:
            writer.write(_respond("404 Not Found", {"error": f"{name!r} not found"}))
        await writer.drain()

    else:
        writer.write(_respond("404 Not Found", {"error": "unknown management endpoint"}))
        await writer.drain()


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    store: CredStore,
) -> None:
    """Dispatch a single proxy client connection."""
    peer = writer.get_extra_info("peername", "<unknown>")
    try:
        result = await _read_request(reader)
        if result is None:
            return
        request_line, raw_headers, body = result

        try:
            method, url, _version = _parse_request_line(request_line)
        except ValueError:
            writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            await writer.drain()
            return

        # Management API — relative paths starting with /_cred/
        # (proxy requests always have absolute URLs like http://host/...)
        if url.startswith("/_cred/"):
            await _handle_management(writer, method, url, body, store)
        elif method.upper() == "CONNECT":
            host, _, port_str = url.rpartition(":")
            try:
                port = int(port_str)
            except ValueError:
                writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                await writer.drain()
                return
            await _handle_connect(reader, writer, host, port)
        else:
            await _handle_http(writer, request_line, raw_headers, body, store)

    except Exception as exc:
        logger.warning("Error handling client %s: %s", peer, exc)
    finally:
        try:
            writer.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_proxy(
    unix_socket: str | None = None,
    on_started=None,
    store: CredStore | None = None,
) -> None:
    """Start the asyncio HTTP proxy on a Unix domain socket and block until shutdown.

    Pass a *unix_socket* path (e.g. ``~/.hermes/state/cred-proxy.sock``).
    The callback ``on_started(socket_path)`` fires once the server is bound.
    """
    if store is None:
        store = CredStore()

    if unix_socket is None:
        raise ValueError("unix_socket path is required")

    import os
    # Remove stale socket file if present (e.g. after unclean shutdown)
    try:
        os.unlink(unix_socket)
    except FileNotFoundError:
        pass

    server = await asyncio.start_unix_server(
        lambda r, w: _handle_client(r, w, store),
        path=unix_socket,
    )

    # Restrict socket permissions — only the owning user can connect
    os.chmod(unix_socket, 0o600)

    async with server:
        if on_started is not None:
            on_started(unix_socket)
        await server.serve_forever()
