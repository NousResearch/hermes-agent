"""Signed asynchronous webhook callbacks (Task 13, #4386/#73828).

A route may declare a ``callback`` (an http(s) URL + optional secret). After
the agent run finishes, the adapter POSTs a signed envelope — execution ID,
event ID, terminal status, output/error, timestamp, attempt number — to that
URL. The URL is an untrusted boundary: private/link-local/metadata
destinations are rejected by default to prevent SSRF; redirects are refused;
DNS resolution is bound to the exact address dialed; delivery is bounded.

``deliver_callback`` is the synchronous transport primitive. Async gateway
callers must use ``deliver_callback_async`` so DNS, connect, TLS, response I/O,
and retry backoff execute off the event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import http.client
import ipaddress
import json
import logging
import socket
import ssl
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)

CALLBACK_TIMEOUT_SECONDS = 10
CALLBACK_MAX_ATTEMPTS = 2
CALLBACK_BACKOFF_SECONDS = 1.0


@dataclass(frozen=True)
class _CallbackDestination:
    scheme: str
    hostname: str
    port: int
    connect_host: str
    request_target: str
    host_header: str


def _unsafe_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return True
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_public_destination(url: str) -> tuple[_CallbackDestination | None, str]:
    """Resolve once, reject every unsafe answer, and return the address to dial."""
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return None, "callback URL is malformed"
    if parsed.scheme not in {"http", "https"}:
        return None, "callback must be http(s)"
    if parsed.username is not None or parsed.password is not None:
        return None, "callback URL must not contain userinfo"
    hostname = parsed.hostname
    if not hostname:
        return None, "callback missing host"
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None, "callback port is invalid"

    try:
        infos = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except OSError:
        return None, "callback host could not be resolved"
    if not infos:
        return None, "callback host could not be resolved"

    resolved: list[str] = []
    for info in infos:
        addr = str(info[4][0])
        if addr in resolved:
            continue
        if _unsafe_ip(addr):
            return None, "callback host resolves to a private/loopback/metadata address (SSRF blocked)"
        resolved.append(addr)
    if not resolved:
        return None, "callback host could not be resolved"

    default_port = 443 if parsed.scheme == "https" else 80
    host_literal = hostname
    if ":" in host_literal and not host_literal.startswith("["):
        host_literal = f"[{host_literal}]"
    host_header = host_literal if port == default_port else f"{host_literal}:{port}"
    request_target = urllib.parse.urlunsplit(
        ("", "", parsed.path or "/", parsed.query, "")
    )
    return (
        _CallbackDestination(
            scheme=parsed.scheme,
            hostname=hostname,
            port=port,
            connect_host=resolved[0],
            request_target=request_target,
            host_header=host_header,
        ),
        "",
    )


def _is_private_host(host: str) -> bool:
    """Compatibility helper: fail closed if any resolved address is unsafe."""
    try:
        infos = socket.getaddrinfo(
            host,
            None,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except OSError:
        return True
    if not infos:
        return True
    return any(_unsafe_ip(str(info[4][0])) for info in infos)


def validate_callback_url(url: str) -> tuple[bool, str]:
    """Validate that a callback URL resolves only to public http(s) addresses."""
    destination, reason = _resolve_public_destination(url)
    return destination is not None, reason


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection whose socket dials a previously validated IP address."""

    def __init__(
        self,
        hostname: str,
        port: int,
        connect_host: str,
        *,
        timeout: int,
    ) -> None:
        super().__init__(hostname, port=port, timeout=timeout)
        self._connect_host = connect_host

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._connect_host, self.port),
            self.timeout,
            self.source_address,
        )
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to an IP while preserving hostname SNI/cert checks."""

    def __init__(
        self,
        hostname: str,
        port: int,
        connect_host: str,
        *,
        timeout: int,
    ) -> None:
        super().__init__(
            hostname,
            port=port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._connect_host = connect_host

    def connect(self) -> None:
        sock = socket.create_connection(
            (self._connect_host, self.port),
            self.timeout,
            self.source_address,
        )
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
            sock = self.sock
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _open_pinned(
    destination: _CallbackDestination,
    body: bytes,
    headers: Mapping[str, str],
    *,
    timeout: int,
) -> int:
    connection_cls = (
        _PinnedHTTPSConnection
        if destination.scheme == "https"
        else _PinnedHTTPConnection
    )
    connection = connection_cls(
        destination.hostname,
        destination.port,
        destination.connect_host,
        timeout=timeout,
    )
    request_headers = dict(headers)
    request_headers["Host"] = destination.host_header
    try:
        connection.request(
            "POST",
            destination.request_target,
            body=body,
            headers=request_headers,
        )
        response = connection.getresponse()
        status = int(response.status)
        # Drain the bounded response stream enough to permit a clean close;
        # callback response bodies are not part of the transport contract.
        response.read(64 * 1024)
        return status
    finally:
        connection.close()


def _sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def build_callback_envelope(
    *,
    execution_id: str,
    event_id: str,
    status: str,
    output: str | None,
    error: str | None,
    attempt: int,
) -> dict:
    return {
        "execution_id": execution_id,
        "event_id": event_id,
        "status": status,
        "output": output,
        "error": error,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "attempt": attempt,
    }


def _valid_envelope(envelope: object) -> bool:
    if not isinstance(envelope, dict):
        return False
    execution_id = envelope.get("execution_id")
    event_id = envelope.get("event_id")
    status = envelope.get("status")
    return all(
        isinstance(value, str) and bool(value.strip())
        for value in (execution_id, event_id, status)
    )


def deliver_callback(
    url: str,
    secret: str | None,
    envelope: dict,
    *,
    timeout: int = CALLBACK_TIMEOUT_SECONDS,
) -> bool:
    """Synchronously POST a signed envelope using one validated DNS generation.

    The host is resolved exactly once for this delivery. Every returned address
    must be public, and the socket is then connected to that validated address
    directly. HTTPS still uses the configured hostname for SNI and certificate
    verification, while the original hostname is retained in ``Host``.

    Redirects are never followed: any 3xx is a terminal delivery failure.
    Async callers must invoke :func:`deliver_callback_async`.
    """
    if not _valid_envelope(envelope):
        logger.warning("[webhook] callback refused: malformed callback envelope")
        return False
    destination, reason = _resolve_public_destination(url)
    if destination is None:
        logger.warning("[webhook] callback refused: %s", reason)
        return False

    body = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Hermes-Delivery": str(envelope["execution_id"]),
    }
    if secret:
        headers["X-Hermes-Signature-256"] = _sign(body, secret)

    last_error = ""
    for attempt in range(1, CALLBACK_MAX_ATTEMPTS + 1):
        try:
            status = _open_pinned(
                destination,
                body,
                headers,
                timeout=timeout,
            )
            if 200 <= status < 300:
                return True
            last_error = f"HTTP {status}"
            if 300 <= status < 500:
                # Redirects and receiver-declared client errors are terminal.
                return False
        except Exception as exc:
            last_error = str(exc) or type(exc).__name__
        if attempt < CALLBACK_MAX_ATTEMPTS:
            time.sleep(CALLBACK_BACKOFF_SECONDS * attempt)

    logger.warning(
        "[webhook] callback delivery failed after %d attempts: %s",
        CALLBACK_MAX_ATTEMPTS,
        last_error,
    )
    return False


async def deliver_callback_async(
    url: str,
    secret: str | None,
    envelope: dict,
    *,
    timeout: int = CALLBACK_TIMEOUT_SECONDS,
) -> bool:
    """Run the blocking callback transport off the asyncio event loop."""
    return await asyncio.to_thread(
        deliver_callback,
        url,
        secret,
        envelope,
        timeout=timeout,
    )
