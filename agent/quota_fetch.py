"""Host-side fetcher for provider-sanctioned quota counters.

Currently the only implemented source is OpenRouter's documented key-info
endpoint (``official_key_limits``). Everything else is reported as unavailable:
no endpoint is guessed, no quota is invented, and failure is never silent.

Hard boundaries (non-negotiable):

- runs with the host-held credential of the *pool entry bound to the requested
  opaque scope* — never a different account, never a credential the plugin
  supplied;
- 2 second wall-clock bound including failure, no retries;
- response body capped (64 KiB) and only the documented counter object is read;
- redirects off the documented host are not followed;
- the credential is never returned, logged, or placed in an exception message;
- the result carries counters only — remaining + used == limit, or unavailable.
"""

from __future__ import annotations

import http.client
import json
import logging
import math
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
OFFICIAL_KEY_LIMITS = "official_key_limits"
DEFAULT_TIMEOUT = 2.0
MAX_TIMEOUT = 2.0
MAX_BODY_BYTES = 64 * 1024
OBSERVATION_TTL = 120.0  # matches the consumer's accepted observation freshness

_OPENROUTER_HOSTS = {"openrouter.ai", "www.openrouter.ai"}


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "provider_id": "openrouter", "quota_source": OFFICIAL_KEY_LIMITS,
            "reason": reason}


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


_opener = urllib.request.build_opener(_RefuseRedirect())


class _DeadlineExceeded(Exception):
    """Internal: the single total operation budget expired."""


class _DeadlineSocket(socket.socket):
    """Socket whose every blocking I/O call is bounded by the *remaining* total budget.

    A socket inactivity timeout alone cannot bound a trickling peer: each
    fragment resets it. Refreshing it to the absolute remaining budget before
    EACH syscall makes DNS-independent connect/header/body I/O honor the total
    deadline; no background worker is involved.
    """

    deadline: float = 0.0

    def _apply_deadline(self) -> None:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise _DeadlineExceeded()
        self.settimeout(remaining)

    def connect(self, address):  # noqa: ANN001,ANN201
        self._apply_deadline()
        return super().connect(address)

    def recv(self, bufsize: int, flags: int = 0) -> bytes:
        self._apply_deadline()
        return super().recv(bufsize, flags)

    def recv_into(self, buffer, nbytes: int = 0, flags: int = 0):  # noqa: ANN001,ANN201
        self._apply_deadline()
        return super().recv_into(buffer, nbytes, flags)

    def send(self, data, flags: int = 0) -> int:  # noqa: ANN001
        self._apply_deadline()
        return super().send(data, flags)

    def sendall(self, data, flags: int = 0) -> None:  # noqa: ANN001
        view = memoryview(data)
        while view:
            self._apply_deadline()
            sent = super().send(view, flags)
            view = view[sent:]


def _bind_deadline_to_socket(sock, deadline: float):  # noqa: ANN001,ANN201
    """Bind EVERY blocking operation on *sock* to the absolute remaining budget.

    ``ssl.SSLSocket`` is a different object from the raw ``_DeadlineSocket``
    handed to ``wrap_socket()``; its encrypted reads go through SSL-level
    methods, so the raw subclass overrides alone are not sufficient. Instance
    wrappers refresh both the socket and SSL timeouts before each operation.
    """
    original = {
        name: getattr(sock, name)
        for name in ("recv", "recv_into", "send", "sendall", "read", "readinto")
        if getattr(sock, name, None) is not None
    }

    def _refresh() -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _DeadlineExceeded()
        sock.settimeout(remaining)

    def _wrap(name, call):
        def wrapped(*args, **kwargs):  # noqa: ANN002,ANN003,ANN201
            _refresh()
            return call(*args, **kwargs)
        return wrapped

    for name, call in original.items():
        setattr(sock, name, _wrap(name, call))
    return sock


def _bounded_getaddrinfo(host: str, port: int, deadline: float):
    """Resolve under the remaining aggregate budget; refuse before connecting on timeout.

    The system resolver is not cancellable from Python; the helper thread is a
    short-lived daemon and performs NO provider request. If it outlives the
    budget we refuse before any socket is opened, so no provider work continues
    on a worker owned by this fetch.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _DeadlineExceeded()
    result: dict[str, Any] = {}

    def _resolve() -> None:
        try:
            result["value"] = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except Exception as exc:  # health: allow BLE001 -- surfaced to caller as unavailable
            result["error"] = exc

    thread = threading.Thread(target=_resolve, daemon=True, name="quota-dns-bound")
    thread.start()
    thread.join(remaining)
    if thread.is_alive():
        raise _DeadlineExceeded()
    if "error" in result:
        raise result["error"]
    return result["value"]


class _DeadlineHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection whose socket and TLS handshake obey the total deadline."""

    def __init__(self, host: str, *, deadline: float, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self.deadline = deadline

    def connect(self) -> None:
        infos = _bounded_getaddrinfo(self.host, self.port, self.deadline)
        family, socktype, proto, _, sockaddr = infos[0]
        raw = _DeadlineSocket(family, socktype, proto)
        raw.deadline = self.deadline
        raw.connect(sockaddr)
        # Always build the verifying default context here so the TLS handshake
        # is owned by the deadline path (and is never weakened by caller state).
        # The handshake itself is driven explicitly so its reads are covered.
        context = ssl.create_default_context()
        tls = context.wrap_socket(raw, server_hostname=self.host, do_handshake_on_connect=False)
        _bind_deadline_to_socket(tls, self.deadline)
        tls.do_handshake()
        self.sock = tls
        self._tls_established = True


class _DeadlineHTTPConnection(http.client.HTTPConnection):
    """Plain-HTTP counterpart used by offline loopback tests."""

    def __init__(self, host: str, *, deadline: float, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self.deadline = deadline

    def connect(self) -> None:
        infos = _bounded_getaddrinfo(self.host, self.port, self.deadline)
        family, socktype, proto, _, sockaddr = infos[0]
        raw = _DeadlineSocket(family, socktype, proto)
        raw.deadline = self.deadline
        raw.connect(sockaddr)
        self.sock = raw


class _DeadlineHTTPSHandler(urllib.request.HTTPSHandler):
    """urllib handler that builds deadline-bound HTTPS connections."""

    def __init__(self, deadline: float) -> None:
        super().__init__()
        self.deadline = deadline

    def https_open(self, req):  # noqa: ANN001,ANN201
        return self.do_open(
            lambda host, **kw: _DeadlineHTTPSConnection(host, deadline=self.deadline, **kw), req)


class _DeadlineHTTPHandler(urllib.request.HTTPHandler):
    """urllib handler that builds deadline-bound plain-HTTP connections."""

    def __init__(self, deadline: float) -> None:
        super().__init__()
        self.deadline = deadline

    def http_open(self, req):  # noqa: ANN001,ANN201
        return self.do_open(
            lambda host, **kw: _DeadlineHTTPConnection(host, deadline=self.deadline, **kw), req)


def _http_get_json(url: str, token: str, timeout: float) -> tuple[int, Any]:
    """GET *url* with *token*; returns (status, parsed-json|None).

    ``timeout`` is a TOTAL monotonic budget covering connect, response headers
    and every body read. urllib's socket timeout is only an inactivity bound:
    bytes that trickle in can outlast it, so the deadline is enforced here as
    well and the socket timeout is refreshed to the remaining budget before
    each read. No worker thread is started, so a refusal leaves no background
    request behind.
    """
    deadline = time.monotonic() + max(float(timeout), 0.0)
    req = urllib.request.Request(
        url, method="GET",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _DeadlineExceeded()
    opener = urllib.request.build_opener(
        _DeadlineHTTPHandler(deadline), _DeadlineHTTPSHandler(deadline), _RefuseRedirect())
    with opener.open(req, timeout=remaining) as resp:
        host = urllib.request.urlparse(resp.geturl()).hostname or ""
        if host.lower() not in _OPENROUTER_HOSTS:
            return 502, None
        sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)
        chunks: list[bytes] = []
        total = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _DeadlineExceeded()
            if sock is not None:
                try:
                    sock.settimeout(remaining)
                except Exception:  # health: allow BLE001 -- best effort, monotonic check still bounds us
                    pass
            want = min(4096, MAX_BODY_BYTES + 1 - total)
            # http.client's read(n) loops internally until n bytes arrive, so a
            # slow trickle can outlast our per-iteration deadline check. read1
            # returns at most one raw read, letting the monotonic check run
            # between network reads.
            read1 = getattr(getattr(resp, "fp", None), "read1", None)
            chunk = read1(want) if callable(read1) else resp.read(min(1, want))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                return 502, None
        raw = b"".join(chunks)
        try:
            return int(resp.status), json.loads(raw.decode("utf-8", "replace"))
        except (ValueError, UnicodeDecodeError):
            return int(resp.status), None


def _int_ge_zero(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or value != int(value):
        return None
    return int(value)


def fetch_openrouter_key_limits(token: str, *, credential_scope_id: str,
                                timeout: float = DEFAULT_TIMEOUT) -> dict:
    """The one implemented quota source. Counters only; anything off-contract is unavailable."""
    effective_timeout = min(max(float(timeout) if timeout else DEFAULT_TIMEOUT, 0.5), MAX_TIMEOUT)
    started = time.time()
    try:
        status, body = _http_get_json(OPENROUTER_KEY_URL, token, effective_timeout)
    except (_DeadlineExceeded, TimeoutError):
        # socket.timeout is a TimeoutError; the refreshed per-read socket
        # timeout is the mechanism that enforces the total monotonic budget.
        logger.debug("quota_fetch: openrouter key endpoint exceeded the total deadline")
        return _unavailable("fetch_timeout")
    except urllib.error.URLError as exc:
        # urllib wraps OSErrors from connect/TLS/headers in URLError.
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (_DeadlineExceeded, TimeoutError)):
            logger.debug("quota_fetch: openrouter key endpoint exceeded the total deadline")
            return _unavailable("fetch_timeout")
        logger.debug("quota_fetch: openrouter key endpoint unavailable")
        return _unavailable("fetch_failed")
    except Exception:  # health: allow BLE001 -- network/TLS/HTTP failures are all 'unavailable'; token never logged
        logger.debug("quota_fetch: openrouter key endpoint unavailable")
        return _unavailable("fetch_failed")
    if status != 200:
        return _unavailable("endpoint_status_%d" % status)
    data = body.get("data") if isinstance(body, dict) else None
    row = data.get("free_model_daily_requests") if isinstance(data, dict) else None
    limit = _int_ge_zero(row.get("limit")) if isinstance(row, dict) else None
    remaining = _int_ge_zero(row.get("remaining")) if isinstance(row, dict) else None
    used = _int_ge_zero(row.get("used")) if isinstance(row, dict) else None
    if limit is None or remaining is None or used is None or remaining + used != limit:
        return _unavailable("unexpected_response_shape")
    now = time.time()
    next_utc_midnight = (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                         .timestamp() + 24 * 3600)
    return {
        "status": "ok",
        "provider_id": "openrouter",
        "credential_scope_id": credential_scope_id,
        "quota_source": OFFICIAL_KEY_LIMITS,
        "observed_at": now,
        "expires_at": min(started + OBSERVATION_TTL, next_utc_midnight),
        "limit": limit,
        "remaining": remaining,
        "used": used,
    }
