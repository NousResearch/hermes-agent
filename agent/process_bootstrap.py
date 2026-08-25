"""Process-level bootstrap helpers for ``run_agent``.

Three concerns, all tied to ``AIAgent`` boot-time / runtime IO setup:

1. **Lazy OpenAI SDK import** — ``_load_openai_cls`` + ``_OpenAIProxy``
   defer the 240ms-ish ``from openai import OpenAI`` cost until first use,
   while preserving ``isinstance(client, OpenAI)`` checks and
   ``patch("run_agent.OpenAI", ...)`` test patterns.

2. **Crash-resistant stdio** — ``_SafeWriter`` wraps stdout/stderr so
   ``OSError: Input/output error`` from broken pipes (systemd, Docker,
   thread teardown races) cannot crash the agent.  ``_install_safe_stdio``
   applies the wrapper.

3. **HTTP proxy resolution** — ``_get_proxy_from_env`` reads
   ``HTTPS_PROXY`` / ``HTTP_PROXY`` / ``ALL_PROXY``;
   ``_get_proxy_for_base_url`` respects ``NO_PROXY`` for the given base URL.
4. **Codex dual-stack resilience** — the synchronous ChatGPT/Codex transport
   races resolved IPv6/IPv4 addresses so a blackholed family cannot exhaust
   the request watchdog before a working address is attempted.

``run_agent`` re-exports every name so existing
``from run_agent import _get_proxy_from_env`` imports keep working
unchanged.
"""

from __future__ import annotations

import errno
import os
import selectors
import socket
import sys
import time
import urllib.request
from typing import Any, Optional

from utils import base_url_hostname, normalize_proxy_url
from agent.proxy_bypass import first_proxy_env_value, should_bypass_proxy


_OPENAI_CLS_CACHE = None
_HAPPY_EYEBALLS_DELAY_SECONDS = 0.25


def _interleave_addrinfos(addrinfos: list[tuple]) -> list[tuple]:
    """Interleave resolved address families while preserving resolver order."""
    queues: dict[int, list[tuple]] = {}
    family_order: list[int] = []
    seen: set[tuple] = set()
    for addrinfo in addrinfos:
        family, socktype, proto, _canonname, sockaddr = addrinfo
        marker = (family, socktype, proto, sockaddr)
        if marker in seen:
            continue
        seen.add(marker)
        if family not in queues:
            queues[family] = []
            family_order.append(family)
        queues[family].append(addrinfo)

    interleaved: list[tuple] = []
    while any(queues.values()):
        for family in family_order:
            if queues[family]:
                interleaved.append(queues[family].pop(0))
    return interleaved


def _happy_eyeballs_create_connection(
    address: tuple[str, int],
    timeout: Optional[float],
    source_address: Optional[tuple[str, int]] = None,
    socket_options=(),
):
    """Connect using staggered non-blocking attempts across resolved families.

    ``socket.create_connection`` tries every address serially. A host with
    broken-but-advertised IPv6 can therefore consume the full connect timeout
    for each AAAA record before trying a working IPv4 address. This follows the
    Happy Eyeballs shape from RFC 8305: retain resolver preference, interleave
    families, and start the next candidate after a short delay.
    """
    host, port = address
    addrinfos = _interleave_addrinfos(
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    )
    if not addrinfos:
        raise OSError(f"getaddrinfo returned no addresses for {host}")

    selector = selectors.DefaultSelector()
    active: set[socket.socket] = set()
    winner = None
    last_error: Optional[OSError] = None
    deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
    next_launch = time.monotonic()
    pending = list(addrinfos)
    in_progress = {
        0,
        errno.EINPROGRESS,
        errno.EWOULDBLOCK,
        errno.EALREADY,
        errno.EINTR,
        getattr(errno, "WSAEWOULDBLOCK", 10035),
    }

    def start_attempt(addrinfo):
        family, socktype, proto, _canonname, sockaddr = addrinfo
        candidate = socket.socket(family, socktype, proto)
        try:
            if source_address is not None:
                local_infos = socket.getaddrinfo(
                    source_address[0],
                    source_address[1],
                    family=family,
                    type=socktype,
                )
                if not local_infos:
                    raise OSError(
                        f"getaddrinfo returned no local {family} address for "
                        f"{source_address[0]}"
                    )
                candidate.bind(local_infos[0][4])
            candidate.setblocking(False)
            result = candidate.connect_ex(sockaddr)
            if result == 0 or result == errno.EISCONN:
                return candidate
            if result not in in_progress:
                raise OSError(result, os.strerror(result))
            selector.register(candidate, selectors.EVENT_WRITE)
            active.add(candidate)
            return None
        except Exception:
            candidate.close()
            raise

    try:
        while pending or active:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise socket.timeout("timed out")

            if pending and now >= next_launch:
                addrinfo = pending.pop(0)
                try:
                    winner = start_attempt(addrinfo)
                except OSError as exc:
                    last_error = exc
                    if not active:
                        next_launch = now
                    continue
                if winner is not None:
                    break
                next_launch = now + _HAPPY_EYEBALLS_DELAY_SECONDS

            wait_timeout = None if deadline is None else max(0.0, deadline - now)
            if pending:
                until_launch = max(0.0, next_launch - now)
                wait_timeout = (
                    until_launch
                    if wait_timeout is None
                    else min(wait_timeout, until_launch)
                )

            events = selector.select(wait_timeout)
            for key, _mask in events:
                candidate = key.fileobj
                error_code = candidate.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                selector.unregister(candidate)
                active.discard(candidate)
                if error_code == 0:
                    winner = candidate
                    break
                candidate.close()
                last_error = OSError(error_code, os.strerror(error_code))
            if winner is not None:
                break
            if not active and pending:
                next_launch = time.monotonic()

        if winner is None:
            if last_error is not None:
                raise last_error
            raise OSError(f"Could not connect to {host}:{port}")

        try:
            selector.unregister(winner)
        except Exception:
            pass
        active.discard(winner)
        winner.settimeout(timeout)
        for option in socket_options or ():
            winner.setsockopt(*option)
        winner.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return winner
    finally:
        for candidate in active:
            try:
                selector.unregister(candidate)
            except Exception:
                pass
            candidate.close()
        selector.close()


class _HappyEyeballsSyncBackend:
    """httpcore sync backend with concurrent IPv6/IPv4 connection fallback."""

    def __init__(self):
        self._fallback = None

    def _default_backend(self):
        if self._fallback is None:
            from httpcore import SyncBackend

            self._fallback = SyncBackend()
        return self._fallback

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        local_address: Optional[str] = None,
        socket_options=None,
    ):
        from httpcore import ConnectError, ConnectTimeout
        from httpcore._backends.sync import SyncStream

        source_address = None if local_address is None else (local_address, 0)
        try:
            sock = _happy_eyeballs_create_connection(
                (host, port),
                timeout,
                source_address=source_address,
                socket_options=socket_options or (),
            )
        except socket.timeout as exc:
            raise ConnectTimeout(str(exc)) from exc
        except OSError as exc:
            raise ConnectError(str(exc)) from exc
        return SyncStream(sock)

    def connect_unix_socket(self, *args, **kwargs):
        return self._default_backend().connect_unix_socket(*args, **kwargs)

    def sleep(self, seconds: float) -> None:
        self._default_backend().sleep(seconds)


def _uses_codex_cloud_transport(base_url: str) -> bool:
    return (
        base_url_hostname(base_url).lower() == "chatgpt.com"
        and "/backend-api/codex" in str(base_url).lower()
    )


def _enable_happy_eyeballs(transport) -> None:
    """Install the sync racing backend on one httpx transport, if compatible.

    Reaches into httpx/httpcore private attributes (``transport._pool`` /
    ``pool._network_backend``); safe because httpcore is pinned (1.0.x) and
    both lookups are hasattr-guarded — on an incompatible httpcore this
    degrades to the default serial backend instead of crashing.
    """
    pool = getattr(transport, "_pool", None)
    if pool is not None and hasattr(pool, "_network_backend"):
        pool._network_backend = _HappyEyeballsSyncBackend()


def _load_openai_cls() -> type:
    """Import and cache ``openai.OpenAI``."""
    global _OPENAI_CLS_CACHE
    if _OPENAI_CLS_CACHE is None:
        from openai import OpenAI as _OPENAI_CLS_CACHE
    return _OPENAI_CLS_CACHE


class _OpenAIProxy:
    """Module-level proxy that looks like ``openai.OpenAI`` but imports lazily."""

    __slots__ = ()

    def __call__(self, *args, **kwargs):
        return _load_openai_cls()(*args, **kwargs)

    def __instancecheck__(self, obj):
        return isinstance(obj, _load_openai_cls())

    def __repr__(self):
        return "<lazy openai.OpenAI proxy>"


class _SafeWriter:
    """Transparent stdio wrapper swallowing OSError/ValueError from broken pipes.

    Headless runs (systemd, Docker) lose the stdout pipe → ``OSError: [Errno 5]``;
    subagent threads can see the shared handle close → ``ValueError``. Either
    would otherwise crash the agent (often via double-fault in an except handler).
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def write(self, data):
        try:
            return self._inner.write(data)
        except (OSError, ValueError):
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass

    def fileno(self):
        return self._inner.fileno()

    def isatty(self):
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _get_proxy_from_env() -> Optional[str]:
    """First configured proxy URL from HTTPS_PROXY / HTTP_PROXY / ALL_PROXY (any case), or None."""
    value = first_proxy_env_value()
    return normalize_proxy_url(value) if value else None


def _get_proxy_for_base_url(base_url: Optional[str]) -> Optional[str]:
    """Env-configured proxy unless NO_PROXY excludes this base URL (same matcher as the
    gateway adapters: CIDR, ``*.`` wildcards and host:port entries all count)."""
    proxy = _get_proxy_from_env()
    if not (proxy and base_url):
        return proxy
    raw = base_url.strip()
    return None if should_bypass_proxy(raw if "://" in raw else f"//{raw}") else proxy


def _shared_transport_cls():
    """Lazily define the per-client transport view (httpx import is deferred)."""
    global _SharedTransport
    if _SharedTransport is not None:
        return _SharedTransport
    import httpx

    class _SharedTransportImpl(httpx.BaseTransport):
        """Per-client view of a process-shared ``httpx.HTTPTransport``.

        ``httpx.Client.close()`` closes every mounted transport, and each OpenAI client still
        owns its own ``httpx.Client`` (closing one client must never poison the next), so the
        mounted object absorbs that close while the shared pool keeps serving other clients.
        ``handle_request`` stamps the owning view into ``request.extensions`` so socket-abort
        sweeps target only this client's in-flight connections on the shared pool.

        See #10933.
        """

        __slots__ = ("_inner", "_closed")

        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self._closed = False

        @property
        def _pool(self) -> Any:  # httpx-private; socket walkers and tests introspect it
            return getattr(self._inner, "_pool", None)

        def handle_request(self, request: Any) -> Any:
            if self._closed:
                raise RuntimeError("Cannot send a request, as the client has been closed.")
            request.extensions[HERMES_TRANSPORT_OWNER_EXT] = id(self)
            return self._inner.handle_request(request)

        def close(self) -> None:
            # Never closes the shared ``_inner``; idle connections are reaped by keepalive_expiry
            # and the pool lives for the process (see ``close_shared_transports``).
            self._closed = True

    _SharedTransportImpl.__name__ = _SharedTransportImpl.__qualname__ = "_SharedTransport"
    _SharedTransport = _SharedTransportImpl
    return _SharedTransport


_SharedTransport: Any = None


def _shared_transport_key(base_url: str, verify: Any, proxy: Optional[str]) -> tuple:
    """Identity under which sync direct transports are pooled process-wide."""
    if verify is True or verify is False:
        verify_key: Any = verify
    elif isinstance(verify, str):
        verify_key = ("path", verify)
    else:
        verify_key = ("id", id(verify))  # SSLContext / custom object: share by identity only
    return (verify_key, proxy, _uses_codex_cloud_transport(base_url))


def _get_shared_transport(key: tuple, build) -> Any:
    with _SHARED_TRANSPORTS_LOCK:
        transport = _SHARED_TRANSPORTS.get(key)
        if transport is None:
            transport = build()
            if len(_SHARED_TRANSPORTS) < _SHARED_TRANSPORTS_MAX:
                _SHARED_TRANSPORTS[key] = transport
        return transport


def close_shared_transports() -> int:
    """Really close every process-shared transport (test teardown / atexit)."""
    with _SHARED_TRANSPORTS_LOCK:
        transports = list(_SHARED_TRANSPORTS.values())
        _SHARED_TRANSPORTS.clear()
    for transport in transports:
        try:
            transport.close()
        except Exception:
            pass
    return len(transports)


def build_keepalive_http_client(base_url: str = "", *, async_mode: bool = False, verify: Any = True) -> Optional[Any]:
    """httpx client for OpenAI SDK calls with env-only proxy policy (None on failure).

    Explicit no-proxy mounts disable httpx's ``trust_env`` path so macOS system
    proxies (which omit the ExceptionsList) are never applied. ``keepalive_expiry``
    reaps idle connections before reverse proxies' 30-60 s timeouts (a custom
    socket_options transport broke streaming and stripped TCP_NODELAY). ``verify``
    goes on the client AND the mounts, since a mounted transport owns its SSL context.

    Every call returns a NEW ``httpx.Client`` (per-client close semantics), but sync clients
    with the same (verify, proxy, happy-eyeballs) identity mount the SAME underlying
    ``HTTPTransport`` through a ``_SharedTransport`` view, so N delegated children share one
    connection pool + SSL context. Async clients are never shared: an httpcore async pool is
    bound to the event loop that first used it. Proxy-backed clients keep httpx's own transport.

    See #12952, #54049.
    See #10933.
    """
    try:
        import httpx
        proxy = _get_proxy_for_base_url(base_url)
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100, keepalive_expiry=20.0)
        timeout = httpx.Timeout(connect=15.0, read=None, write=15.0, pool=10.0)  # read=None for SSE streaming
        transport_cls = httpx.AsyncHTTPTransport if async_mode else httpx.HTTPTransport
        client_cls = httpx.AsyncClient if async_mode else httpx.Client
        mounts = None
        if proxy is None:
            http_transport = transport_cls(verify=verify)
            https_transport = transport_cls(verify=verify)
            if not async_mode and _uses_codex_cloud_transport(base_url):
                _enable_happy_eyeballs(http_transport)
                _enable_happy_eyeballs(https_transport)
            mounts = {"http://": http_transport, "https://": https_transport}
        return client_cls(
            limits=limits,
            timeout=timeout,
            proxy=proxy,
            mounts=mounts or None,
            verify=verify,
        )
    except Exception:
        return None


def _install_safe_stdio() -> None:
    """Wrap stdout/stderr so best-effort console output cannot crash the agent."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not isinstance(stream, _SafeWriter):
            setattr(sys, stream_name, _SafeWriter(stream))


# Drop-in for ``openai.OpenAI``.
OpenAI = _OpenAIProxy()


__all__ = [
    "OpenAI", "_OpenAIProxy", "_load_openai_cls", "_SafeWriter", "_install_safe_stdio", "_get_proxy_from_env",
    "_get_proxy_for_base_url", "build_keepalive_http_client", "close_shared_transports",
    "enable_happy_eyeballs_on_client",
]
