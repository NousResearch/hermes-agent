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
import threading
import time
import urllib.request
from typing import Any, Optional

from utils import base_url_hostname, normalize_proxy_url


# Cached at module level so we only pay the OpenAI SDK import cost once
# per process (after the first lazy load).
_OPENAI_CLS_CACHE = None
_HAPPY_EYEBALLS_DELAY_SECONDS = 0.25

# Process-wide pool of sync ``httpx.HTTPTransport`` objects shared by every
# keepalive client with the same (verify, proxy, happy-eyeballs) identity.
# Each delegated child AIAgent used to get its own transport = its own TLS
# pool, so a fan-out of N children held N separate socket sets to the same
# provider. Bounded: past the cap, callers get a private transport again.
_SHARED_TRANSPORTS: dict[tuple, Any] = {}
_SHARED_TRANSPORTS_LOCK = threading.Lock()
_SHARED_TRANSPORTS_MAX = 32
# ``request.extensions`` key stamped by ``_SharedTransport.handle_request``;
# the socket-abort walker in agent_runtime_helpers uses it to find only the
# owning client's in-flight connections on a shared pool.
HERMES_TRANSPORT_OWNER_EXT = "hermes_transport_owner"


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


def enable_happy_eyeballs_on_client(client) -> None:
    """Install the sync racing backend on every direct transport of a client.

    Covers a ready-built ``httpx.Client`` (its default transport plus any
    mounts), for callers that construct clients inline instead of going
    through :func:`build_keepalive_http_client` — e.g. the Codex OAuth token
    refresh / device-login / usage-probe clients in ``hermes_cli.auth``.

    Proxy-backed transports (``httpcore.HTTPProxy`` / SOCKS pools) are left
    untouched: with a proxy in play the TCP connect goes to the proxy host,
    which is out of scope for the direct-transport racing added in #94388.
    Async clients are also left untouched — httpcore's async backend already
    performs RFC 8305 racing natively via
    ``anyio.connect_tcp(happy_eyeballs_delay=0.25)``.

    Best-effort and hasattr-guarded like ``_enable_happy_eyeballs``; on an
    incompatible httpx/httpcore this silently keeps the default backend.
    """
    try:
        import httpcore

        proxy_pool_types = tuple(
            t
            for t in (
                getattr(httpcore, "HTTPProxy", None),
                getattr(httpcore, "SOCKSProxy", None),
            )
            if t is not None
        )
    except Exception:
        return

    transports = [getattr(client, "_transport", None)]
    transports.extend((getattr(client, "_mounts", None) or {}).values())
    for transport in transports:
        pool = getattr(transport, "_pool", None)
        if pool is None or not hasattr(pool, "_network_backend"):
            continue
        if proxy_pool_types and isinstance(pool, proxy_pool_types):
            continue
        pool._network_backend = _HappyEyeballsSyncBackend()


def _load_openai_cls() -> type:
    """Import and cache ``openai.OpenAI``."""
    global _OPENAI_CLS_CACHE
    if _OPENAI_CLS_CACHE is None:
        from openai import OpenAI as _cls
        _OPENAI_CLS_CACHE = _cls
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
    """Transparent stdio wrapper that catches OSError/ValueError from broken pipes.

    When hermes-agent runs as a systemd service, Docker container, or headless
    daemon, the stdout/stderr pipe can become unavailable (idle timeout, buffer
    exhaustion, socket reset). Any print() call then raises
    ``OSError: [Errno 5] Input/output error``, which can crash agent setup or
    run_conversation() — especially via double-fault when an except handler
    also tries to print.

    Additionally, when subagents run in ThreadPoolExecutor threads, the shared
    stdout handle can close between thread teardown and cleanup, raising
    ``ValueError: I/O operation on closed file`` instead of OSError.

    This wrapper delegates all writes to the underlying stream and silently
    catches both OSError and ValueError. It is transparent when the wrapped
    stream is healthy.
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
    """Read proxy URL from environment variables.

    Checks HTTPS_PROXY, HTTP_PROXY, ALL_PROXY (and lowercase variants) in order.
    Returns the first valid proxy URL found, or None if no proxy is configured.
    """
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        value = os.environ.get(key, "").strip()
        if value:
            return normalize_proxy_url(value)
    return None


def _get_proxy_for_base_url(base_url: Optional[str]) -> Optional[str]:
    """Return an env-configured proxy unless NO_PROXY excludes this base URL."""
    proxy = _get_proxy_from_env()
    if not proxy or not base_url:
        return proxy

    host = base_url_hostname(base_url)
    if not host:
        return proxy

    try:
        if urllib.request.proxy_bypass_environment(host):
            return None
    except Exception:
        pass

    return proxy


def _shared_transport_cls():
    """Lazily define the per-client transport view (httpx import is deferred)."""
    global _SharedTransport
    if _SharedTransport is not None:
        return _SharedTransport
    import httpx

    class _SharedTransportImpl(httpx.BaseTransport):
        """Per-client view of a process-shared ``httpx.HTTPTransport``.

        ``httpx.Client.close()`` closes every mounted transport. Each OpenAI
        client still owns its own ``httpx.Client`` (the #10933 contract:
        closing one client must never poison the next), so the object we
        mount must absorb that close while the underlying connection pool
        keeps serving every other client. ``handle_request`` stamps the
        owning view into ``request.extensions`` so socket-abort sweeps can
        target only this client's in-flight connections on the shared pool.
        """

        __slots__ = ("_inner", "_closed")

        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self._closed = False

        # httpx-private ``_pool`` is what our socket walkers and the
        # happy-eyeballs / ssl-verify tests introspect: expose the shared one.
        @property
        def _pool(self) -> Any:
            return getattr(self._inner, "_pool", None)

        def handle_request(self, request: Any) -> Any:
            if self._closed:
                raise RuntimeError("Cannot send a request, as the client has been closed.")
            request.extensions[HERMES_TRANSPORT_OWNER_EXT] = id(self)
            return self._inner.handle_request(request)

        def close(self) -> None:
            # Deliberately does NOT close ``_inner``: it is shared. Idle
            # connections are reaped by ``keepalive_expiry``; the pool lives
            # for the process (see ``close_shared_transports``).
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
        # An ssl.SSLContext (or custom object): share only by object identity,
        # which is what a caller passing the same context twice expects.
        verify_key = ("id", id(verify))
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


def build_keepalive_http_client(
    base_url: str = "",
    *,
    async_mode: bool = False,
    verify: Any = True,
) -> Optional[Any]:
    """Build an httpx client for OpenAI SDK calls with env-only proxy policy.

    Uses explicit ``HTTPS_PROXY`` / ``NO_PROXY`` env vars via
    ``_get_proxy_for_base_url``. Plain no-proxy mounts disable httpx's default
    ``trust_env`` proxy path, so macOS system proxy settings from
    ``urllib.request.getproxies()`` (which omit the ExceptionsList) are not
    applied. Mirrors ``AIAgent._build_keepalive_http_client``.

    Connection lifecycle is managed at the HTTP pool layer
    (``keepalive_expiry=20.0`` reaps idle connections before reverse proxies'
    typical 30-60 s timeouts) instead of the former custom
    ``socket_options`` transport, which broke streaming behind reverse
    proxies (#54049, #12952) and stalled TLS handshakes by stripping
    ``TCP_NODELAY``.

    ``verify`` is forwarded to httpx so auxiliary-client calls (compression,
    vision, web_extract, title generation, etc.) honor the same per-provider
    ``ssl_ca_cert`` / ``ssl_verify`` and ``HERMES_CA_BUNDLE`` settings the main
    client uses. It is passed on the client AND on the plain no-proxy mounts
    (a mounted transport owns the SSL context for its scheme).

    Every call returns a NEW ``httpx.Client`` (per-client close semantics are
    what #10933 pins), but sync clients with the same
    (verify, proxy, happy-eyeballs) identity mount the SAME underlying
    ``HTTPTransport`` through a :class:`_SharedTransport` view, so N delegated
    children share one connection pool + SSL context instead of N. Async
    clients are never shared: an httpcore async pool is bound to the event
    loop that first used it.
    """
    try:
        import httpx

        proxy = _get_proxy_for_base_url(base_url)

        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=20.0,
        )
        # Generous read=None for SSE streaming endpoints.
        timeout = httpx.Timeout(connect=15.0, read=None, write=15.0, pool=10.0)

        transport_cls = httpx.AsyncHTTPTransport if async_mode else httpx.HTTPTransport
        client_cls = httpx.AsyncClient if async_mode else httpx.Client
        mounts = {}
        if proxy is None:
            happy_eyeballs = not async_mode and _uses_codex_cloud_transport(base_url)
            # One pool now serves every agent in the process, so its ceiling
            # must cover a whole fan-out of concurrently streaming children,
            # not one client. (Note: previously the mounts silently ran on
            # httpx defaults — keepalive_expiry=5s — since Client-level
            # ``limits`` only reach the default transport.)
            direct_limits = limits if async_mode else httpx.Limits(
                max_keepalive_connections=50,
                max_connections=1000,
                keepalive_expiry=20.0,
            )

            def _build_direct():
                transport = transport_cls(verify=verify, limits=direct_limits)
                # Async transports need no explicit racing: httpcore's anyio
                # backend already implements RFC 8305 natively
                # (``anyio.connect_tcp(happy_eyeballs_delay=0.25)``), covered
                # by tests/agent/test_codex_happy_eyeballs.py.
                if happy_eyeballs:
                    _enable_happy_eyeballs(transport)
                return transport

            if async_mode:
                mounts = {"http://": _build_direct(), "https://": _build_direct()}
            else:
                key = _shared_transport_key(base_url, verify, proxy)
                view_cls = _shared_transport_cls()
                mounts = {
                    f"{scheme}://": view_cls(
                        _get_shared_transport((scheme, *key), _build_direct)
                    )
                    for scheme in ("http", "https")
                }
                # Without this httpx builds a third, never-used direct
                # transport (and pool + SSL context) per client.
                return client_cls(
                    limits=limits,
                    timeout=timeout,
                    transport=mounts["https://"],
                    mounts=mounts,
                )
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


# Module-level proxy instance — drops in for ``openai.OpenAI``.  Imported as
# ``from agent.process_bootstrap import OpenAI`` (or re-exported via
# ``run_agent`` for legacy tests).
OpenAI = _OpenAIProxy()


__all__ = [
    "OpenAI",
    "_OpenAIProxy",
    "_load_openai_cls",
    "_SafeWriter",
    "_install_safe_stdio",
    "_get_proxy_from_env",
    "_get_proxy_for_base_url",
    "build_keepalive_http_client",
    "close_shared_transports",
    "enable_happy_eyeballs_on_client",
]
