"""Process-level bootstrap helpers for ``run_agent``.

Lazy OpenAI SDK import (``_OpenAIProxy`` keeps ``isinstance`` and
``patch("agent.process_bootstrap.OpenAI")`` working), crash-resistant stdio
(``_SafeWriter``), env-only HTTP proxy resolution, and the httpcore backend that
runs sync httpx connects through the process-wide Happy Eyeballs racer
(``hermes_bootstrap``), and the optional ``network.tls_max_version`` handshake
cap applied to every keepalive client.
"""

from __future__ import annotations

import functools
import socket
import sys
import threading
from typing import Any, Optional

from hermes_bootstrap import _happy_eyeballs_create_connection
from utils import base_url_hostname, normalize_proxy_url
from agent.proxy_bypass import first_proxy_env_value, should_bypass_proxy
from agent.thread_scoped_output import _ThreadRoutingStream


_OPENAI_CLS_CACHE = None

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


class _HappyEyeballsSyncBackend:
    """httpcore sync backend with concurrent IPv6/IPv4 connection fallback."""

    def __init__(self):
        self._fallback = None

    def _default_backend(self):
        if self._fallback is None:
            from httpcore import SyncBackend
            self._fallback = SyncBackend()
        return self._fallback

    def connect_tcp(self, host: str, port: int, timeout: Optional[float] = None, local_address: Optional[str] = None,
                    socket_options=None):
        from httpcore import ConnectError, ConnectTimeout
        from httpcore._backends.sync import SyncStream
        source_address = None if local_address is None else (local_address, 0)
        try:
            sock = _happy_eyeballs_create_connection((host, port), timeout, source_address=source_address,
                                                     socket_options=socket_options or ())
        except TimeoutError as exc:
            raise ConnectTimeout(str(exc)) from exc
        except OSError as exc:
            raise ConnectError(str(exc)) from exc
        return SyncStream(sock)

    def connect_unix_socket(self, *args, **kwargs):
        return self._default_backend().connect_unix_socket(*args, **kwargs)

    def sleep(self, seconds: float) -> None:
        self._default_backend().sleep(seconds)


def _uses_codex_cloud_transport(base_url: str) -> bool:
    return base_url_hostname(base_url).lower() == "chatgpt.com" and "/backend-api/codex" in str(base_url).lower()


def _enable_happy_eyeballs(transport, skip_pool_types: tuple = ()) -> None:
    """Install the racing backend on one httpx transport.

    Reaches into private ``transport._pool._network_backend`` (httpcore pinned
    1.0.x); hasattr-guarded so an incompatible httpcore degrades to the default
    serial backend. Pools of ``skip_pool_types`` (proxies) are left alone.
    """
    pool = getattr(transport, "_pool", None)
    if pool is not None and hasattr(pool, "_network_backend") and not (skip_pool_types and isinstance(pool, skip_pool_types)):
        pool._network_backend = _HappyEyeballsSyncBackend()


def enable_happy_eyeballs_on_client(client) -> None:
    """Install the racing backend on every direct transport of a ready-built httpx.Client.

    For callers that build clients inline (Codex OAuth/device-login). Proxy-backed
    pools are skipped (TCP connect goes to the proxy host); async clients need
    nothing (anyio already races per RFC 8305). Best-effort.

    Proxy-backed transports (``httpcore.HTTPProxy`` / SOCKS pools) are left untouched: with a proxy in play
    the TCP connect goes to the proxy host, which is out of scope for the direct-transport racing added in
    #94388. Async clients are also left untouched — httpcore's async backend already performs RFC 8305
    racing natively via ``anyio.connect_tcp(happy_eyeballs_delay=0.25)``.
    """
    try:
        import httpcore
        proxy_pool_types = tuple(
            t for t in (getattr(httpcore, "HTTPProxy", None), getattr(httpcore, "SOCKSProxy", None)) if t is not None)
    except Exception:
        return
    for transport in (getattr(client, "_transport", None), *(getattr(client, "_mounts", None) or {}).values()):
        _enable_happy_eyeballs(transport, proxy_pool_types)


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

        __slots__ = ("_closed", "_inner")

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


@functools.lru_cache(maxsize=8)
def _parse_tls_max_version(raw: str) -> Any:
    """``ssl.TLSVersion`` for a ``network.tls_max_version`` value, else None.

    Accepts ``1.2`` / ``1.3``, optionally prefixed ``tls``/``tlsv`` (any case).
    1.0/1.1 are rejected: the knob dodges broken TLS 1.3 paths, it does not
    re-enable deprecated protocols. Cached so a bad value warns once.
    """
    import ssl

    normalized = raw.strip().lower()
    for prefix in ("tlsv", "tls"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    version = {"1.2": ssl.TLSVersion.TLSv1_2, "1.3": ssl.TLSVersion.TLSv1_3}.get(normalized)
    if version is None:
        import logging

        logging.getLogger(__name__).warning(
            "Ignoring network.tls_max_version=%r: expected '1.2' or '1.3'", raw)
    return version


def _configured_tls_max_version() -> Any:
    """The ``network.tls_max_version`` cap from config.yaml (None when unset/invalid).

    Some CDN edges and middleboxes accept TLS 1.2 ClientHellos but kill TLS 1.3
    ones, surfacing as ``[SSL: UNEXPECTED_EOF_WHILE_READING]`` ~15 s into every
    request while curl (the OS TLS stack) works (#44365). Read from config at
    client build time; every process (gateway, CLI, spawned agents) loads its
    own config, so nothing has to be inherited through the environment.
    """
    try:
        from hermes_cli.config import load_config_readonly

        net = (load_config_readonly() or {}).get("network")
    except Exception:
        return None
    raw = net.get("tls_max_version") if isinstance(net, dict) else None
    if raw is None or raw == "" or isinstance(raw, bool):
        return None
    # An unquoted ``tls_max_version: 1.2`` parses as a YAML float.
    return _parse_tls_max_version(str(raw))


def _set_maximum_version(ctx: Any, max_version: Any) -> None:
    """Set ``ctx.maximum_version`` on either SSLContext class.

    truststore's context forwards the property to the stdlib context it wraps, so a
    plain assignment is right there. On a stdlib-class context (a provider
    ``ssl_ca_cert`` context) the Python setter resolves the module-global
    ``SSLContext``, which truststore's injection replaces, and recurses until
    RecursionError; go through the C-level descriptor instead.
    """
    import _ssl

    if type(ctx).__module__.startswith("truststore") or not isinstance(ctx, _ssl._SSLContext):
        ctx.maximum_version = max_version
    else:
        _ssl._SSLContext.maximum_version.__set__(ctx, max_version)


_CAPPED_DEFAULT_CONTEXTS: dict[Any, Any] = {}
_CAPPED_DEFAULT_CONTEXTS_LOCK = threading.Lock()


def _capped_default_context(max_version: Any) -> Any:
    """httpx's own ``verify=True`` context, capped, with a stable identity per version.

    ``httpx.create_ssl_context()`` is exactly what httpx builds for
    ``verify=True``, so default trust semantics are unchanged. Memoized because
    sync transports are pooled by the ``verify`` object's identity
    (``_shared_transport_key``): a fresh context per client would give every
    client its own pool.
    """
    with _CAPPED_DEFAULT_CONTEXTS_LOCK:
        ctx = _CAPPED_DEFAULT_CONTEXTS.get(max_version)
        if ctx is None:
            import httpx

            ctx = httpx.create_ssl_context()
            _set_maximum_version(ctx, max_version)
            _CAPPED_DEFAULT_CONTEXTS[max_version] = ctx
        return ctx


def _apply_tls_max_version(verify: Any) -> Any:
    """Compose the configured TLS cap onto an httpx ``verify`` value.

    * ``ssl.SSLContext`` (per-provider ``ssl_ca_cert`` or the platform-store
      context from ``agent.ssl_verify``) -> capped in place, so its CA material
      and identity survive; no CA lookup of our own can make this step fail.
    * ``True`` -> the memoized capped default context.
    * ``False`` (``ssl_verify: false``) and anything else -> unchanged; a caller
      that turned verification off has opted out of TLS policy.
    """
    max_version = _configured_tls_max_version()
    if max_version is None:
        return verify
    import _ssl

    # ``_ssl._SSLContext`` is the base of both the stdlib class (provider ``ssl_ca_cert``
    # contexts) and truststore's injected one; ``ssl.SSLContext`` names only the latter.
    if isinstance(verify, _ssl._SSLContext):
        try:
            if verify.maximum_version != max_version:
                _set_maximum_version(verify, max_version)
        except (ValueError, OSError):
            pass
        return verify
    if verify is True:
        return _capped_default_context(max_version)
    return verify


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
        # network.tls_max_version (#44365): cap before the shared-transport key is computed.
        verify = _apply_tls_max_version(verify)
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100, keepalive_expiry=20.0)
        timeout = httpx.Timeout(connect=15.0, read=None, write=15.0, pool=10.0)  # read=None for SSE streaming
        transport_cls = httpx.AsyncHTTPTransport if async_mode else httpx.HTTPTransport
        client_cls = httpx.AsyncClient if async_mode else httpx.Client
        mounts = None
        # Official Codex backend only: zstd request bodies when the stdlib codec exists (None otherwise).
        from agent.codex_headers import codex_request_compression_hooks
        hooks = codex_request_compression_hooks(base_url, async_mode=async_mode)
        if hooks:
            client_cls = functools.partial(client_cls, event_hooks=hooks)
        if proxy is None:
            happy_eyeballs = not async_mode and _uses_codex_cloud_transport(base_url)
            # One pool serves every agent in the process, so its ceiling must cover a whole
            # fan-out of concurrently streaming children. (Client-level ``limits`` never reach
            # mounted transports — they used to run on httpx defaults, keepalive_expiry=5s.)
            direct_limits = limits if async_mode else httpx.Limits(
                max_keepalive_connections=50, max_connections=1000, keepalive_expiry=20.0,
            )

            def _build_direct():
                transport = transport_cls(verify=verify, limits=direct_limits)
                # Async transports race natively (anyio happy_eyeballs_delay=0.25).
                if happy_eyeballs:
                    _enable_happy_eyeballs(transport)
                return transport

            if async_mode:
                mounts = {"http://": _build_direct(), "https://": _build_direct()}
            else:
                key = _shared_transport_key(base_url, verify, proxy)
                view_cls = _shared_transport_cls()
                mounts = {
                    f"{scheme}://": view_cls(_get_shared_transport((scheme, *key), _build_direct))
                    for scheme in ("http", "https")
                }
                # Default transport = the https view; otherwise httpx builds a third, never-used
                # direct transport (pool + SSL context) per client.
                return client_cls(limits=limits, timeout=timeout, transport=mounts["https://"], mounts=mounts)
        return client_cls(limits=limits, timeout=timeout, proxy=proxy, mounts=mounts or None, verify=verify)
    except Exception:
        return None


def _install_safe_stdio() -> None:
    """Wrap stdout/stderr so best-effort console output cannot crash the agent."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        # The routing proxy already absorbs these errors; wrapping it stacks a new proxy per build (#139687).
        if stream is not None and not isinstance(stream, (_SafeWriter, _ThreadRoutingStream)):
            setattr(sys, stream_name, _SafeWriter(stream))


# Drop-in for ``openai.OpenAI``.
OpenAI = _OpenAIProxy()


__all__ = [
    "OpenAI",
    "_OpenAIProxy",
    "_SafeWriter",
    "_get_proxy_for_base_url",
    "_get_proxy_from_env",
    "_install_safe_stdio",
    "_load_openai_cls",
    "build_keepalive_http_client",
    "close_shared_transports",
    "enable_happy_eyeballs_on_client",
]
