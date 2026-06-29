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

``run_agent`` re-exports every name so existing
``from run_agent import _get_proxy_from_env`` imports keep working
unchanged.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from typing import Any, Optional

from utils import base_url_hostname, normalize_proxy_url


# Cached at module level so we only pay the OpenAI SDK import cost once
# per process (after the first lazy load).
_OPENAI_CLS_CACHE = None


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


def build_keepalive_http_client(
    base_url: str = "",
    *,
    async_mode: bool = False,
) -> Optional[Any]:
    """Build an httpx client for OpenAI SDK calls with env-only proxy policy.

    Uses explicit ``HTTPS_PROXY`` / ``NO_PROXY`` env vars via
    ``_get_proxy_for_base_url``. A custom transport disables httpx's default
    ``trust_env`` path, so macOS system proxy settings from
    ``urllib.request.getproxies()`` (which omit the ExceptionsList) are not
    applied. Mirrors ``AIAgent._build_keepalive_http_client``.
    """
    try:
        import httpx
        import socket

        if "api.githubcopilot.com" in str(base_url or "").lower():
            client_cls = httpx.AsyncClient if async_mode else httpx.Client
            return client_cls()

        sock_opts = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock_opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30))
            sock_opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10))
            sock_opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3))
        elif hasattr(socket, "TCP_KEEPALIVE"):
            sock_opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 30))

        proxy = _get_proxy_for_base_url(base_url)
        client_cls = httpx.AsyncClient if async_mode else httpx.Client
        from agent.clinepass_transport import (
            build_clinepass_transport,
            is_clinepass_base_url,
        )
        if is_clinepass_base_url(base_url):
            # ClinePass wraps non-streaming responses in a {data, success}
            # envelope; unwrap it so the OpenAI SDK sees top-level choices.
            transport = build_clinepass_transport(
                async_mode=async_mode, socket_options=sock_opts
            )
        else:
            transport_cls = httpx.AsyncHTTPTransport if async_mode else httpx.HTTPTransport
            transport = transport_cls(socket_options=sock_opts)
        return client_cls(
            transport=transport,
            proxy=proxy,
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
]
