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

4. **TLS version cap** — ``_get_tls_ssl_context`` honors
   ``network.tls_max_version`` from config.yaml (bridged to the internal
   ``HERMES_TLS_MAX_VERSION`` env var at startup) and ``_apply_tls_max_version``
   composes that cap onto the httpx ``verify`` value the client builders already
   carry, so users behind TLS-1.3-hostile networks or CDN edges can cap provider
   connections at TLS 1.2 without losing per-provider CA settings.

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


def _get_tls_ssl_context() -> Optional["ssl.SSLContext"]:
    """Build an ``ssl.SSLContext`` capped by ``HERMES_TLS_MAX_VERSION``.

    Some CDN edges and middleboxes accept TLS 1.2 handshakes but kill
    TLS 1.3 ClientHellos, surfacing as ``[SSL: UNEXPECTED_EOF_WHILE_READING]``
    roughly 15s into every request while ``curl`` (which uses the OS TLS
    stack on Windows/macOS) works fine (#44365, DeepSeek's edge).  The
    user-facing knob is ``network.tls_max_version: "1.2"`` in config.yaml,
    bridged onto ``HERMES_TLS_MAX_VERSION`` at process startup by
    ``hermes_constants.apply_tls_max_version`` (this layer has no config
    access, and spawned agent subprocesses must inherit the cap; an
    explicitly exported env var wins over config.yaml).

    Accepted values: ``1.2`` / ``1.3``, optionally prefixed ``tls``/``tlsv``
    (case-insensitive).  Returns ``None`` — meaning "use httpx defaults" —
    when the variable is unset, empty, or invalid.  The context honors the
    same CA-bundle overrides as the rest of the CLI: ``HERMES_CA_BUNDLE`` >
    ``REQUESTS_CA_BUNDLE`` > ``SSL_CERT_FILE``.
    """
    raw = os.environ.get("HERMES_TLS_MAX_VERSION", "").strip()
    if not raw:
        return None

    import logging
    import ssl

    logger = logging.getLogger(__name__)

    normalized = raw.lower()
    for prefix in ("tlsv", "tls"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    max_version = {
        "1.2": ssl.TLSVersion.TLSv1_2,
        "1.3": ssl.TLSVersion.TLSv1_3,
    }.get(normalized)
    if max_version is None:
        logger.warning(
            "Ignoring HERMES_TLS_MAX_VERSION=%r: expected '1.2' or '1.3'", raw
        )
        return None

    cafile = None
    for env_var in ("HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        path = os.environ.get(env_var, "").strip()
        if path:
            cafile = path
            break
    try:
        ctx = ssl.create_default_context(cafile=cafile)
    except OSError as exc:
        logger.warning(
            "Ignoring HERMES_TLS_MAX_VERSION=%r: failed to load CA bundle "
            "%r: %s", raw, cafile, exc
        )
        return None
    ctx.maximum_version = max_version
    return ctx


def _apply_tls_max_version(verify: Any) -> Any:
    """Compose the ``HERMES_TLS_MAX_VERSION`` cap onto an httpx ``verify``.

    Main resolves per-provider ``ssl_ca_cert`` / ``ssl_verify`` into the httpx
    ``verify`` argument (``True``, ``False``, or an ``ssl.SSLContext``).  The
    TLS cap must ride on top of that rather than replace it:

    * an existing ``SSLContext`` (a per-provider CA bundle) → cap it in place
      so the caller's CA material survives;
    * ``True`` (certifi default) → swap in a fresh capped default context;
    * ``False`` (verification disabled, local dev only) → left untouched; a
      max-version cap needs a context, and a caller who turned verification
      off has opted out of TLS policy entirely.

    Returns ``verify`` unchanged when the cap is unset or invalid.
    """
    capped = _get_tls_ssl_context()
    if capped is None:
        return verify

    import ssl

    if isinstance(verify, ssl.SSLContext):
        try:
            verify.maximum_version = capped.maximum_version
        except (ValueError, OSError):
            pass
        return verify
    if verify is True:
        return capped
    return verify


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
    """
    try:
        import httpx

        proxy = _get_proxy_for_base_url(base_url)
        # Cap the TLS handshake if HERMES_TLS_MAX_VERSION is set (#44365),
        # composing onto any per-provider verify context already resolved.
        verify = _apply_tls_max_version(verify)

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
            mounts = {
                "http://": transport_cls(verify=verify),
                "https://": transport_cls(verify=verify),
            }
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
    "_get_tls_ssl_context",
    "_apply_tls_max_version",
    "build_keepalive_http_client",
]
