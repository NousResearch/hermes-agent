"""Shared httpx client helpers for agent/provider runtimes.

Windows can spend hundreds of milliseconds in ``SSLContext.load_verify_locations``
for each new httpx/OpenAI/Anthropic client.  A gateway process creates several
clients while constructing every AIAgent, so reusing a process-wide SSL context
keeps session startup from paying that certificate-load cost repeatedly.
"""

from __future__ import annotations

import os
import ssl
import urllib.request
from functools import lru_cache
from typing import Iterable

import certifi

from utils import base_url_hostname, normalize_proxy_env_vars, normalize_proxy_url


_CA_FILE_ENV_VARS = (
    "HERMES_CA_BUNDLE",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)


def _env_file(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value and os.path.isfile(value):
            return value
    return ""


def _env_dir(name: str) -> str:
    value = os.getenv(name, "").strip()
    return value if value and os.path.isdir(value) else ""


@lru_cache(maxsize=8)
def _shared_ssl_context(cafile: str, capath: str) -> ssl.SSLContext:
    if cafile:
        return ssl.create_default_context(cafile=cafile)
    if capath:
        return ssl.create_default_context(capath=capath)
    return ssl.create_default_context(cafile=certifi.where())


def get_shared_ssl_context() -> ssl.SSLContext:
    """Return a cached SSL context matching httpx's CA lookup policy.

    httpx normally builds a fresh context per client.  Passing this context into
    client constructors preserves certificate verification while amortizing the
    expensive CA-bundle load across the process.
    """

    cafile = _env_file(*_CA_FILE_ENV_VARS)
    capath = "" if cafile else _env_dir("SSL_CERT_DIR")
    return _shared_ssl_context(cafile, capath)


def _proxy_from_env() -> str | None:
    for key in (
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ):
        normalized = normalize_proxy_url(os.getenv(key, ""))
        if normalized:
            return normalized
    return None


def proxy_for_base_url(base_url: str | None) -> str | None:
    """Return the env proxy unless NO_PROXY excludes ``base_url``."""

    proxy = _proxy_from_env()
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


def keepalive_socket_options() -> list[tuple]:
    """TCP keepalive socket options for long-lived provider connections.

    Without these, a half-open connection (provider crash / silent network
    drop / LB idle-kill with no FIN) leaves a blocking ``recv`` waiting
    indefinitely — the kernel never learns the peer is gone.  Enabling
    keepalive makes the kernel probe after 30s idle, every 10s, giving up
    after 3 misses (~60s), at which point the socket errors out and the
    streaming reader unblocks so the agent's retry loop can reconnect.

    Falls back to bare ``SO_KEEPALIVE`` on platforms without the per-socket
    tuning knobs (e.g. macOS lacks ``TCP_KEEPIDLE``).
    """

    import socket as _socket

    opts: list[tuple] = [(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)]
    if hasattr(_socket, "TCP_KEEPIDLE"):
        opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPIDLE, 30))
        opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPINTVL, 10))
        opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPCNT, 3))
    elif hasattr(_socket, "TCP_KEEPALIVE"):
        opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPALIVE, 30))
    return opts


def build_httpx_client(
    *,
    base_url: str | None = None,
    timeout=None,
    headers=None,
    socket_options: Iterable[tuple] | None = None,
):
    """Build a sync httpx client using the shared SSL context.

    ``socket_options`` is used by the primary chat client to keep TCP
    keepalives.  Supplying a custom transport disables httpx's env-proxy
    discovery, so in that path we explicitly forward the matching proxy URL.
    """

    import httpx

    normalize_proxy_env_vars()
    verify = get_shared_ssl_context()
    kwargs = {"verify": verify}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if headers is not None:
        kwargs["headers"] = headers
    if socket_options is not None:
        kwargs["transport"] = httpx.HTTPTransport(
            verify=verify,
            socket_options=socket_options,
        )
        proxy = proxy_for_base_url(base_url)
        if proxy:
            kwargs["proxy"] = proxy
    return httpx.Client(**kwargs)


def build_openai_http_client():
    """Build an OpenAI SDK default client with cached certificate state."""

    from openai import DefaultHttpxClient

    return DefaultHttpxClient(verify=get_shared_ssl_context())


def build_openai_async_http_client():
    """Build an OpenAI SDK async default client with cached certificate state."""

    from openai import DefaultAsyncHttpxClient

    return DefaultAsyncHttpxClient(verify=get_shared_ssl_context())
