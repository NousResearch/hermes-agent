"""HERMES_TLS_MAX_VERSION must cap the TLS handshake on the primary client.

Some CDN edges and middleboxes accept TLS 1.2 handshakes but kill TLS 1.3
ClientHellos, surfacing as ``[SSL: UNEXPECTED_EOF_WHILE_READING]`` ~15s into
every request while ``curl`` (OS TLS stack) works fine — #44365 hit this on
Windows desktop against DeepSeek's edge.  ``HERMES_TLS_MAX_VERSION=1.2``
caps the handshake for the keepalive-enabled httpx.Client injected by
``_create_openai_client``.

httpx ignores client-level ``verify`` when an explicit ``transport`` is
passed, so these tests pin that the context lands on the transport's pool —
and, when a proxy env var is set, on the proxy mount too (the path #44365's
reporter would actually exercise has no proxy, but a capped direct transport
next to an uncapped proxy mount would silently reintroduce the bug for
proxied users).
"""
import ssl

import httpx

from agent.process_bootstrap import _get_tls_ssl_context
from run_agent import AIAgent

_PROXY_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
               "https_proxy", "http_proxy", "all_proxy")


def _clear_env(monkeypatch):
    for key in _PROXY_KEYS + ("HERMES_TLS_MAX_VERSION",):
        monkeypatch.delenv(key, raising=False)


def _pool_ssl_contexts(client: httpx.Client):
    """Collect the ssl contexts of the base transport and all mounts."""
    contexts = []
    for transport in [client._transport, *client._mounts.values()]:
        pool = getattr(transport, "_pool", None)
        ctx = getattr(pool, "_ssl_context", None)
        if ctx is not None:
            contexts.append(ctx)
    return contexts


def test_unset_env_returns_none(monkeypatch):
    _clear_env(monkeypatch)
    assert _get_tls_ssl_context() is None
    monkeypatch.setenv("HERMES_TLS_MAX_VERSION", "   ")
    assert _get_tls_ssl_context() is None


def test_caps_maximum_version_at_tls12(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_TLS_MAX_VERSION", "1.2")
    ctx = _get_tls_ssl_context()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.maximum_version == ssl.TLSVersion.TLSv1_2


def test_accepts_tls_prefixes_case_insensitive(monkeypatch):
    _clear_env(monkeypatch)
    for raw in ("TLSv1.2", "tls1.2", "TLS1.2"):
        monkeypatch.setenv("HERMES_TLS_MAX_VERSION", raw)
        ctx = _get_tls_ssl_context()
        assert ctx is not None and ctx.maximum_version == ssl.TLSVersion.TLSv1_2, raw
    monkeypatch.setenv("HERMES_TLS_MAX_VERSION", "1.3")
    ctx = _get_tls_ssl_context()
    assert ctx is not None and ctx.maximum_version == ssl.TLSVersion.TLSv1_3


def test_invalid_values_are_ignored(monkeypatch):
    _clear_env(monkeypatch)
    # 1.0/1.1 are deliberately rejected — the knob exists to dodge broken
    # TLS 1.3 paths, not to enable deprecated protocol versions.
    for raw in ("1.1", "1.0", "garbage", "ssl3"):
        monkeypatch.setenv("HERMES_TLS_MAX_VERSION", raw)
        assert _get_tls_ssl_context() is None, raw


def test_keepalive_client_transport_gets_capped_context(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_TLS_MAX_VERSION", "1.2")
    client = AIAgent._build_keepalive_http_client("https://api.deepseek.com")
    assert isinstance(client, httpx.Client)
    try:
        contexts = _pool_ssl_contexts(client)
        assert contexts, "expected the keepalive transport to expose an ssl context"
        for ctx in contexts:
            assert ctx.maximum_version == ssl.TLSVersion.TLSv1_2
    finally:
        client.close()


def test_proxy_mount_inherits_capped_context(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_TLS_MAX_VERSION", "1.2")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    client = AIAgent._build_keepalive_http_client("https://api.deepseek.com")
    assert isinstance(client, httpx.Client)
    try:
        pools = [
            type(mount._pool).__name__
            for mount in client._mounts.values()
            if mount is not None and hasattr(mount, "_pool")
        ]
        assert "HTTPProxy" in pools, pools
        for ctx in _pool_ssl_contexts(client):
            assert ctx.maximum_version == ssl.TLSVersion.TLSv1_2
    finally:
        client.close()


def test_unset_env_keeps_default_tls(monkeypatch):
    _clear_env(monkeypatch)
    client = AIAgent._build_keepalive_http_client("https://api.deepseek.com")
    assert isinstance(client, httpx.Client)
    try:
        for ctx in _pool_ssl_contexts(client):
            assert ctx.maximum_version == ssl.TLSVersion.MAXIMUM_SUPPORTED
    finally:
        client.close()
