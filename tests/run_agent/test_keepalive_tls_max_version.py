"""HERMES_TLS_MAX_VERSION must cap the TLS handshake on the primary client.

Some CDN edges and middleboxes accept TLS 1.2 handshakes but kill TLS 1.3
ClientHellos, surfacing as ``[SSL: UNEXPECTED_EOF_WHILE_READING]`` ~15s into
every request while ``curl`` (OS TLS stack) works fine — #44365 hit this on
Windows desktop against DeepSeek's edge.  ``network.tls_max_version: "1.2"``
in config.yaml (bridged to the internal ``HERMES_TLS_MAX_VERSION`` env var by
``hermes_constants.apply_tls_max_version`` at startup) caps the handshake for
the keepalive-enabled httpx.Client injected by ``_create_openai_client``.

httpx ignores client-level ``verify`` when an explicit ``transport`` is
passed, so main mounts a verify-carrying transport per scheme; these tests
pin that the cap lands on every pool the client can route through — the base
transport, the plain no-proxy mounts, and (when a proxy env var is set) the
proxy mount too.  The path #44365's reporter would actually exercise has no
proxy, but a capped direct transport next to an uncapped proxy mount would
silently reintroduce the bug for proxied users.

The cap must also *compose* with main's per-provider ``verify`` plumbing
(``ssl_ca_cert`` / ``ssl_verify`` resolved to an ``ssl.SSLContext``): capping
must not discard a caller-supplied CA bundle.
"""
import ssl

import httpx

from agent.process_bootstrap import _apply_tls_max_version, _get_tls_ssl_context
from run_agent import AIAgent

_PROXY_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
               "https_proxy", "http_proxy", "all_proxy")


def _clear_env(monkeypatch):
    for key in _PROXY_KEYS + ("HERMES_TLS_MAX_VERSION",):
        monkeypatch.delenv(key, raising=False)


def _pool_ssl_contexts(client: httpx.Client):
    """Collect the ssl contexts of the base transport and all mounts."""
    contexts = []
    transports = [client._transport, *client._mounts.values()]
    for transport in transports:
        if transport is None:
            continue
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


# ─── composition with main's per-provider verify plumbing ───────────────────


def test_apply_cap_noop_when_unset(monkeypatch):
    _clear_env(monkeypatch)
    assert _apply_tls_max_version(True) is True
    assert _apply_tls_max_version(False) is False
    existing = ssl.create_default_context()
    assert _apply_tls_max_version(existing) is existing


def test_apply_cap_swaps_true_for_capped_context(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_TLS_MAX_VERSION", "1.2")
    result = _apply_tls_max_version(True)
    assert isinstance(result, ssl.SSLContext)
    assert result.maximum_version == ssl.TLSVersion.TLSv1_2


def test_apply_cap_preserves_existing_ca_context(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_TLS_MAX_VERSION", "1.2")
    provided = ssl.create_default_context()
    result = _apply_tls_max_version(provided)
    # Same object — per-provider CA material must survive; only the ceiling moves.
    assert result is provided
    assert result.maximum_version == ssl.TLSVersion.TLSv1_2


def test_apply_cap_leaves_verify_false_untouched(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_TLS_MAX_VERSION", "1.2")
    assert _apply_tls_max_version(False) is False


# ─── end-to-end through the keepalive client builder ────────────────────────


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
        contexts = _pool_ssl_contexts(client)
        assert contexts
        for ctx in contexts:
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


# ─── config.yaml bridge — hermes_constants.apply_tls_max_version ────────────


def test_default_config_ships_tls_max_version_off():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["network"]["tls_max_version"] == ""


def test_bridge_sets_env_from_config(monkeypatch):
    from hermes_constants import apply_tls_max_version

    _clear_env(monkeypatch)
    apply_tls_max_version("1.2")
    import os

    assert os.environ["HERMES_TLS_MAX_VERSION"] == "1.2"


def test_bridge_does_not_override_explicit_env(monkeypatch):
    from hermes_constants import apply_tls_max_version

    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_TLS_MAX_VERSION", "1.3")
    apply_tls_max_version("1.2")
    import os

    assert os.environ["HERMES_TLS_MAX_VERSION"] == "1.3"


def test_bridge_noop_on_empty(monkeypatch):
    from hermes_constants import apply_tls_max_version

    _clear_env(monkeypatch)
    for value in ("", None, 0):
        apply_tls_max_version(value)
    import os

    assert "HERMES_TLS_MAX_VERSION" not in os.environ


def test_bridge_handles_yaml_float_and_whitespace(monkeypatch):
    """yaml parses an unquoted ``tls_max_version: 1.2`` as a float."""
    from hermes_constants import apply_tls_max_version

    _clear_env(monkeypatch)
    apply_tls_max_version(1.2)
    ctx = _get_tls_ssl_context()
    assert ctx is not None and ctx.maximum_version == ssl.TLSVersion.TLSv1_2

    _clear_env(monkeypatch)
    apply_tls_max_version("  1.2  ")
    ctx = _get_tls_ssl_context()
    assert ctx is not None and ctx.maximum_version == ssl.TLSVersion.TLSv1_2
