"""F5 — an IPv4-only Telegram transport on hosts with no global IPv6 (t_d2133284).

The 2026-10-23 card claimed the host's IPv6-first DNS "burns the Telegram retry
budget". That premise was DISPROVEN by direct measurement on this host:

  * the unroutable AAAA attempt costs ~8ms — the kernel rejects it instantly with
    "Network is unreachable" because there is no route at all;
  * glibc's RFC 3484 defaults already return the A record FIRST (no global IPv6
    source address demotes IPv6), so the hostname path is not IPv6-first here;
  * 10 consecutive HTTPS requests through plain httpx: 10/10 in 4.29s.

The residual defect that IS real: anyio's ``connect_tcp`` resolves every family
and applies Happy Eyeballs (RFC 6555) — the IPv6 attempt starts first and the
IPv4 attempt only starts if the first has not connected within 250ms. On a host
whose IPv6 path blackholes (a route that exists but silently drops) every connect
pays that 250ms penalty.

These tests pin the fix:
  * the global-IPv6 detection reads /proc/net/if_inet6 and reads the real SCOPE;
  * a host with no global IPv6 gets the IPv4-only transport, one with a global
    IPv6 address keeps the stock dual-stack transport;
  * a proxy keeps the stock transport (the proxy owns the socket);
  * the IPv4-only transport preserves the Host header (TLS SNI / cert verification
    are unchanged);
  * the transport still works end-to-end against the real Bot API.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from plugins.platforms.telegram import telegram_network as tn


# --------------------------------------------------------------------------
# Global-IPv6 detection
# --------------------------------------------------------------------------

def test_global_ipv6_detection_reads_the_scope_field(monkeypatch, tmp_path):
    """Scope 00 = global counts; link (20), host (10) and site (40) do not."""
    proc = tmp_path / "if_inet6"
    proc.write_text(
        "fe80000000000000be2411fffe5782bc 02 40 20 80    ens18\n"
        "00000000000000000000000000000001 01 80 10 80       lo\n",
        encoding="ascii",
    )
    monkeypatch.setattr("builtins.open", lambda f, *a, **k: proc.open() if f == "/proc/net/if_inet6" else open(f, *a, **k))

    assert tn._host_has_global_ipv6() is False


def test_global_ipv6_detection_finds_a_global_address(monkeypatch, tmp_path):
    proc = tmp_path / "if_inet6"
    proc.write_text(
        "fe80000000000000be2411fffe5782bc 02 40 20 80    ens18\n"
        "20010db8000000000000000000000001 03 64 00 80    eth0\n",
        encoding="ascii",
    )
    monkeypatch.setattr("builtins.open", lambda f, *a, **k: proc.open() if f == "/proc/net/if_inet6" else open(f, *a, **k))

    assert tn._host_has_global_ipv6() is True


def test_global_ipv6_detection_fails_safe(monkeypatch):
    """If the proc file cannot be read, keep the default dual-stack behaviour."""
    def _boom(*args, **kwargs):
        raise OSError("no /proc here")

    monkeypatch.setattr("builtins.open", _boom)
    assert tn._host_has_global_ipv6() is True


# --------------------------------------------------------------------------
# Transport selection
# --------------------------------------------------------------------------

def test_ipv4_only_transport_is_used_without_a_global_ipv6(monkeypatch):
    monkeypatch.setattr(tn, "_host_has_global_ipv6", lambda: False)
    transport = tn._build_transport()
    try:
        assert isinstance(transport._pool._network_backend, tn._IPv4OnlyBackend)
    finally:
        asyncio.run(transport.aclose())


def test_stock_transport_is_used_with_a_global_ipv6(monkeypatch):
    monkeypatch.setattr(tn, "_host_has_global_ipv6", lambda: True)
    transport = tn._build_transport()
    try:
        assert not isinstance(transport._pool._network_backend, tn._IPv4OnlyBackend)
    finally:
        asyncio.run(transport.aclose())


def test_proxy_keeps_the_stock_transport(monkeypatch):
    """A proxy owns the socket; forcing the target onto IPv4 would be wrong."""
    monkeypatch.setattr(tn, "_host_has_global_ipv6", lambda: False)
    transport = tn._build_transport(proxy="http://127.0.0.1:9")
    try:
        assert not isinstance(transport._pool._network_backend, tn._IPv4OnlyBackend)
    finally:
        asyncio.run(transport.aclose())


def test_ipv4_only_pool_is_installed_without_a_global_ipv6(monkeypatch):
    monkeypatch.setattr(tn, "_host_has_global_ipv6", lambda: False)
    transport = tn._build_transport()
    try:
        backend = transport._pool._network_backend
        assert isinstance(backend, tn._IPv4OnlyBackend)
    finally:
        asyncio.run(transport.aclose())


def test_stock_pool_backend_without_a_global_ipv6_is_replaced(monkeypatch):
    """With no global IPv6 the pool must carry a backend that overrides connect_tcp.

    The IPv4-only backend IS an AnyIOBackend subclass (it only overrides
    ``connect_tcp``), so the meaningful assertion is the override, not the type.
    """
    monkeypatch.setattr(tn, "_host_has_global_ipv6", lambda: False)
    transport = tn._build_transport()
    try:
        backend = transport._pool._network_backend
        assert type(backend).connect_tcp is not tn._AnyIOBackend.connect_tcp
    finally:
        asyncio.run(transport.aclose())


def test_building_the_transport_does_not_mutate_the_global_httpcore_pool(monkeypatch):
    """The IPv4-only backend must not be installed by patching a global.

    An earlier implementation swapped ``httpcore.AsyncConnectionPool`` for the
    duration of the transport construction.  httpx resolves that attribute at
    call time, so the swap was visible to ANY other thread constructing a
    transport concurrently — a gateway-wide race.  The pool is now built
    explicitly and installed on the transport instance instead.
    """
    import httpcore

    monkeypatch.setattr(tn, "_host_has_global_ipv6", lambda: False)
    original_pool_cls = httpcore.AsyncConnectionPool

    transport = tn._build_transport()
    try:
        assert httpcore.AsyncConnectionPool is original_pool_cls
        assert isinstance(transport._pool, original_pool_cls)
        assert isinstance(transport._pool._network_backend, tn._IPv4OnlyBackend)
    finally:
        asyncio.run(transport.aclose())


def test_the_stock_transport_is_untouched_when_the_host_has_global_ipv6(monkeypatch):
    """A host with global IPv6 keeps httpx's own pool object, unmodified."""
    monkeypatch.setattr(tn, "_host_has_global_ipv6", lambda: True)
    transport = tn._build_transport()
    try:
        assert not isinstance(transport._pool._network_backend, tn._IPv4OnlyBackend)
    finally:
        asyncio.run(transport.aclose())


# --------------------------------------------------------------------------
# The fallback transport uses it too
# --------------------------------------------------------------------------

def test_the_fallback_transport_uses_the_ipv4_only_pool(monkeypatch):
    monkeypatch.setattr(tn, "_host_has_global_ipv6", lambda: False)
    transport = tn.TelegramFallbackTransport(["149.154.166.110"])
    try:
        assert isinstance(transport._primary._pool._network_backend, tn._IPv4OnlyBackend)
    finally:
        asyncio.run(transport.aclose())


# --------------------------------------------------------------------------
# Live behaviour
# --------------------------------------------------------------------------

def test_ipv4_only_transport_preserves_the_host_header(monkeypatch):
    """TLS SNI and the Host header stay api.telegram.org — only the socket moves."""
    monkeypatch.setattr(tn, "_host_has_global_ipv6", lambda: False)

    async def _run():
        transport = tn._build_transport()
        try:
            async with httpx.AsyncClient(transport=transport,
                                         timeout=httpx.Timeout(20.0)) as client:
                response = await client.get("https://api.telegram.org")
                return response
        finally:
            await transport.aclose()

    response = asyncio.run(_run())
    assert response.status_code in (200, 302, 401, 404, 429)
    assert response.request.headers.get("host") == "api.telegram.org"
