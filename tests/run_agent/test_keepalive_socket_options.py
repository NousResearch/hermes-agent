"""Regression coverage for #28834 — first tool call after a 2-3 minute
idle pause freezes at "[Calling tool: ...]" because the provider
socket has silently zombified and neither the macOS TCP keepalive
budget nor the httpx pool retried fast enough to recover.

The fix has two halves, both inside
``AIAgent._build_keepalive_http_client``:

* set ``TCP_KEEPINTVL`` / ``TCP_KEEPCNT`` on **both** the Linux and
  macOS branches so the dead-peer detection budget is bounded at
  ~60 s on both platforms (was ~10 min on macOS)
* pass ``retries=1`` to ``httpx.HTTPTransport`` so a stale-pool
  connection that beats the keepalive timer triggers a single
  transparent re-dial instead of bubbling up as a freeze

These tests pin both halves at the socket-option / transport level
so neither half can silently regress.
"""
from __future__ import annotations

import socket
from unittest.mock import patch

import httpx
import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _socket_options() -> list:
    """Return the list of ``(level, opt, value)`` socket options that the
    keepalive client would apply on the current host."""
    client = AIAgent._build_keepalive_http_client("https://api.example.com/v1")
    assert client is not None, "keepalive client must be built on a supported host"
    try:
        transport = client._transport
        # httpx 0.28 exposes the SOCKET_OPTION iterable on _pool, older
        # builds stored it on the transport directly.
        opts = getattr(transport, "_socket_options", None)
        if opts is None:
            opts = getattr(getattr(transport, "_pool", None), "_socket_options", None)
        return list(opts or [])
    finally:
        client.close()


def _opt_value(opts, level, name) -> int | None:
    """Return the value set for ``(level, getattr(socket, name))`` or
    ``None`` when the option is absent.  Resolves the constant by name
    so missing attributes (e.g. ``TCP_KEEPIDLE`` on macOS) don't
    AttributeError the helper itself."""
    opt = getattr(socket, name, None)
    if opt is None:
        return None
    for _l, _o, _v in opts:
        if _l == level and _o == opt:
            return _v
    return None


# ---------------------------------------------------------------------------
# Both platforms share these knobs.
# ---------------------------------------------------------------------------


class TestKeepaliveSharedKnobs:
    def test_so_keepalive_enabled(self):
        opts = _socket_options()
        assert _opt_value(opts, socket.SOL_SOCKET, "SO_KEEPALIVE") == 1

    def test_keepintvl_set_to_10s(self):
        # Without this, macOS inherits KEEPINTVL=75 s and the dead-peer
        # detection window blows past the issue's "2-3 minute pause".
        # The production helper sets this knob best-effort behind a
        # ``hasattr`` gate, so skip on hosts that don't expose the
        # constant — the exact value contract is pinned in the macOS
        # facade test where the constant is deliberately present.
        if not hasattr(socket, "TCP_KEEPINTVL"):
            pytest.skip("host socket module lacks TCP_KEEPINTVL")
        opts = _socket_options()
        assert _opt_value(opts, socket.IPPROTO_TCP, "TCP_KEEPINTVL") == 10

    def test_keepcnt_set_to_3(self):
        # Same story for KEEPCNT — kernel default is 8 on macOS. Best-effort
        # behind ``hasattr`` in the helper, so skip when the host lacks the
        # constant; the value is pinned in the macOS facade test.
        if not hasattr(socket, "TCP_KEEPCNT"):
            pytest.skip("host socket module lacks TCP_KEEPCNT")
        opts = _socket_options()
        assert _opt_value(opts, socket.IPPROTO_TCP, "TCP_KEEPCNT") == 3

    def test_idle_warmup_set_to_30s(self):
        # One of TCP_KEEPIDLE (Linux) or TCP_KEEPALIVE (macOS) must
        # carry the 30 s warm-up — whichever the host happens to
        # expose.  Both being unset would mean the agent waits the
        # full kernel default (2 h on Linux) before probing.
        opts = _socket_options()
        idle = _opt_value(opts, socket.IPPROTO_TCP, "TCP_KEEPIDLE")
        alive = _opt_value(opts, socket.IPPROTO_TCP, "TCP_KEEPALIVE")
        assert 30 in {idle, alive}


# ---------------------------------------------------------------------------
# macOS-specific branch — emulate the platform's hasattr profile so the
# test passes on any host (CI runners are Linux).
# ---------------------------------------------------------------------------


class TestMacOSKeepaliveParity:
    def test_macos_branch_still_sets_intvl_and_cnt(self):
        """On a host where ``socket.TCP_KEEPIDLE`` is missing (the macOS
        profile), the helper must still emit the INTVL/CNT options.

        Stubs the module-level attribute lookup so the helper takes the
        ``elif hasattr(_socket, "TCP_KEEPALIVE")`` branch even on Linux
        CI runners that natively have ``TCP_KEEPIDLE``.
        """
        # Build a socket module facade whose attribute surface mirrors
        # macOS: TCP_KEEPALIVE + TCP_KEEPINTVL + TCP_KEEPCNT, no
        # TCP_KEEPIDLE.  ``hasattr`` checks inside the helper resolve
        # against this facade.
        class _MacSocket:
            SOL_SOCKET = socket.SOL_SOCKET
            SO_KEEPALIVE = socket.SO_KEEPALIVE
            IPPROTO_TCP = socket.IPPROTO_TCP
            TCP_KEEPALIVE = 0x10
            TCP_KEEPINTVL = 0x101
            TCP_KEEPCNT = 0x102
            # Intentionally NO TCP_KEEPIDLE — that's the whole point.

        with patch.dict("sys.modules", {"socket": _MacSocket}):
            client = AIAgent._build_keepalive_http_client("https://api.example.com/v1")
        assert client is not None
        try:
            transport = client._transport
            opts = list(
                getattr(transport, "_socket_options", None)
                or getattr(getattr(transport, "_pool", None), "_socket_options", None)
                or []
            )
        finally:
            client.close()

        flat = [(level, opt) for (level, opt, _v) in opts]
        assert (_MacSocket.SOL_SOCKET, _MacSocket.SO_KEEPALIVE) in flat
        assert (_MacSocket.IPPROTO_TCP, _MacSocket.TCP_KEEPALIVE) in flat
        assert (_MacSocket.IPPROTO_TCP, _MacSocket.TCP_KEEPINTVL) in flat
        assert (_MacSocket.IPPROTO_TCP, _MacSocket.TCP_KEEPCNT) in flat
        # And the values must match the documented 30 s / 10 s / 3
        # budget — a future tweak that silently relaxes them would
        # re-open #28834.
        as_dict = {(level, opt): val for (level, opt, val) in opts}
        assert as_dict[(_MacSocket.IPPROTO_TCP, _MacSocket.TCP_KEEPALIVE)] == 30
        assert as_dict[(_MacSocket.IPPROTO_TCP, _MacSocket.TCP_KEEPINTVL)] == 10
        assert as_dict[(_MacSocket.IPPROTO_TCP, _MacSocket.TCP_KEEPCNT)] == 3


# ---------------------------------------------------------------------------
# Stale-pool retry — the second half of the fix.
# ---------------------------------------------------------------------------


class TestStalePoolRetry:
    def test_http_transport_retries_once(self):
        """The transport must be built with ``retries=1`` so a
        zombie connection from the keepalive pool gets transparently
        re-dialled instead of hanging the next ``chat.completions``
        call (#28834).  httpx only retries connection-establishment
        failures, so this can't double-submit a half-sent request.
        """
        client = AIAgent._build_keepalive_http_client("https://api.example.com/v1")
        assert client is not None
        try:
            transport = client._transport
            assert isinstance(transport, httpx.HTTPTransport)
            # httpcore's ConnectionPool stores ``_retries`` — both
            # http1 and http2 pool variants honour the kwarg.
            pool = getattr(transport, "_pool", None)
            retries = getattr(pool, "_retries", None)
            assert retries == 1, (
                f"expected httpx.HTTPTransport(retries=1) for stale-pool recovery "
                f"(#28834); pool reports retries={retries!r}"
            )
        finally:
            client.close()
