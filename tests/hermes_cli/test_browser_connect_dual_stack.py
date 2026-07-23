"""Dual-stack loopback discovery + port-squatter handling for /browser connect.

Regression context: on Windows, an IDE debugger (VS Code js-debug) holding
127.0.0.1:9222 pushes a Chrome launched with --remote-debugging-port=9222
onto [::1]:9222 only. The old IPv4-only probe missed the live browser AND
hung against the squatter (accepts TCP, never answers HTTP), driving the
whole connect past the desktop GUI's RPC timeout
("error: request timed out: browser.manage").
"""

from __future__ import annotations

import socket
import threading

import pytest

from hermes_cli.browser_connect import (
    DEFAULT_BROWSER_CDP_PORT,
    discover_local_cdp_url,
    find_free_debug_port,
    local_port_in_use,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def ipv4_squatter():
    """A listener on the IPv4 loopback that accepts TCP but never speaks HTTP."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    port = server.getsockname()[1]

    conns: list[socket.socket] = []
    stop = threading.Event()

    def _accept_loop() -> None:
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
                conns.append(conn)  # hold open, say nothing — like a debug adapter
            except OSError:
                continue

    thread = threading.Thread(target=_accept_loop, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        stop.set()
        thread.join(timeout=2)
        for conn in conns:
            try:
                conn.close()
            except OSError:
                pass
        server.close()


class TestDiscoverLocalCdpUrl:
    def test_returns_none_when_nothing_listens(self):
        port = _free_port()
        assert discover_local_cdp_url(port, timeout=0.3) is None

    def test_does_not_hang_on_non_cdp_squatter(self, ipv4_squatter):
        """A TCP-accepting, HTTP-silent squatter must fail the probe within
        the timeout instead of being mistaken for a browser."""
        assert discover_local_cdp_url(ipv4_squatter, timeout=0.3) is None

    def test_finds_ipv6_only_endpoint(self, monkeypatch):
        """When only [::1] speaks CDP (IPv4 side squatted), discovery
        returns the IPv6 URL instead of giving up."""
        import hermes_cli.browser_connect as bc

        def _ready(url: str, timeout: float = 1.0) -> bool:
            return "[::1]" in url

        monkeypatch.setattr(bc, "is_browser_debug_ready", _ready)
        assert bc.discover_local_cdp_url(9222) == "http://[::1]:9222"

    def test_prefers_ipv4_when_both_answer(self, monkeypatch):
        import hermes_cli.browser_connect as bc

        monkeypatch.setattr(bc, "is_browser_debug_ready", lambda *_a, **_k: True)
        assert bc.discover_local_cdp_url(9222) == "http://127.0.0.1:9222"


class TestLocalPortInUse:
    def test_free_port_reports_unused(self):
        assert local_port_in_use(_free_port(), timeout=0.3) is False

    def test_squatted_port_reports_used(self, ipv4_squatter):
        assert local_port_in_use(ipv4_squatter, timeout=0.5) is True


class TestFindFreeDebugPort:
    def test_returns_port_above_preferred(self):
        port = find_free_debug_port(DEFAULT_BROWSER_CDP_PORT)
        assert port > DEFAULT_BROWSER_CDP_PORT

    def test_skips_occupied_successor(self):
        """When preferred+1 is held on IPv4, the next candidate is chosen."""
        preferred = _free_port()
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            blocker.bind(("127.0.0.1", preferred + 1))
            blocker.listen(1)
            port = find_free_debug_port(preferred)
            assert port != preferred + 1
            assert port > preferred
        except OSError:
            pytest.skip("successor port unavailable to bind in this environment")
        finally:
            blocker.close()
