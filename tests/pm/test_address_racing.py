"""The downloader's HTTPS connection races address families instead of trying them in order.

``socket.create_connection`` moves on to the next address only when the TCP connect
itself fails. A middlebox that accepts the TCP handshake and then resets the TLS
ClientHello fails *after* ``connect()`` returns, so the failure is invisible to that
loop — the caller sees a reset handshake from a perfectly good network, and the
working IPv4 address is never tried. These tests pin the racing loop and, as a
contrast, the behaviour it replaces.
"""

import socket
import ssl

import pytest

from pm import downloader
from pm.downloader import _AddressRacingHTTPSConnection, _address_families, _OPENER


class _FakeSocket:
    """A socket that accepts every TCP connect and records which address it got."""

    def __init__(self, family, kind, proto=0):
        self.family = family
        self._tls_ok = True

    def settimeout(self, timeout):
        pass

    def setsockopt(self, *args):
        pass

    def connect(self, address):
        pass

    def close(self):
        pass


class _Context(ssl.SSLContext):
    """A real SSLContext that fails the TLS wrap for the first socket it is given."""

    def __new__(cls, fail_first=True):
        return super().__new__(cls, ssl.PROTOCOL_TLS_CLIENT)

    def __init__(self, fail_first=True):
        self.fail_first = fail_first
        self.wrapped = []
        self.check_hostname = False
        self.verify_mode = ssl.CERT_NONE

    def wrap_socket(self, *args, server_hostname=None, **kwargs):
        sock = args[0] if args else kwargs["sock"]
        self.wrapped.append(sock)
        if self.fail_first and len(self.wrapped) == 1:
            raise OSError("[WinError 10054] An existing connection was forcibly closed "
                          "by the remote host")
        return sock


def _getaddrinfo(addresses):
    """Build the 5-tuples getaddrinfo returns, from (family, sockaddr) pairs."""
    return [(family, socket.SOCK_STREAM, 0, "", sockaddr) for family, sockaddr in addresses]


def _two_addresses(monkeypatch):
    """Pretend the host has two IPv4 addresses; both accept TCP."""
    addresses = [(socket.AF_INET, ("203.0.113.10", 443)), (socket.AF_INET, ("198.51.100.20", 443))]
    monkeypatch.setattr(downloader, "_address_families", lambda host, port: list(addresses))
    return addresses


def test_connection_tries_the_next_address_when_tls_fails(monkeypatch):
    """The exact case create_connection cannot handle: TCP up, TLS reset."""
    attempted = []

    class _RecordingSocket(_FakeSocket):
        def connect(self, address):
            attempted.append(address)

    monkeypatch.setattr(downloader.socket, "socket", _RecordingSocket)
    _two_addresses(monkeypatch)
    context = _Context(fail_first=True)

    conn = _AddressRacingHTTPSConnection("example.com", 443, context=context)
    conn.connect()

    assert attempted == [("203.0.113.10", 443), ("198.51.100.20", 443)], (
        "the second address was never tried, so a TLS reset on the first is fatal")
    assert conn.sock is context.wrapped[-1]


def test_create_connection_cannot_see_a_tls_failure(monkeypatch):
    """Contrast: the stock path returns after the first TCP win, TLS failure included.

    This is why the downloader carries its own loop — not a stylistic choice.
    """
    attempted = []

    class _RecordingSocket(_FakeSocket):
        def connect(self, address):
            attempted.append(address)

    monkeypatch.setattr(downloader.socket, "socket", _RecordingSocket)
    addresses = [(socket.AF_INET, ("203.0.113.10", 443)), (socket.AF_INET, ("198.51.100.20", 443))]
    monkeypatch.setattr(downloader.socket, "getaddrinfo",
                        lambda host, port, family=0, kind=0: _getaddrinfo(addresses))

    sock = downloader.socket.create_connection(("example.com", 443), timeout=5)

    assert attempted == [("203.0.113.10", 443)], (
        "create_connection only iterates on TCP failure; the TLS reset is invisible to it")
    assert sock is not None


def test_connection_raises_when_no_address_works(monkeypatch):
    class _DeadSocket(_FakeSocket):
        def connect(self, address):
            raise OSError("network is unreachable")

    monkeypatch.setattr(downloader.socket, "socket", _DeadSocket)
    _two_addresses(monkeypatch)

    conn = _AddressRacingHTTPSConnection("example.com", 443, context=_Context())
    with pytest.raises(OSError, match="could not connect"):
        conn.connect()


def test_connection_uses_the_tunnel_path_for_a_proxy():
    """A proxy tunnel is one address, not a race — the base class must handle it."""
    conn = _AddressRacingHTTPSConnection("proxy.example", 443, context=_Context())
    conn.set_tunnel("internal.example", 443)
    assert conn._tunnel_host == "internal.example"


def test_address_families_prefers_ipv4_and_dedupes(monkeypatch):
    """IPv4 first so a broken IPv6 path is a fallback, not a failure."""
    v4 = (socket.AF_INET, ("203.0.113.10", 443))
    v6 = (socket.AF_INET6, ("2600:9000::1", 443, 0, 0))
    monkeypatch.setattr(downloader.socket, "getaddrinfo",
                        lambda host, port, family=0, kind=0: _getaddrinfo([v6, v4, v6, v4]))

    families = _address_families("example.com", 443)

    assert families == [v4, v6]
    assert families[0][0] == socket.AF_INET


def test_address_families_keeps_an_ipv6_only_host_working(monkeypatch):
    """IPv4-first is a bias, not a block: an IPv6-only host still resolves."""
    v6 = (socket.AF_INET6, ("2600:9000::1", 443, 0, 0))
    monkeypatch.setattr(downloader.socket, "getaddrinfo",
                        lambda host, port, family=0, kind=0: _getaddrinfo([v6]))

    assert _address_families("example.com", 443) == [v6]


def test_opener_is_served_by_the_racing_handler():
    handlers = [type(h).__name__ for h in getattr(_OPENER, "handlers", [])]
    assert "_AddressRacingHTTPSHandler" in handlers
    assert not any("HTTPSHandler" == name for name in handlers), (
        "the stock HTTPSHandler must not also be installed, or it may win the protocol")


def test_server_hostname_strips_ipv6_brackets():
    assert downloader._server_hostname("[2600:9000::1]") == "2600:9000::1"
    assert downloader._server_hostname("example.com") == "example.com"
