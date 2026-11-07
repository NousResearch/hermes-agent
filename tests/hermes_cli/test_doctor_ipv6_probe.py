"""``hermes doctor`` dead-IPv6 probe (#114265 secondary finding 1).

An advertised-but-blackholed IPv6 route stalls every serial connect for the full timeout;
the racer hides most of it, but nothing told the user that ``network.force_ipv4`` exists.
The probe connects to one known AAAA host over IPv6 with a short timeout and, on a
timeout, names the remedy. No IPv6 route at all is healthy (skip), as is an explicit
``force_ipv4: true``.
"""

from __future__ import annotations

import socket

from hermes_cli import doctor_connectivity as dc

_AAAA = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::1", 443, 0, 0))]


def _run(monkeypatch, connect, *, config=None, addrinfo=_AAAA):
    monkeypatch.setattr(dc, "_load_network_config", lambda: config or {})
    monkeypatch.setattr(dc.socket, "getaddrinfo", lambda *_a, **_k: addrinfo)
    monkeypatch.setattr(dc, "_tcp_connect", connect)
    return dc._probe_ipv6_path()


def test_dead_ipv6_route_warns_and_names_force_ipv4(monkeypatch):
    def timed_out(_sockaddr, _timeout):
        raise TimeoutError("timed out")

    result = _run(monkeypatch, timed_out)
    (glyph, _label, detail), = result.lines
    assert "⚠" in glyph and "network.force_ipv4" in detail
    assert result.issues and "network.force_ipv4" in result.issues[0]


def test_healthy_or_absent_ipv6_never_warns(monkeypatch):
    def reachable(_sockaddr, _timeout):
        return None

    def no_route(_sockaddr, _timeout):
        raise OSError(101, "Network is unreachable")

    assert _run(monkeypatch, reachable).issues == []
    assert _run(monkeypatch, no_route).issues == []
    assert "✓" in _run(monkeypatch, reachable).lines[0][0]
    # force_ipv4 already set, or no AAAA record: nothing to probe, nothing to say.
    assert _run(monkeypatch, reachable, config={"force_ipv4": True}).lines == []
    assert _run(monkeypatch, reachable, addrinfo=[]).lines == []


def test_proxy_routed_host_skips_direct_ipv6_probe(monkeypatch):
    from urllib import request

    monkeypatch.setattr(request, "getproxies", lambda: {"https": "http://127.0.0.1:8080"})
    monkeypatch.setattr(request, "proxy_bypass", lambda _host: False)
    called = False

    def timed_out(_sockaddr, _timeout):
        nonlocal called
        called = True
        raise TimeoutError("direct route timed out")

    result = _run(monkeypatch, timed_out)
    (glyph, _label, detail), = result.lines
    assert "✓" in glyph
    assert "proxy configured" in detail
    assert result.issues == []
    assert called is False


def test_no_proxy_host_still_gets_direct_ipv6_diagnosis(monkeypatch):
    from urllib import request

    monkeypatch.setattr(request, "getproxies", lambda: {"https": "http://127.0.0.1:8080"})
    monkeypatch.setattr(request, "proxy_bypass", lambda host: host == "openrouter.ai")

    def timed_out(_sockaddr, _timeout):
        raise TimeoutError("direct route timed out")

    result = _run(monkeypatch, timed_out)
    (glyph, _label, detail), = result.lines
    assert "⚠" in glyph
    assert "network.force_ipv4" in detail
    assert result.issues and "network.force_ipv4" in result.issues[0]
