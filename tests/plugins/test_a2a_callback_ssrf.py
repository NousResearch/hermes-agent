"""A2A push callbacks must not reach internal addresses by another spelling (#126755).

``is_safe_callback_url`` used to trust the hostname text. Integer and hex IPv4
literals, and names that resolve to a metadata address, were allowed. The POST
also followed redirects, so a public URL could bounce onto those targets. The
guarded opener must dial directly and keep credentials on their original origin.
"""

import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from plugins.platforms.a2a import security


def test_integer_and_hex_ipv4_literals_are_internal(monkeypatch):
    monkeypatch.setenv("A2A_BEARER_TOKEN", "tok")
    assert security.is_safe_callback_url("http://2852039166/latest/meta-data/") is False
    assert security.is_safe_callback_url("http://0x7f000001/") is False
    assert security.is_safe_callback_url("http://0177.0.0.1/") is False
    # A public literal must stay allowed, or every dotted host looks internal.
    assert security.is_safe_callback_url("http://8.8.8.8/hook") is True


def test_dns_answers_classify_embedded_ipv4_before_connect(monkeypatch):
    monkeypatch.setenv("A2A_BEARER_TOKEN", "tok")

    answers = {
        "metadata.invalid": "169.254.169.254",
        "cgnat.invalid": "100.64.0.1",
        "mapped-cgnat.invalid": "::ffff:100.64.0.1",
        "translated-loopback.invalid": "::ffff:0:127.0.0.1",
        "translated-public.invalid": "::ffff:0:8.8.8.8",
    }

    def fake_getaddrinfo(host, port, *args, **kwargs):
        ip = answers[host]
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        sockaddr = (ip, port or 0, 0, 0) if family == socket.AF_INET6 else (ip, port or 0)
        return [(family, socket.SOCK_STREAM, 6, "", sockaddr)]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    for host in (
        "metadata.invalid",
        "cgnat.invalid",
        "mapped-cgnat.invalid",
        "translated-loopback.invalid",
    ):
        assert security.is_safe_callback_url(f"http://{host}/hook") is False
    assert security.is_safe_callback_url("http://translated-public.invalid/hook") is True


def _addrinfo(ip, port):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]


def test_second_lookup_cannot_rebind_the_socket(monkeypatch):
    """The check may see a public address. The POST must not then reach loopback."""
    monkeypatch.setenv("A2A_BEARER_TOKEN", "tok")

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"reached")

        def log_message(self, fmt, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    calls = {"n": 0}

    def fake_getaddrinfo(host, port_arg, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _addrinfo("8.8.8.8", port_arg)
        return _addrinfo("127.0.0.1", port)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    try:
        assert security.is_safe_callback_url("http://rebind.example/hook") is True
        opener = security.callback_opener(localhost_mode=False)
        req = urllib.request.Request(
            "http://rebind.example/hook", data=b"{}", method="POST",
        )
        with pytest.raises(urllib.error.URLError):
            opener.open(req, timeout=2)
    finally:
        server.shutdown()
        server.server_close()


def test_connect_dials_checked_origin_even_with_environment_proxy(monkeypatch):
    monkeypatch.setenv("A2A_BEARER_TOKEN", "tok")
    for name in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("http_proxy", "http://proxy.example:3128")

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return _addrinfo({
            "public.example": "8.8.8.8",
            "proxy.example": "9.9.9.9",
        }[host], port)

    seen = []

    def fake_create_connection(address, *args, **kwargs):
        seen.append(address)
        raise TimeoutError("stopped before the network")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    opener = security.callback_opener(localhost_mode=False)
    req = urllib.request.Request("http://public.example/hook", data=b"{}", method="POST")
    with pytest.raises(urllib.error.URLError):
        opener.open(req, timeout=1)
    assert seen == [("8.8.8.8", 80)]


def test_redirect_onto_metadata_is_refused(monkeypatch):
    monkeypatch.setenv("A2A_BEARER_TOKEN", "tok")
    opener = security.callback_opener(localhost_mode=False)
    handler = next(h for h in opener.handlers if isinstance(h, urllib.request.HTTPRedirectHandler))
    req = urllib.request.Request("https://example.com/cb", method="POST")
    with pytest.raises(urllib.error.HTTPError):
        handler.redirect_request(
            req, None, 302, "Found", {}, "http://169.254.169.254/latest/meta-data/")

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, *args, **kwargs: _addrinfo("8.8.8.8", port),
    )
    signed = urllib.request.Request(
        "https://example.com/cb",
        data=b"{}",
        headers={"Authorization": "Bearer callback-token", "X-A2A-Signature": "digest"},
        method="POST",
    )
    redirected = handler.redirect_request(
        signed, None, 302, "Found", {}, "https://other.example/cb",
    )
    assert redirected is not None
    forwarded = {name.lower() for name, _value in redirected.header_items()}
    assert forwarded.isdisjoint({"authorization", "x-a2a-signature"})

    same_origin = handler.redirect_request(
        signed, None, 302, "Found", {}, "https://example.com/next",
    )
    assert same_origin is not None
    retained = {name.lower() for name, _value in same_origin.header_items()}
    assert {"authorization", "x-a2a-signature"} <= retained
