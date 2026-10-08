"""A2A push callbacks must not reach internal addresses by another spelling (#126755).

``is_safe_callback_url`` used to trust the hostname text. Integer and hex IPv4
literals, and names that resolve to a metadata address, were allowed. The POST
also followed redirects, so a public URL could bounce onto those targets.
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


def test_dns_name_that_resolves_to_metadata_is_blocked(monkeypatch):
    monkeypatch.setenv("A2A_BEARER_TOKEN", "tok")

    def fake_getaddrinfo(host, port, *args, **kwargs):
        assert host == "metadata.invalid"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert security.is_safe_callback_url("http://metadata.invalid/latest/meta-data/") is False


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


def test_connect_dials_the_address_it_just_checked(monkeypatch):
    monkeypatch.setenv("A2A_BEARER_TOKEN", "tok")

    def fake_getaddrinfo(host, port, *args, **kwargs):
        assert host == "public.example"
        return _addrinfo("8.8.8.8", port)

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
