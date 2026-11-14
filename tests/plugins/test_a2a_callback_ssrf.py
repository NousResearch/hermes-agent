"""A2A push callbacks must not reach an internal address by another spelling (#126755).

The callback check used to compare the host *text* against private prefixes: an integer/hex IPv4
literal or a DNS name pointing at metadata passed, the POST followed redirects the check never saw,
and urllib resolved the name a second time for the socket.
"""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from plugins.platforms.a2a import protocol, security

_real_getaddrinfo = socket.getaddrinfo


class _Recorder(BaseHTTPRequestHandler):
    def _handle(self):
        self.server.hits.append((self.command, self.path))
        if self.server.endless_body:
            self.send_response(200)
            self.send_header("Content-Length", str(1 << 40))  # declared, never sent
            self.end_headers()
            return
        if self.server.redirect_to:
            self.send_response(302)
            self.send_header("Location", self.server.redirect_to)
        else:
            self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = do_POST = _handle

    def log_message(self, *_args):
        return


@pytest.fixture
def serve():
    servers = []

    def start(redirect_to: str = "", endless_body: bool = False) -> HTTPServer:
        server = HTTPServer(("127.0.0.1", 0), _Recorder)
        server.hits, server.redirect_to, server.endless_body = [], redirect_to, endless_body
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)


def _push(monkeypatch, url: str) -> None:
    """Register *url* as a task's callback and fire the push the way a finished task does."""
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter

    adapter = A2AAdapter(PlatformConfig(enabled=True))
    adapter.tasks.create("t1", "c1", "peer")
    assert adapter.tasks.set_push_config("t1", url) is not None
    adapter._send_push_notification("t1", "c1", "done", protocol.STATE_COMPLETED)


def _resolve_as(monkeypatch, answers):
    """Answer the named hosts with the given IPv4 address; other hosts resolve normally."""
    def fake(host, port, *args, **kwargs):
        if host not in answers:
            return _real_getaddrinfo(host, port, *args, **kwargs)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (answers[host], port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)


@pytest.mark.parametrize("url", [
    "http://2852039166/latest/meta-data/",  # 169.254.169.254 as one integer
    "http://0xa9fea9fe/latest/meta-data/",  # ... and as hex
    "http://0x7f000001/",                   # 127.0.0.1 as hex
    "http://metadata.google.internal/computeMetadata/v1/",
])
@pytest.mark.parametrize("localhost_mode", [False, True])
def test_alternate_spellings_of_internal_addresses_are_refused(url, localhost_mode):
    assert security.is_safe_callback_url(url, localhost_mode=localhost_mode) is False


def test_dns_name_is_judged_by_what_it_resolves_to(monkeypatch):
    _resolve_as(monkeypatch, {"meta.example": "169.254.169.254", "lan.example": "10.0.0.5",
                              "public.example": "8.8.8.8"})
    assert security.is_safe_callback_url("http://meta.example/", localhost_mode=False) is False
    assert security.is_safe_callback_url("http://lan.example/", localhost_mode=True) is False
    assert security.is_safe_callback_url("https://public.example/hook", localhost_mode=False) is True


def test_push_follows_the_owning_profiles_private_url_policy(monkeypatch, serve, tmp_path):
    """The push runs on an HTTP handler thread, which carries no profile scope: a secondary
    profile's own ``security.allow_private_urls`` must decide, not the launch profile's."""
    from agent.runtime_scope import bind_state, scope_for_profile
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter

    for var in ("A2A_BEARER_TOKEN", "A2A_PEER_TOKENS", "HERMES_ALLOW_PRIVATE_URLS"):
        monkeypatch.delenv(var, raising=False)
    _resolve_as(monkeypatch, {"receiver.lan": "127.0.0.1"})
    secondary = tmp_path / "profiles" / "sec"
    secondary.mkdir(parents=True)
    (secondary / "config.yaml").write_text("security:\n  allow_private_urls: true\n")
    target = serve()
    tokens = bind_state(scope_for_profile(secondary), hydrate_secrets=False)
    try:
        adapter = A2AAdapter(PlatformConfig(enabled=True))
    finally:
        tokens.reset()
    adapter.tasks.create("t1", "c1", "peer")
    adapter.tasks.set_push_config("t1", f"http://receiver.lan:{target.server_port}/hook")

    pusher = threading.Thread(target=adapter._send_push_notification,
                              args=("t1", "c1", "done", protocol.STATE_COMPLETED))
    pusher.start()
    pusher.join(30)

    assert target.hits == [("POST", "/hook")]


def test_redirect_onto_a_refused_address_is_not_followed(monkeypatch, serve):
    """Localhost-only mode allows a loopback callback; its 302 to a name the check refuses (it
    resolves to an internal address) must not be followed."""
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    _resolve_as(monkeypatch, {"internal.example": "127.0.0.1"})
    assert security.is_safe_callback_url("http://internal.example/", localhost_mode=True) is False
    target = serve()
    redirector = serve(redirect_to=f"http://internal.example:{target.server_port}/latest/meta-data/")
    failed = protocol.metrics.push_failed

    _push(monkeypatch, f"http://127.0.0.1:{redirector.server_port}/hook")

    assert redirector.hits == [("POST", "/hook")]
    assert target.hits == []
    assert protocol.metrics.push_failed == failed + 1


def test_callback_name_cannot_rebind_to_loopback_between_check_and_connect(monkeypatch, serve):
    """The name answers public to the port-less pre-flight lookup and loopback to the lookup made
    for the socket; the POST must not land on loopback."""
    monkeypatch.setenv("A2A_BEARER_TOKEN", "tok")  # remote mode: loopback is never a valid callback
    target = serve()

    def fake(host, port, *args, **kwargs):
        if host != "rebind.example":
            return _real_getaddrinfo(host, port, *args, **kwargs)
        ip = "127.0.0.1" if port else "8.8.8.8"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    assert security.is_safe_callback_url("http://rebind.example/hook", localhost_mode=False) is True

    _push(monkeypatch, f"http://rebind.example:{target.server_port}/hook")

    assert target.hits == []


def test_push_reads_the_status_and_never_the_body(monkeypatch, serve):
    """The callback is the peer's server: a 200 that declares a 1 TiB body must count as
    delivered without downloading it (reading it pins the push thread and grows memory)."""
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    target = serve(endless_body=True)
    sent, failed = protocol.metrics.push_sent, protocol.metrics.push_failed

    _push(monkeypatch, f"http://127.0.0.1:{target.server_port}/hook")

    assert target.hits == [("POST", "/hook")]
    assert (protocol.metrics.push_sent, protocol.metrics.push_failed) == (sent + 1, failed)
