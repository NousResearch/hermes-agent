"""Provider logins (Codex browser, OpenRouter, PKCE plugins) share one loopback listener; an idle
browser preconnect accepted ahead of the redirect must not hold the authorization code."""
import socket
import threading
import time
from http.client import HTTPConnection

from hermes_cli.auth import AuthError
from hermes_cli.auth_device_flow import (
    _bind_loopback_callback_server, _make_loopback_callback_handler, _serve_loopback_callback)


def test_idle_preconnect_does_not_delay_the_provider_login_code():
    handler, result = _make_loopback_callback_handler("/callback", display_name="Probe")
    server = _bind_loopback_callback_server("127.0.0.1", 0, handler, err=AuthError, bind_failed_code="bind")
    port = server.server_address[1]
    out: dict = {}

    def run():
        out["result"] = _serve_loopback_callback(server, result, timeout_seconds=30, err=AuthError, timeout_code="t")
        out["returned_at"] = time.monotonic()

    waiter = threading.Thread(target=run, daemon=True)
    waiter.start()
    idle = socket.create_connection(("127.0.0.1", port), timeout=5)  # the browser's silent speculative socket
    try:
        # The accept queue is FIFO, so the serve loop takes the idle socket ahead of the redirect.
        sent_at = time.monotonic()
        conn = HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/callback?code=abc&state=xyz")
        assert conn.getresponse().status == 200
        conn.close()
        waiter.join(timeout=8)
        assert not waiter.is_alive(), "listener still parked on the idle preconnect"
        assert out["result"]["code"] == "abc"
        assert out["returned_at"] - sent_at < 5
    finally:
        idle.close()
