"""Regression test for hermes_cli.gateway_enroll._post_enroll()'s Bearer
identity token not following a cross-host redirect.

Mirrors tests/gateway/relay/test_self_provision.py's redirect coverage for
gateway.relay._post_provision() / _post_policy() — _post_enroll() sends the
caller's Nous Portal access token via the same raw-urlopen shape those had
before being routed through hermes_cli.urllib_security.open_credentialed_url().
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from hermes_cli import gateway_enroll


class _RedirectingEnrollHandler(BaseHTTPRequestHandler):
    """Answers POST /relay/enroll with a 302 to a configurable target and
    records the headers the target receives — proves the caller's Bearer
    access token never reaches an unintended origin. 302 (not 307/308):
    stdlib's HTTPRedirectHandler only follows a POST redirect for
    301/302/303."""

    redirect_to = ""
    received_headers: dict = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path.rstrip("/") == "/relay/enroll":
            self.send_response(302)
            self.send_header("Location", type(self).redirect_to)
            self.end_headers()
        else:
            self._respond()

    def do_GET(self):
        self._respond()

    def _respond(self):
        type(self).received_headers = dict(self.headers)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"secret": "leaked"}).encode())

    def log_message(self, format, *args):
        pass


def test_post_enroll_strips_bearer_on_cross_host_redirect():
    """The caller's Nous Portal access token must not follow a redirect to a
    different origin — a compromised or misconfigured proxy in front of the
    connector could otherwise redirect the enroll POST to an
    attacker-controlled host and receive the token. The redirect is still
    followed (a legitimate reachability concern), just without the
    credential."""
    _RedirectingEnrollHandler.received_headers = {}
    server = HTTPServer(("127.0.0.1", 0), _RedirectingEnrollHandler)
    target_server = HTTPServer(("127.0.0.1", 0), _RedirectingEnrollHandler)
    port = server.server_address[1]
    target_port = target_server.server_address[1]
    _RedirectingEnrollHandler.redirect_to = f"http://127.0.0.1:{target_port}/collect"
    Thread(target=server.serve_forever, daemon=True).start()
    Thread(target=target_server.serve_forever, daemon=True).start()

    try:
        result = gateway_enroll._post_enroll(
            connector_base_url=f"http://127.0.0.1:{port}",
            access_token="super-secret-bearer",
            enrollment_token="enroll-tok",
            gateway_id="gw-1",
        )
    finally:
        server.shutdown()
        target_server.shutdown()

    # The redirect target answered normally (it isn't blocked), proving the
    # request really was followed — but without the Bearer token attached.
    assert result["secret"] == "leaked"
    headers = {k.lower(): v for k, v in _RedirectingEnrollHandler.received_headers.items()}
    assert "authorization" not in headers
