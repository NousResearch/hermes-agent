"""Every credential-bearing caller drops its secret when the server redirects to another origin.

Wire-level: a real source server answers 302 to a real sink on a different origin, and the
sink records what actually arrived.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest


class _Handler(BaseHTTPRequestHandler):
    redirect_to = ""
    seen: list[dict[str, str]] = []

    def _answer(self):
        if self.command == "POST":
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path.startswith("/redirect"):
            self.send_response(302)
            self.send_header("Location", type(self).redirect_to)
            self.end_headers()
            return
        type(self).seen.append({k.lower(): v for k, v in self.headers.items()})
        body = json.dumps({}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _answer

    def log_message(self, format, *args):
        pass


def _nous_account(base):
    from hermes_cli import nous_account
    nous_account._fetch_nous_account_info("portal-secret", portal_base_url=base)


def _dashboard_register(base):
    from hermes_cli import dashboard_register
    dashboard_register._register_self_hosted_client(
        access_token="portal-secret", portal_base_url=base, name="n", custom_redirect_uri=None)


def _gateway_enroll(base):
    from hermes_cli import gateway_enroll
    gateway_enroll._post_enroll(connector_base_url=base, access_token="portal-secret",
                                enrollment_token="enroll-secret", gateway_id="gw-1")


def _relay_post(base):
    from gateway import relay
    relay._json_post(f"{base}/relay/provision", "relay-secret", {}, 3).close()


def _a2a_peer(base):
    from plugins.platforms.a2a import tools
    tools._http_json(f"{base}/a2a", {"Authorization": "Bearer peer-secret"}, 3, "GET")


def _voice_live(base, monkeypatch):
    from tools import voice_live
    monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: ("sk-secret", base))
    monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: {})
    voice_live.create_webrtc_session("v=0")


@pytest.mark.parametrize("call", [
    _nous_account, _dashboard_register, _gateway_enroll, _relay_post, _a2a_peer, _voice_live,
], ids=lambda f: f.__name__.lstrip("_"))
def test_cross_origin_redirect_never_carries_the_credential(call, monkeypatch):
    source = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    sink = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    for server in (source, sink):
        Thread(target=server.serve_forever, daemon=True).start()
    _Handler.seen = []
    # localhost vs 127.0.0.1: a different origin on the same machine.
    _Handler.redirect_to = f"http://localhost:{sink.server_port}/sink"
    base = f"http://127.0.0.1:{source.server_port}/redirect"
    try:
        try:
            if call is _voice_live:
                call(base, monkeypatch)
            else:
                call(base)
        except Exception:
            pass  # the sink's empty answer may not parse for the caller; only the wire matters
    finally:
        source.shutdown()
        sink.shutdown()

    assert _Handler.seen, "the redirect was never followed to the sink"
    for headers in _Handler.seen:
        assert "authorization" not in headers
        assert not any("secret" in value for value in headers.values())
