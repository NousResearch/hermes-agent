"""``hermes peer`` reads a peer's "no such profile" 404 by its typed code, not by its English sentence.

The ``/p/<profile>/`` prefix middleware answers a profile the gateway does not have with
``{"error": "Unknown or unconfigured profile", "code": "profile_unknown"}``. Clients key on the code,
so rewording the sentence cannot change where ``box/hermes`` goes, and another code beside the old
sentence is not the verdict. A peer older than the code sends the sentence alone, which still counts.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli.subcommands import peer as peer_cmd

_SENTENCE = "Unknown or unconfigured profile"


class _RefusingPeer(BaseHTTPRequestHandler):
    """A peer serving no ``/p/hermes`` profile, answering every request with one 404 body."""

    body: dict[str, Any] = {}
    hits: list[str] = []

    def _reply(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).hits.append(self.path)
        payload = json.dumps(type(self).body).encode()
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = _reply

    def log_message(self, format, *args):
        pass


_AMBIGUOUS = "box/default"
_RELAYED = "rejected the request (HTTP 404)"


@pytest.mark.parametrize("body,verdict", [
    # Typed verdict: the code decides, whatever the sentence says.
    ({"error": "No profile by that name here", "code": "profile_unknown"}, _AMBIGUOUS),
    ({"error": _SENTENCE, "code": "profile_unknown"}, _AMBIGUOUS),
    # The old sentence beside another code is not the verdict.
    ({"error": _SENTENCE, "code": "route_disabled"}, _RELAYED),
    # A peer older than the code: the exact sentence alone still counts.
    ({"error": _SENTENCE}, _AMBIGUOUS),
    ({"error": "Unknown profile"}, _RELAYED),
])
def test_hermes_handle_reads_the_peers_typed_unknown_profile_verdict(monkeypatch, capsys, body, verdict):
    _RefusingPeer.body, _RefusingPeer.hits = body, []
    server = HTTPServer(("127.0.0.1", 0), _RefusingPeer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: {"box": {"url": f"http://127.0.0.1:{server.server_port}"}})
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda name: "secret-key-123456")
    try:
        rc = peer_cmd.cmd_peer(SimpleNamespace(peer_action="dm", target="box/hermes", message="hi", json=True))
    finally:
        server.shutdown()
    err = capsys.readouterr().err
    assert rc == 1
    other = _RELAYED if verdict == _AMBIGUOUS else _AMBIGUOUS
    assert verdict in err and other not in err, err
    # Every request addressed the profile asked for: nothing is ever rerouted to /p/default.
    assert {hit.split("/")[2] for hit in _RefusingPeer.hits} == {"hermes"}


@pytest.mark.asyncio
async def test_the_prefix_middleware_types_its_unknown_profile_404(tmp_path, monkeypatch):
    """The real middleware's body carries the code AND the unchanged sentence older clients match."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 8642, "key": "k"}))

    class _Runner:
        config = GatewayConfig(multiplex_profiles=True)

    adapter.gateway_runner = _Runner()

    async def handler(request):
        return web.json_response({"ok": True})

    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_get("/p/{profile}/health", handler)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/p/nosuch/health")
        assert (resp.status, await resp.json()) == (404, {"error": _SENTENCE, "code": "profile_unknown"})
