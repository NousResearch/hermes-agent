"""POST /api/webhooks/{name}/test signs server-side so the real adapter accepts it, and each test is
its own delivery (a fresh X-Request-ID the adapter reads as the delivery id)."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock

import pytest
from multidict import CIMultiDict

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _header_delivery_id


@pytest.fixture
def receiver():
    """A loopback HTTP receiver that records each POST's path, headers and body and answers 202."""
    seen = []
    answer = {"code": 202, "body": b'{"status": "accepted"}'}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            seen.append((self.path, CIMultiDict(self.headers.items()), body))
            self.send_response(answer["code"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(answer["body"])

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1], seen, answer
    server.shutdown()
    server.server_close()


@pytest.fixture
def client(_isolate_hermes_home, receiver):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.config import load_config, save_config
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    port, _, _ = receiver
    cfg = load_config()
    cfg.setdefault("platforms", {})["webhook"] = {"enabled": True, "extra": {"host": "127.0.0.1", "port": port}}
    save_config(cfg)
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _adapter_accepts(route: dict, headers, body: bytes) -> bool:
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "routes": {}}))
    request = MagicMock()
    request.headers = headers
    request.match_info = {"route_name": "deploy"}
    return adapter._validate_signature(request, body, route["secret"], route.get("signature"))


@pytest.mark.parametrize("pin", ["generic", "github"])
def test_send_test_posts_a_signed_unique_delivery(client, receiver, pin):
    from hermes_cli.webhook import _load_subscriptions

    _, seen, _ = receiver
    assert client.post("/api/webhooks", json={"name": "deploy", "signature": pin}).status_code == 200

    first = client.post("/api/webhooks/deploy/test")
    second = client.post("/api/webhooks/deploy/test")

    assert first.status_code == 200, first.text
    assert first.json()["ok"] is True and first.json()["status"] == 202
    route = _load_subscriptions()["deploy"]
    assert [path for path, _, _ in seen] == ["/webhooks/deploy", "/webhooks/deploy"]
    for (_, headers, body), response in zip(seen, (first, second)):
        assert _adapter_accepts(route, headers, body)
        assert _header_delivery_id(headers) == response.json()["delivery_id"]
    assert first.json()["delivery_id"] != second.json()["delivery_id"]
    assert route["secret"] not in first.text


def test_send_test_reports_unknown_route_and_unreachable_receiver(client, receiver):
    from hermes_cli.config import load_config, save_config

    assert client.post("/api/webhooks/missing/test").status_code == 404
    assert client.post("/api/webhooks", json={"name": "deploy"}).status_code == 200
    cfg = load_config()
    cfg["platforms"]["webhook"]["extra"]["port"] = 1  # nothing listens on a privileged port here
    save_config(cfg)
    r = client.post("/api/webhooks/deploy/test")
    assert r.status_code == 502
    assert "gateway" in r.json()["detail"].lower()


def test_send_test_reports_an_ignored_event_as_not_delivered(client, receiver):
    """The adapter answers 200 ``ignored`` when the route's event filter drops ``test``: nothing ran."""
    _, _, answer = receiver
    answer.update(code=200, body=b'{"status": "ignored", "event": "test"}')
    assert client.post("/api/webhooks", json={"name": "deploy", "events": ["push"]}).status_code == 200

    r = client.post("/api/webhooks/deploy/test").json()

    assert r["status"] == 200
    assert r["outcome"] == "ignored"
    assert r["ok"] is False
