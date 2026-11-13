"""Team Gateway account links minted from the instance (#141546).

A relay-exclusive instance reaches Telegram only through the shared bot, so the Channels surfaces
mint ``/link <code>`` through the gateway's ``relay-link`` verb, which alone holds the per-gateway
secret. The stand-in connector below checks the bearer the way the connector does (gatewayId
peeked from the token, HMAC over ``payload:exp`` with the stored secret).
"""
import base64
import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

_SECRET = "per-gateway-secret"
_GATEWAY_ID = "gw-test"


class _Connector(BaseHTTPRequestHandler):
    seen: list = []
    managed_by = None  # None: a connector without the access route (404)

    def log_message(self, format, *args):
        pass

    def _reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _bearer_valid(self) -> bool:
        token = self.headers.get("Authorization", "").removeprefix("Bearer ")
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()
        signed, _, sig = raw.rpartition(":")
        return hmac.compare_digest(sig, hmac.new(_SECRET.encode(), signed.encode(), hashlib.sha256).hexdigest())

    def do_GET(self):  # /manage/access/list: who governs this agent's access
        if not self._bearer_valid():
            return self._reply(401, {"error": "bad bearer"})
        if _Connector.managed_by is None:
            return self._reply(404, {"error": "not found"})
        self._reply(200, {"agent": "inst-1", "managedBy": _Connector.managed_by, "grants": []})

    def do_POST(self):
        token = self.headers.get("Authorization", "").removeprefix("Bearer ")
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()
        signed, _, sig = raw.rpartition(":")
        valid = hmac.compare_digest(sig, hmac.new(_SECRET.encode(), signed.encode(), hashlib.sha256).hexdigest())
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Connector.seen.append((self.path, signed.split(":")[0], body))
        self._reply(*((401, {"error": "bad bearer"}) if not valid else (
            200, {"code": "AB12CD", "expiresAt": "2026-11-03T00:10:00.000Z"})))


@pytest.fixture
def connector(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Connector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Connector.seen = []
    _Connector.managed_by = None
    monkeypatch.setenv("GATEWAY_RELAY_URL", f"ws://127.0.0.1:{server.server_port}/relay")
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "telegram")
    monkeypatch.setenv("GATEWAY_RELAY_BOT_IDS", json.dumps({"telegram": {"botId": "42", "username": "@NousHermesBot"}}))
    monkeypatch.setenv("GATEWAY_RELAY_ID", _GATEWAY_ID)
    yield _Connector
    server.shutdown()


def test_link_code_is_minted_with_the_gateway_bearer_and_points_at_the_shared_bot(connector, monkeypatch):
    from gateway.relay.link import relay_link_verb

    monkeypatch.delenv("GATEWAY_RELAY_SECRET", raising=False)
    not_ready = relay_link_verb({"action": "request", "platform": "telegram"})
    assert not_ready["ok"] is False and not_ready["error"] == "relay_not_provisioned"
    assert connector.seen == []

    monkeypatch.setenv("GATEWAY_RELAY_SECRET", _SECRET)
    result = relay_link_verb({"action": "request", "platform": "telegram"})

    assert result["ok"] is True
    assert result["command"] == "/link AB12CD"
    assert result["bot_url"] == "https://t.me/NousHermesBot"
    assert connector.seen == [("/manage/link", _GATEWAY_ID, {"platform": "telegram"})]
    # A platform the shared bot does not carry never reaches the connector.
    assert relay_link_verb({"action": "request", "platform": "slack"})["error"] == "platform_not_fronted"
    assert len(connector.seen) == 1
    # A bearer the connector rejects is named as such, never shown as the connector's raw text.
    monkeypatch.setenv("GATEWAY_RELAY_SECRET", "rotated-elsewhere")
    refused = relay_link_verb({"action": "request", "platform": "telegram"})
    assert refused["error"] == "relay_not_authorized" and refused["status"] == 401


def test_portal_managed_agent_is_sent_to_the_portal_instead_of_a_code_that_binds_nothing(connector, monkeypatch):
    """On a Portal-managed agent a gateway-bearer code carries no owner, so its redeem binds no
    account (connector scopeService: managed agents take only the Portal setting)."""
    from gateway.relay.link import relay_link_verb

    monkeypatch.setenv("GATEWAY_RELAY_SECRET", _SECRET)
    connector.managed_by = "portal"

    refused = relay_link_verb({"action": "request", "platform": "telegram"})
    status = relay_link_verb({"action": "status"})

    assert refused["ok"] is False and refused["error"] == "managed_by_portal"
    assert [path for path, *_ in connector.seen] == ["/manage/link-status"]
    assert status["managed_by"] == "portal" and status["portal_url"].endswith("/agents")
