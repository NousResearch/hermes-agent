"""A remote client that sends a ticket subprotocol without ``Origin`` reaches the Webapp's ticket gate (#147361).

The Android app (ktor) dials ``/api/ws`` from a remote peer with ``hermes-gateway-ticket.<ticket>`` and no
``Origin``. The runtime wrapper took that for a native local Desktop dial and closed it 4403 before the
Webapp ever saw the ticket its own ``/api/auth/ws-ticket`` had minted.
"""
from types import SimpleNamespace

import pytest

from gateway.run_api import GatewayRuntimeAPI

_TICKET_PROTOCOLS = "hermes-gateway-v1, hermes-gateway-ticket.webapp-minted"


class _NativeTickets:
    def __init__(self):
        self.redeemed = []

    def redeem(self, ticket, *, profile_id, purpose):
        self.redeemed.append((ticket, purpose))
        raise PermissionError("not a native grant")


async def _dial(peer, *, protocols=_TICKET_PROTOCOLS, origin=None):
    tickets = _NativeTickets()
    runner = SimpleNamespace(_draining=False, _running=True, session_runtime_descriptor={"state": "ready"},
                             session_ticket_store=tickets)
    reached_webapp, sent = [], []

    async def webapp(scope, receive, send):
        reached_webapp.append(scope["client"])

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    headers = [(b"host", b"hermes.example.lan"), (b"sec-websocket-protocol", protocols.encode())]
    if origin:
        headers.append((b"origin", origin.encode()))
    scope = {"type": "websocket", "path": "/api/ws", "scheme": "ws", "query_string": b"",
             "client": (peer, 40000), "server": ("0.0.0.0", 9119), "headers": headers,
             "subprotocols": [p.strip() for p in protocols.split(",")]}
    await GatewayRuntimeAPI(webapp, runner, SimpleNamespace(state=None))(scope, receive, send)
    closes = [m.get("code") for m in sent if m["type"] == "websocket.close"]
    return reached_webapp, closes, tickets.redeemed


@pytest.mark.asyncio
async def test_remote_originless_ticket_reaches_the_webapp_gate():
    reached, closes, redeemed = await _dial("192.168.178.206")
    assert reached == [("192.168.178.206", 40000)] and closes == []
    assert redeemed == []  # a remote peer never redeems a native grant


@pytest.mark.asyncio
async def test_remote_originless_dial_matches_the_same_dial_with_an_origin():
    with_origin = await _dial("192.168.178.206", origin="https://hermes.example.lan")
    assert await _dial("192.168.178.206") == with_origin


@pytest.mark.asyncio
async def test_loopback_ticket_still_takes_the_native_grant_first():
    reached, closes, redeemed = await _dial("127.0.0.1")
    assert closes == []
    assert [purpose for _ticket, purpose in redeemed] == ["interactive", "worker-adoption"]
    assert reached == [("127.0.0.1", 40000)]  # neither grant matched: the Webapp's issuer, as before


@pytest.mark.asyncio
async def test_remote_malformed_ticket_set_is_still_refused():
    reached, closes, redeemed = await _dial("192.168.178.206", protocols="hermes-gateway-ticket.no-base")
    assert reached == [] and closes == [4403] and redeemed == []
