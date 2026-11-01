"""Relayed sockets: a browser's socket on the hub reaches a paired gateway over a real connection.

The gateway here is this same app served by a real uvicorn on loopback, so the hub dials it the way
it dials another machine (``webapp_relay_proxy.relay_ws``): the browser presents the hub's ``relay``
ticket, the gateway's own ticket minted through the relay rides along, frames cross both ways, and
the gateway closing the socket (its grant was revoked) closes the browser's too.
"""
import json
import socket
import threading
import time
import urllib.parse
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli.web_routers import webapp_relay_proxy
# Fixtures: the relay app (one app plays hub and gateway), and the voice streams' stand-in providers.
from tests.hermes_cli.test_webapp_relay import ORIGIN, _pair, _relay_app, _sign_in
from tests.hermes_cli.test_webapp_socket_tickets import voice

_REAL_RELAY_CLIENT = webapp_relay_proxy.relay_client
_SPEAK = "/api/audio/speak-stream"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def live_gateway(relay_app, monkeypatch):
    """The relay app served on a real loopback port; the hub reaches it over the network."""
    import uvicorn

    monkeypatch.setattr(webapp_relay_proxy, "relay_client", _REAL_RELAY_CLIENT)
    state = relay_app.server.app.state
    monkeypatch.setattr(state, "trusted_public_hosts", frozenset({"hermes.lan", "127.0.0.1"}), raising=False)
    # One app plays both gateways, so it has no single public URL: each origin checks its own Host.
    monkeypatch.setattr(state, "listener_public_url", None, raising=False)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(relay_app.server.app, host="127.0.0.1", port=port, lifespan="off",
                                           log_level="warning", ws="websockets"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        assert time.monotonic() < deadline and thread.is_alive(), "loopback gateway did not start"
        time.sleep(0.02)
    try:
        yield SimpleNamespace(app=relay_app, url=f"http://127.0.0.1:{port}")
    finally:
        server.should_exit = True
        thread.join(10)


def _pair_live(live, scopes=("voice",)):
    """Pair the hub with the loopback gateway, approving on its consent page as the same account."""
    hub = live.app.browser()
    _sign_in(hub)
    there = TestClient(live.app.server.app, base_url=live.url)
    _sign_in(there)
    started = hub.post("/api/webapp/relay/gateways", json={"url": live.url, "label": "Loopback"}, headers=ORIGIN)
    assert started.status_code == 200, started.text
    authorize = urllib.parse.urlsplit(started.json()["authorize_url"])
    form = dict(urllib.parse.parse_qsl(authorize.query))
    decided = there.post(authorize.path, data={**form, "decision": "allow", "grant_scope": list(scopes)},
                         headers={"Origin": live.url}, follow_redirects=False)
    assert decided.status_code == 303, decided.text
    callback = urllib.parse.urlsplit(decided.headers["location"])
    back = hub.get(f"{callback.path}?{callback.query}", follow_redirects=False)
    assert "relay=connected" in back.headers["location"], back.headers["location"]
    row = hub.get("/api/webapp/relay/gateways").json()["gateways"][0]
    return hub, there, row


def _relayed_url(hub, row, path, socket_name):
    far = hub.post(f"/api/relay/{row['id']}/api/webapp/socket-ticket?socket={socket_name}", headers=ORIGIN)
    assert far.status_code == 200, far.text
    near = hub.post("/api/webapp/socket-ticket?socket=relay", headers=ORIGIN)
    assert near.status_code == 200, near.text
    return (f"ws://hermes.lan:9120/api/relay/{row['id']}{path}"
            f"?ticket={far.json()['ticket']}&relay_ticket={near.json()['ticket']}")


@pytest.mark.usefixtures("voice")
def test_a_relayed_voice_stream_carries_frames_both_ways(live_gateway):
    hub, _there, row = _pair_live(live_gateway)
    with hub.websocket_connect(_relayed_url(hub, row, _SPEAK, "speak-stream"), headers=ORIGIN) as ws:
        ws.send_text(json.dumps({"text": "Hello there.", "done": True}))
        assert ws.receive_json() == {"type": "start", "sample_rate": 24000, "channels": 1}
        assert ws.receive_bytes() == b"\x01\x02"


@pytest.mark.usefixtures("voice")
def test_a_relayed_socket_opens_only_with_the_hubs_relay_ticket(live_gateway):
    hub, _there, row = _pair_live(live_gateway)
    url = _relayed_url(hub, row, _SPEAK, "speak-stream")
    bare = url.split("&relay_ticket=")[0]
    for attempt in (bare, url.replace("relay_ticket=", "relay_ticket=x")):
        with pytest.raises(WebSocketDisconnect) as refused:
            with hub.websocket_connect(attempt, headers=ORIGIN) as ws:
                ws.receive_text()
        assert refused.value.code == 4401


@pytest.mark.usefixtures("voice")
def test_revoking_the_grant_closes_the_relayed_socket(live_gateway):
    hub, there, row = _pair_live(live_gateway)
    with hub.websocket_connect(_relayed_url(hub, row, _SPEAK, "speak-stream"), headers=ORIGIN) as ws:
        grant = there.get("/api/webapp/relay/grants").json()["grants"][0]
        assert there.delete(f"/api/webapp/relay/grants/{grant['id']}",
                            headers={"Origin": live_gateway.url}).status_code == 200
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
        # The gateway's own close, passed through: not the hub refusing or dropping the socket.
        assert (closed.value.code, closed.value.reason) == (4401, "Gateway relay revoked")


def _rpc(ws, rid, method, params=None):
    ws.send_text(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}))
    while (frame := ws.receive_json()).get("id") != rid:
        pass  # events (gateway.ready, ...) arrive between replies
    return frame


def _refused(frame) -> bool:
    error = frame.get("error") or {}
    return error.get("code") == 4030 and "not granted" in error.get("message", "")


#: A call each optional scope must open: the scope, the method and its params.
_SCOPED_CALLS = (
    ("terminal", "shell.exec", {"command": "echo relayed"}),
    ("terminal", "cli.exec", {"argv": ["config", "set", "terminal.backend", "local"]}),
    ("display", "display.thumbnail", {}), ("display", "display.lease.acquire", {}),
    ("voice", "voice.record", {}), ("voice", "wake.start", {}))


@pytest.mark.parametrize("scopes", [(), ("display",), ("terminal",)])
def test_a_relayed_gateway_socket_reaches_scoped_methods_only_with_their_scope(live_gateway, scopes):
    # The gateway's own socket also runs shell and CLI commands, drives the Bot Screen and records
    # voice, so a grant without those scopes must not reach them there either.
    hub, _there, row = _pair_live(live_gateway, scopes=scopes)
    with hub.websocket_connect(_relayed_url(hub, row, "/api/ws", "gateway"), headers=ORIGIN) as ws:
        for rid, (scope, method, params) in enumerate(_SCOPED_CALLS, start=1):
            if scope not in scopes:
                assert _refused(_rpc(ws, rid, method, params)), method
        if "display" in scopes:  # read-only, so it may run here
            assert not _refused(_rpc(ws, 90, "display.thumbnail"))
        if "terminal" in scopes:
            ran = _rpc(ws, 91, "shell.exec", {"command": "echo relayed"})
            assert ran.get("result", {}).get("stdout", "").strip() == "relayed", ran


def test_a_relayed_gateway_socket_without_terminal_access_still_edits_bot_profiles(live_gateway):
    # Bot Mode describes, deletes and resets a bot's profile through cli.exec. The profile and
    # settings routes every grant reaches make the same changes, so those command lines stay open.
    hub, _there, row = _pair_live(live_gateway, scopes=())
    with hub.websocket_connect(_relayed_url(hub, row, "/api/ws", "gateway"), headers=ORIGIN) as ws:
        edit = _rpc(ws, 1, "cli.exec", {"argv": ["profile", "describe", "no-such-bot", "--text", "Helps"]})
        assert edit.get("result", {}).get("blocked") is False, edit  # the CLI ran it
        for rid, argv in enumerate((["profile", "describe", "no-such-bot", "--text", "Helps", "--yes"],
                                    ["profile", "delete", "--help", "--yes"], ["--version"]), start=2):
            assert _refused(_rpc(ws, rid, "cli.exec", {"argv": argv})), argv


def test_a_socket_opened_under_a_grant_needs_that_grants_scope(relay_app):
    """The upgrade never crosses the relayed-request gate, so a ticket minted through the relay is
    checked again against the socket it opens."""
    from hermes_cli import webapp_relay_grants

    hub = relay_app.browser()
    _sign_in(hub)
    _pair(hub, scopes=("voice",))
    grant = webapp_relay_grants.list_grants(relay_app.home, {"user_id": "stub-user-1", "provider": "stub"})[0]

    def opened(path):
        ws = SimpleNamespace(app=relay_app.server.app, url=SimpleNamespace(path=path))
        return webapp_relay_grants.admit_relayed_socket(ws, grant["id"])

    assert opened("/api/ws") is None and opened(_SPEAK) is None
    for path in ("/api/pty", "/api/console", "/api/host-terminal", "/api/display/ws"):
        assert "not granted" in (opened(path) or ""), path
    webapp_relay_grants.revoke(relay_app.home, grant["id"])
    assert "revoked" in (opened("/api/ws") or "")
