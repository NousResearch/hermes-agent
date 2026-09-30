"""Regression tests for browser/gateway.py — the JS<->Python transport
surface that runs under Pyodide.

The `js` module does not exist under CPython, so tests inject a fake
``js.hermesBridge`` recording every call the worker bridge would have
forwarded to the page. These pin the invariants the browser runtime
depends on:

- ws frames route to the right logical socket; replies go out through
  the bridge;
- REST frames become ASGI invocations of the real app object and reply
  with status/headers/body;
- fetch_blocking round-trips through the bridge (the page's fetch()
  answers via a `net-resp` ring frame).
"""

import base64
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from browser import gateway as gw  # noqa: E402
from browser import runtime as br  # noqa: E402


class _Bridge:
    """Stand-in for js.hermesBridge — records calls like worker.mjs would."""

    def __init__(self):
        self.ws_events = []      # (ws_id, event, payload)
        self.rest_replies = []   # (id, status, headers, bodyB64)
        self.fetch_requests = [] # (id, url, method, headers, bodyB64)
        self.logs = []
        self.pump_frames = []    # frames returned by pump()

    def pump(self, ms):
        frames, self.pump_frames = self.pump_frames, []
        return frames

    def emit(self, ws_id, text):
        self.ws_events.append((ws_id, "message", text))

    def wsAccepted(self, ws_id):
        self.ws_events.append((ws_id, "open"))

    def wsClosed(self, ws_id, code, reason):
        self.ws_events.append((ws_id, "close", code))

    def restReply(self, rid, status, headers_json, body_b64):
        self.rest_replies.append((rid, status, json.loads(headers_json), body_b64))

    def fetchRequest(self, req_id, url, method, headers_json, body_b64):
        self.fetch_requests.append((req_id, url, method, json.loads(headers_json), body_b64))

    def log(self, level, msg):
        self.logs.append((level, msg))


@pytest.fixture
def bridge(monkeypatch):
    b = _Bridge()
    js_mod = types.ModuleType("js")
    js_mod.hermesBridge = b
    monkeypatch.setitem(sys.modules, "js", js_mod)
    gw.install()
    return b


@pytest.fixture
def stub_app(monkeypatch):
    """Install a stub ASGI app in place of hermes_cli.web_server.app."""

    async def app(scope, receive, send):
        if scope["type"] == "websocket":
            return
        msg = await receive()
        assert msg["type"] == "http.request"
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        body = json.dumps({"path": scope["path"], "method": scope["method"]}).encode()
        await send({"type": "http.response.body", "body": body})

    monkeypatch.setattr(gw, "_app", app)
    return app


class TestWebSocketGlue:
    def test_ws_open_registers_socket_and_schedules_handler(self, bridge):
        gw.ws_open(7, "/api/ws?x=1", {"host": "localhost"})
        assert 7 in gw._sockets
        ws = gw._sockets[7]
        assert ws.query_params == {"x": "1"}
        assert ws.headers["host"] == "localhost"

    def test_ws_send_feeds_socket_queue(self, bridge):
        ws = gw._FakeWS(3)
        gw._sockets[3] = ws
        gw.ws_send(3, '{"id":1}')
        assert ws._inq.get_nowait() == '{"id":1}'

    def test_ws_send_unknown_id_is_noop(self, bridge):
        gw.ws_send(999, "x")  # must not raise

    def test_ws_close_feeds_disconnect(self, bridge):
        ws = gw._FakeWS(4)
        gw._sockets[4] = ws
        gw.ws_close(4)
        assert 4 not in gw._sockets
        item = ws._inq.get_nowait()
        assert isinstance(item, BaseException)

    def test_send_text_goes_to_bridge_emit(self, bridge):
        ws = gw._FakeWS(5)
        coro = ws.send_text("hello")
        try:
            coro.send(None)
        except StopIteration:
            pass
        assert (5, "message", "hello") in bridge.ws_events

    def test_handle_routes_ws_frames(self, bridge):
        ws = gw._FakeWS(9)
        gw._sockets[9] = ws
        gw.handle(json.dumps({"t": "ws-send", "id": 9, "data": "ping"}))
        assert ws._inq.get_nowait() == "ping"

    def test_handle_tolerates_bad_json(self, bridge):
        gw.handle("not json")
        gw.handle('{"t":"unknown-kind"}')


class TestRestBridge:
    def test_rest_frame_invokes_asgi_app_and_replies(self, bridge, stub_app):
        br.get_loop()
        gw.handle(json.dumps({
            "t": "rest", "id": 42, "method": "GET",
            "path": "/api/profiles?p=1", "headers": {"host": "x"},
        }))
        # Task is queued on the cooperative loop; drain it.
        for _ in range(50):
            br.pump_once(0)
            if bridge.rest_replies:
                break
        assert len(bridge.rest_replies) == 1
        rid, status, headers, body_b64 = bridge.rest_replies[0]
        assert rid == 42 and status == 200
        assert headers["content-type"] == "application/json"
        payload = json.loads(base64.b64decode(body_b64))
        assert payload == {"path": "/api/profiles", "method": "GET"}

    def test_rest_reply_on_app_exception(self, bridge, monkeypatch):
        async def boom(scope, receive, send):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(gw, "_app", boom)
        br.get_loop()
        gw.handle(json.dumps({"t": "rest", "id": 7, "method": "GET",
                              "path": "/x", "headers": {}}))
        for _ in range(50):
            br.pump_once(0)
            if bridge.rest_replies:
                break
        rid, status, _, body_b64 = bridge.rest_replies[0]
        assert rid == 7 and status == 500
        assert "kaboom" in base64.b64decode(body_b64).decode()


class TestFetchBlocking:
    def test_round_trips_through_bridge(self, bridge):
        # Simulate the page answering instantly: fetchRequest records the
        # call, then we inject the net-resp before fetch_blocking pumps.
        orig = bridge.fetchRequest

        def answer(req_id, url, method, headers_json, body_b64):
            orig(req_id, url, method, headers_json, body_b64)
            gw._pending_net[req_id] = {"status": 200, "headers": {},
                                       "body": base64.b64encode(b"ok").decode()}

        bridge.fetchRequest = answer
        resp = gw.fetch_blocking("https://api.example.test/v1", "POST",
                                 {"authorization": "Bearer x"}, "aGk=")
        assert resp["status"] == 200
        assert base64.b64decode(resp["body"]) == b"ok"
        rid, url, method, headers, body_b64 = bridge.fetch_requests[0]
        assert url == "https://api.example.test/v1" and method == "POST"
        assert headers == {"authorization": "Bearer x"} and body_b64 == "aGk="

    def test_net_resp_frame_unblocks(self, bridge):
        # The pump router is the same path the worker drives after ringDrain.
        gw.handle(json.dumps({"t": "net-resp", "id": 55, "status": 204,
                              "headers": {}, "body": ""}))
        assert gw._pending_net[55]["status"] == 204


class TestSuspensionPolicyGuardrail:
    """The relay of frames while Python is blocked is the runtime's core
    safety property: a nested handle() inside pump() must not wedge."""

    def test_frame_router_bound_by_install(self, bridge):
        gw.install()
        assert br._sched.frame_router is not None
        ws = gw._FakeWS(11)
        gw._sockets[11] = ws
        # Deliver a frame through the router as pump() would.
        br._sched.frame_router({"t": "ws-send", "id": 11, "data": "nested"})
        assert ws._inq.get_nowait() == "nested"
