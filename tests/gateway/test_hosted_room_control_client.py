"""Credential-safe reciprocal Group Chat control client."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from gateway import hosted_room_controls
from gateway.hosted_room_control_client import (
    RoomControlHTTPClient,
    revoke_stored_peer_control,
)
from gateway.hosted_room_controls import StoredPeerRoomControl


class ControlHandler(BaseHTTPRequestHandler):
    requests = []

    def _reply(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        type(self).requests.append(("GET", self.path, dict(self.headers), None))
        self._reply({"room": {"room_id": "room-1"}, "events": []})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(("POST", self.path, dict(self.headers), body))
        self._reply({"action": body["action"], "summary": {"events": []}})

    def do_DELETE(self):
        type(self).requests.append(("DELETE", self.path, dict(self.headers), None))
        self._reply({"revoked": 1})

    def log_message(self, *_args):
        pass


@pytest.fixture
def control_server():
    ControlHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), ControlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _link(url: str) -> StoredPeerRoomControl:
    return StoredPeerRoomControl(
        room_id="room-1",
        member_id="member-peer",
        home_url=url,
        transport_security="loopback",
        authority_gateway_id="install:home",
        authority_epoch=1,
        room_name="Planning",
        member_count=2,
        control_token="A" * 43,
        status="active",
        created_at=1,
        updated_at=1,
        expires_at=10_000_000_000,
    )


def test_summary_and_mutation_keep_the_token_in_headers(control_server):
    client = RoomControlHTTPClient(_link(control_server))

    assert client.summary()["room"]["room_id"] == "room-1"
    assert (
        client.mutate(
            action="send",
            command_id="command-1",
            text="hello",
            actor_display_name="Signal",
        )["action"]
        == "send"
    )
    client.revoke()

    assert [request[0] for request in ControlHandler.requests] == [
        "GET",
        "POST",
        "DELETE",
    ]
    for _method, path, headers, body in ControlHandler.requests:
        assert path == "/v1/room-controls/room-1"
        assert headers["Authorization"] == "HermesRoomControl " + "A" * 43
        assert headers["X-Hermes-Room-Member"] == "member-peer"
        assert "A" * 43 not in repr(body)


def test_control_client_refuses_cross_origin_redirects():
    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9/stolen")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = RoomControlHTTPClient(_link(f"http://127.0.0.1:{server.server_port}"))
        with pytest.raises(Exception):
            client.summary()
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_stored_peer_revoke_contacts_home_before_erasing_bearer(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    saved = hosted_room_controls.save_peer_control_link(
        db,
        room_id="room-1",
        member_id="member-peer",
        room_name="Planning",
        member_count=2,
        home_url="https://home.example.test",
        authority_gateway_id="install:home",
        authority_epoch=1,
        control_token="A" * 43,
        expires_at=10_000_000_000,
        now=20,
    )
    revoked = []
    monkeypatch.setattr(
        RoomControlHTTPClient,
        "revoke",
        lambda self: revoked.append(self.link.room_id),
    )

    assert (
        revoke_stored_peer_control(db, room_id="room-1", member_id="member-peer") == 1
    )
    assert revoked == [saved.link.room_id]
    assert (
        hosted_room_controls.load_peer_control_links(
            db, include_inactive=True, now=30
        ).links
        == ()
    )
