"""Credential-safe reciprocal Group Chat control client."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from gateway import hosted_room_controls, hosted_rooms
from gateway.hosted_room_control_client import (
    RoomControlHTTPClient,
    RoomControlClientError,
    revoke_stored_peer_control,
)
from gateway.hosted_room_controls import StoredPeerRoomControl
from gateway.hosted_room_file_contract import FileAccessError
from gateway.hosted_room_messaging import MessagingRoomBackend


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
        target_profile="reviewer",
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
    assert client.revoke() == 1

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
        target_profile="reviewer",
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
        lambda self: revoked.append(self.link.room_id) or 1,
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

    replacement = hosted_room_controls.save_peer_control_link(
        db,
        room_id="room-1",
        member_id="member-peer",
        target_profile="reviewer",
        room_name="Planning",
        member_count=2,
        home_url="https://home.example.test",
        authority_gateway_id="install:home",
        authority_epoch=1,
        control_token="A" * 43,
        expires_at=10_000_000_000,
        now=20,
    ).link

    def replace_before_ack(_client):
        hosted_room_controls.save_peer_control_link(
            db,
            room_id=replacement.room_id,
            member_id=replacement.member_id,
            target_profile="reviewer",
            room_name="Planning",
            member_count=2,
            home_url=replacement.home_url,
            authority_gateway_id=replacement.authority_gateway_id,
            authority_epoch=replacement.authority_epoch,
            control_token="B" * 43,
            expires_at=10_000_000_000,
            allow_rotation=True,
            now=30,
        )
        return 1

    monkeypatch.setattr(RoomControlHTTPClient, "revoke", replace_before_ack)
    assert (
        revoke_stored_peer_control(db, room_id="room-1", member_id="member-peer")
        == 0
    )
    current = hosted_room_controls.load_peer_control_links(db, now=40).links
    assert len(current) == 1
    assert current[0].control_token == "B" * 43


def test_partial_revoke_survives_reservation_gc_and_retries_the_bearer(
    tmp_path, monkeypatch
):
    db = tmp_path / "state.db"
    profile_db = tmp_path / "profile.db"
    now = time.time()
    claims = {
        "room_id": "room-1",
        "home_install_id": "install:home",
        "member_id": "member-peer",
        "target_profile": "reviewer",
        "target_install_id": "install:target",
        "authority_gateway_id": "install:home",
        "authority_epoch": 1,
    }
    reservation_expiry = now + 60
    hosted_rooms.reserve_peer_room(
        db, claims=claims, expires_at=reservation_expiry, now=now
    )
    saved = hosted_room_controls.save_peer_control_link(
        db,
        room_id=claims["room_id"],
        member_id=claims["member_id"],
        target_profile=claims["target_profile"],
        authority_gateway_id=claims["authority_gateway_id"],
        authority_epoch=claims["authority_epoch"],
        room_name="Planning",
        member_count=2,
        home_url="https://home.example.test",
        control_token="A" * 43,
        expires_at=10_000_000_000,
        now=now,
    ).link

    # Exercise the shared-store commit followed by a real second-store failure
    # without importing a coordinator owned by the separate route-repair PR.
    profile_db.mkdir()
    with pytest.raises(sqlite3.OperationalError):
        for path in (db, profile_db):
            hosted_rooms.revoke_room_grant_scope(
                path, claims=claims, expires_at=reservation_expiry
            )

    retained = hosted_room_controls.load_peer_control_links(
        db, include_inactive=True, now=now + 1
    ).links
    assert len(retained) == 1
    assert retained[0].status == "revoked"
    assert retained[0].control_token == saved.control_token
    assert hosted_room_controls.load_peer_control_links(db, now=now + 1).links == ()

    responses = [
        RoomControlClientError("home unavailable", retryable=True),
        {"revoked": 2},
        {"revoked": 0, "server_revision": 2},
    ]
    attempts = []

    def request(client, *, method, body=None):
        assert method == "DELETE" and body is None
        attempts.append(client.link.control_token)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    hosted_rooms.reserve_peer_room(
        db,
        claims={
            "room_id": "room-other",
            "member_id": "member-other",
            "target_profile": "other",
            "authority_gateway_id": "install:other",
            "authority_epoch": 1,
        },
        expires_at=reservation_expiry + 120,
        now=reservation_expiry + 1,
    )
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM hosted_room_peer_reservations WHERE room_id='room-1'"
        ).fetchone()[0] == 0

    monkeypatch.setattr(
        RoomControlHTTPClient,
        "list_files",
        lambda *_args, **_kwargs: pytest.fail("revoked control reused for Files"),
    )
    backend = MessagingRoomBackend(db_path=db)
    room = {
        **claims,
        "_room_mode": "remote",
        "_remote_member_id": "member-peer",
    }
    with pytest.raises(FileAccessError) as error:
        backend.list_files(room=room)
    assert error.value.code == "file_access_denied"

    monkeypatch.setattr(RoomControlHTTPClient, "_request", request)
    with pytest.raises(RoomControlClientError, match="unavailable"):
        revoke_stored_peer_control(db, room_id="room-1", member_id="member-peer")
    assert len(
        hosted_room_controls.load_peer_control_links(
            db, include_inactive=True, now=now + 1
        ).links
    ) == 1

    with pytest.raises(RoomControlClientError, match="acknowledgement"):
        revoke_stored_peer_control(db, room_id="room-1", member_id="member-peer")
    assert len(
        hosted_room_controls.load_peer_control_links(
            db, include_inactive=True, now=now + 1
        ).links
    ) == 1

    assert (
        revoke_stored_peer_control(db, room_id="room-1", member_id="member-peer")
        == 1
    )
    assert (
        hosted_room_controls.load_peer_control_links(
            db, include_inactive=True, now=now + 1
        ).links
        == ()
    )
    assert attempts == [saved.control_token] * 3
