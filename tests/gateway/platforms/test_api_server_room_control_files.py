"""Real reciprocal HTTP and SQLite tests for published file access."""

import asyncio
import sqlite3
import threading
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_controls as controls, hosted_rooms
from gateway.hosted_room_control_client import RoomControlHTTPClient
from gateway.hosted_room_file_contract import FileAccessError
from gateway.hosted_room_messaging import MessagingRoomBackend, list_messaging_rooms
from gateway.platforms import api_server_room_controls
from gateway.platforms import api_server_room_control_files
from tests.gateway.test_hosted_room_file_access import file_state, publish


@pytest.fixture
def file_api(file_state, monkeypatch):
    state = file_state
    issued = controls.issue_home_control_token(
        state.db,
        room_id="room-1",
        member_id="peer",
        authority_gateway_id=state.authority,
        authority_epoch=1,
        expires_at=time.time() + 300,
    )
    monkeypatch.setattr(api_server_room_controls, "_backend", lambda: state.backend)
    app = web.Application()
    for method, path, handler in api_server_room_controls._http_routes(None):
        app.router.add_route(method, path, handler)
    state.headers = {
        "Authorization": f"HermesRoomControl {issued.control_token}",
        "X-Hermes-Room-Member": "peer",
        "X-Hermes-Room-Profile": "reviewer",
        "X-Hermes-Room-Authority": state.authority,
        "X-Hermes-Room-Epoch": "1",
    }
    state.token = issued.control_token
    state.app = app
    return state


def peer_backend(state, url):
    db = state.db.parent.parent / "peer" / "state.db"
    hosted_rooms.reserve_peer_room(
        db,
        claims={
            "room_id": "room-1",
            "member_id": "peer",
            "target_profile": "reviewer",
            "authority_gateway_id": state.authority,
            "authority_epoch": 1,
        },
        expires_at=time.time() + 300,
    )
    saved = controls.save_peer_control_link(
        db,
        room_id="room-1",
        member_id="peer",
        target_profile="reviewer",
        home_url=url,
        authority_gateway_id=state.authority,
        authority_epoch=1,
        room_name="Files",
        member_count=2,
        control_token=state.token,
        expires_at=time.time() + 300,
    )
    return MessagingRoomBackend(db_path=db), saved.link


@pytest.mark.asyncio
async def test_real_http_and_messaging_backend_return_exact_user_and_bot_files(
    file_api,
):
    state = file_api
    user = publish(state, "Résumé.txt", b"user bytes")
    bot = publish(
        state,
        "reply.md",
        b"# Full canonical file\n",
        actor={"kind": "member", "id": "peer", "profile": "reviewer"},
    )
    excluded = publish(state, "local-only.txt", recipients=["ops"])
    publish(state, "pending.txt", published=False)
    publish(state, "private.txt", published=False, viewer=False)
    async with TestServer(state.app) as server:
        backend, link = peer_backend(state, str(server.make_url("")))
        room = {**state.room, "_room_mode": "remote", "_remote_member_id": "peer"}
        page = await asyncio.to_thread(
            backend.list_files, room=room, profile="reviewer"
        )
        assert [item["attachment_id"] for item in page["items"]] == [
            bot["attachment_id"],
            user["attachment_id"],
        ]
        assert "content_base64" not in repr(page)
        read = await asyncio.to_thread(
            backend.read_file,
            room=room,
            profile="reviewer",
            event_id=user["event_id"],
            attachment_id=user["attachment_id"],
        )
        assert read.data == b"user bytes"
        assert read.attachment["name"] == "Résumé.txt"
        client = RoomControlHTTPClient(link)
        for item in (excluded, state.files[-1], state.files[-2]):
            with pytest.raises(FileAccessError) as error:
                await asyncio.to_thread(
                    client.read_file,
                    target_profile="reviewer",
                    event_id=item["event_id"],
                    attachment_id=item["attachment_id"],
                )
            assert error.value.code == "file_unavailable"
        controls.revoke_home_control_tokens(
            state.db, room_id="room-1", member_id="peer"
        )
        with pytest.raises(FileAccessError) as error:
            await asyncio.to_thread(backend.list_files, room=room)
        assert error.value.code == "file_access_denied"


@pytest.mark.asyncio
async def test_remote_files_outlive_the_admission_reservation(file_api):
    state = file_api
    item = publish(state, "overnight.md", b"durable room file")
    state.backend.service.status = lambda _room_id: {
        "working": False,
        "blocked": False,
        "counts": {},
    }
    async with TestServer(state.app) as server:
        backend, link = peer_backend(state, str(server.make_url("")))
        with sqlite3.connect(backend.db_path) as conn:
            conn.execute(
                "UPDATE hosted_room_peer_reservations SET expires_at=1"
            )

        summary = await asyncio.to_thread(RoomControlHTTPClient(link).summary)
        assert summary["room"]["room_id"] == "room-1"
        named_rooms = list_messaging_rooms(backend, profile="reviewer")
        assert [room["room_id"] for room in named_rooms] == ["room-1"]
        room = named_rooms[0]

        page = await asyncio.to_thread(
            backend.list_files, room=room, profile="reviewer"
        )
        assert [entry["attachment_id"] for entry in page["items"]] == [
            item["attachment_id"]
        ]
        stored = await asyncio.to_thread(
            backend.read_file,
            room=room,
            event_id=item["event_id"],
            attachment_id=item["attachment_id"],
        )
        assert stored.data == b"durable room file"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "token",
        "member",
        "profile",
        "authority",
        "epoch",
        "room",
        "expired",
        "revoked",
        "removed",
        "disbanded",
        "quarantine",
    ],
)
async def test_http_scope_failures_reveal_neither_names_nor_bytes(file_api, failure):
    state = file_api
    item = publish(state, "secret-name.txt", b"secret-data")
    headers = dict(state.headers)
    room_id = "room-1"
    if failure == "token":
        headers["Authorization"] = "HermesRoomControl " + "B" * 43
    elif failure in {"member", "profile", "authority", "epoch"}:
        header = {
            "member": "Member",
            "profile": "Profile",
            "authority": "Authority",
            "epoch": "Epoch",
        }[failure]
        headers[f"X-Hermes-Room-{header}"] = "2" if failure == "epoch" else "wrong"
    elif failure == "room":
        room_id = "wrong-room"
    elif failure == "expired":
        with sqlite3.connect(state.db) as conn:
            conn.execute("UPDATE hosted_room_control_tokens SET expires_at=1")
    elif failure == "revoked":
        controls.revoke_home_control_tokens(state.db, room_id="room-1")
    elif failure == "removed":
        with sqlite3.connect(state.db) as conn:
            conn.execute("UPDATE hosted_rooms SET members_json='[]'")
    elif failure == "disbanded":
        hosted_rooms.disband_room(
            state.db,
            room_id="room-1",
            expected_gateway_id=state.authority,
            expected_epoch=1,
        )
    else:
        with sqlite3.connect(state.db) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS hosted_room_quarantine (
                    room_id TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    detected_at REAL NOT NULL
                )"""
            )
            conn.execute(
                "INSERT INTO hosted_room_quarantine VALUES ('room-1', 'test', 1)"
            )
    async with TestClient(TestServer(state.app)) as client:
        for path in (
            f"/v1/room-controls/{room_id}/files",
            f"/v1/room-controls/{room_id}/files/{item['attachment_id']}?event_id={item['event_id']}",
            f"/v1/room-controls/{room_id}/files/resolve?code=12345678",
            f"/v1/room-controls/{room_id}/latest-reply",
        ):
            response = await client.get(path, headers=headers)
            assert response.status == 403
            body = await response.text()
            assert "secret-name" not in body and "secret-data" not in body
            assert state.token not in body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "event_id=wrong",
        "event_id=event-1&max_bytes=1",
        "event_id=event-1&path=/etc/passwd",
        "event_id=event-1&event_id=event-2",
    ],
)
async def test_http_exact_event_size_and_query_guards(file_api, query):
    state = file_api
    item = publish(state)
    async with TestClient(TestServer(state.app)) as client:
        response = await client.get(
            f"/v1/room-controls/room-1/files/{item['attachment_id']}?{query}",
            headers=state.headers,
        )
        assert response.status in {400, 404, 413}
        assert "published bytes" not in await response.text()


@pytest.mark.asyncio
async def test_revocation_during_real_blob_read_prevents_http_delivery(
    file_api, monkeypatch
):
    state = file_api
    item = publish(state)
    original = state.store._read_blob

    def revoked_during_read(**kwargs):
        data = original(**kwargs)
        controls.revoke_home_control_tokens(state.db, room_id="room-1")
        return data

    monkeypatch.setattr(state.store, "_read_blob", revoked_during_read)
    async with TestClient(TestServer(state.app)) as client:
        response = await client.get(
            f"/v1/room-controls/room-1/files/{item['attachment_id']}?event_id={item['event_id']}",
            headers=state.headers,
        )
        assert response.status == 403
        assert b"published bytes" not in await response.read()


@pytest.mark.asyncio
async def test_peer_link_revocation_after_fetch_prevents_backend_return(
    file_api, monkeypatch
):
    state = file_api
    item = publish(state)
    async with TestServer(state.app) as server:
        backend, _link = peer_backend(state, str(server.make_url("")))
        room = {**state.room, "_room_mode": "remote", "_remote_member_id": "peer"}
        original = RoomControlHTTPClient.read_file

        def revoke_after_fetch(client, **kwargs):
            result = original(client, **kwargs)
            controls.revoke_peer_control_links(backend.db_path, room_id="room-1")
            return result

        monkeypatch.setattr(RoomControlHTTPClient, "read_file", revoke_after_fetch)
        with pytest.raises(FileAccessError) as error:
            await asyncio.to_thread(
                backend.read_file,
                room=room,
                event_id=item["event_id"],
                attachment_id=item["attachment_id"],
            )
        assert error.value.code == "file_access_denied"


@pytest.mark.asyncio
async def test_catalogue_stays_metadata_only_and_rejects_bodies(file_api, monkeypatch):
    state = file_api
    publish(state)
    monkeypatch.setattr(
        state.store, "_read_blob", lambda **kwargs: pytest.fail("browse read bytes")
    )
    async with TestClient(TestServer(state.app)) as client:
        response = await client.get(
            "/v1/room-controls/room-1/files", headers=state.headers
        )
        assert response.status == 200
        invalid = await client.get(
            "/v1/room-controls/room-1/files",
            headers=state.headers,
            data=b"unexpected body",
        )
        assert invalid.status == 400


@pytest.mark.asyncio
async def test_server_deadline_returns_no_bytes_and_does_not_block_catalogue(
    file_api, monkeypatch
):
    state = file_api
    item = publish(state)
    entered, release = threading.Event(), threading.Event()
    original = state.store._read_blob

    def blocked(**kwargs):
        entered.set()
        assert release.wait(10)
        return original(**kwargs)

    monkeypatch.setattr(state.store, "_read_blob", blocked)
    monkeypatch.setattr(api_server_room_control_files, "FILE_TIMEOUT_SECONDS", 2.0)
    async with TestClient(TestServer(state.app)) as client:
        pending = asyncio.create_task(
            client.get(
                f"/v1/room-controls/room-1/files/{item['attachment_id']}?event_id={item['event_id']}",
                headers=state.headers,
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            listing = await client.get(
                "/v1/room-controls/room-1/files", headers=state.headers
            )
            assert listing.status == 200
            response = await pending
            assert response.status == 504
            assert b"published bytes" not in await response.read()
        finally:
            release.set()
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_exact_full_reply_uses_shared_event_text_without_model_or_blob_reads(
    file_api, monkeypatch
):
    state = file_api
    text = "# Work\n\n" + "A complete paragraph.\n" * 500
    hosted_rooms.append_event(
        state.db,
        room_id="room-1",
        event_id="full-reply",
        kind="message.member",
        actor={"kind": "member", "id": "peer", "display_name": "Reviewer"},
        payload={"member_id": "peer", "text": text},
        authority_gateway_id=state.authority,
        authority_epoch=1,
    )
    user = publish(state)
    monkeypatch.setattr(
        state.store, "_read_blob", lambda **kwargs: pytest.fail("reply read a blob")
    )
    assert (
        state.backend.read_shared_message(room=state.room, event_id="full-reply")[
            "text"
        ]
        == text
    )
    async with TestServer(state.app) as server:
        backend, link = peer_backend(state, str(server.make_url("")))
        room = {**state.room, "_room_mode": "remote", "_remote_member_id": "peer"}
        reply = await asyncio.to_thread(
            backend.read_shared_message, room=room, event_id="full-reply"
        )
        assert reply["text"] == text
        assert reply["producer"]["id"] == "peer"
        for event_id in ("missing", user["event_id"]):
            with pytest.raises(FileAccessError):
                await asyncio.to_thread(
                    RoomControlHTTPClient(link).read_shared_message,
                    target_profile="reviewer",
                    event_id=event_id,
                )
        with sqlite3.connect(state.db) as conn:
            conn.execute(
                "UPDATE hosted_room_events SET payload_json=? WHERE event_id='full-reply'",
                ('{"text":"' + "x" * (64 * 1024 + 1) + '"}',),
            )
        with pytest.raises(FileAccessError):
            await asyncio.to_thread(
                backend.read_shared_message, room=room, event_id="full-reply"
            )
