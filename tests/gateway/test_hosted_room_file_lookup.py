"""Code collisions and exact latest replies use one bounded authority lookup."""

import asyncio
import hashlib
import json
import sqlite3

import pytest
from aiohttp.test_utils import TestServer

from gateway import hosted_rooms
from gateway import hosted_room_file_lookup as lookup
from gateway.hosted_room_file_contract import FileAccessError
from tests.gateway.test_hosted_room_file_access import file_state, publish
from tests.gateway.test_hosted_room_attachment_catalog import _seed_events
from tests.gateway.platforms.test_api_server_room_control_files import (
    file_api,
    peer_backend,
)


@pytest.mark.parametrize(
    "actor",
    [
        {"kind": "user", "id": "desktop"},
        {"kind": "member", "id": "peer"},
        {"kind": "member", "id": "peer", "display_name": "Explicit label"},
    ],
)
def test_resolved_metadata_is_exactly_the_canonical_catalogue_item(file_state, actor):
    item = publish(file_state, actor=actor)
    catalogued = file_state.backend.list_files(room=file_state.room)["items"][0]
    resolved = lookup.resolve_file(
        file_state.backend,
        room=file_state.room,
        code=lookup.selection_digest(file_state.room, item),
    )
    assert resolved == catalogued


@pytest.mark.parametrize(
    "damage",
    [
        "encoded_manifest",
        "future_epoch",
        "invalid_blob_id",
        "missing_blob",
        "actor",
        "wrong_event",
        "staged",
        "expired",
        "private",
    ],
)
def test_resolver_never_bypasses_canonical_eligibility(file_state, damage):
    state = file_state
    item = publish(state)
    with sqlite3.connect(state.db) as conn:
        if damage == "encoded_manifest":
            payload = json.loads(
                conn.execute(
                    "SELECT payload_json FROM hosted_room_events WHERE event_id=?",
                    (item["event_id"],),
                ).fetchone()[0]
            )
            payload["attachments"] = json.dumps(payload["attachments"])
            conn.execute(
                "UPDATE hosted_room_events SET payload_json=? WHERE event_id=?",
                (json.dumps(payload), item["event_id"]),
            )
        elif damage == "future_epoch":
            conn.execute("UPDATE hosted_room_events SET authority_epoch=2")
        elif damage == "invalid_blob_id":
            conn.execute("UPDATE hosted_room_attachments SET blob_id='bad'")
        elif damage == "missing_blob":
            conn.execute("DELETE FROM hosted_room_attachment_blobs")
        elif damage == "actor":
            conn.execute("UPDATE hosted_room_events SET actor_json='{}'")
        elif damage == "wrong_event":
            conn.execute("UPDATE hosted_room_attachments SET event_id='wrong'")
        elif damage == "staged":
            conn.execute("UPDATE hosted_room_attachments SET state='uploaded'")
        elif damage == "expired":
            conn.execute("UPDATE hosted_room_attachments SET expires_at=1")
        elif damage == "private":
            conn.execute("UPDATE hosted_room_attachments SET viewer_access=0")
    assert state.backend.list_files(room=state.room)["items"] == []
    with pytest.raises(FileAccessError) as error:
        lookup.resolve_file(
            state.backend,
            room=state.room,
            code=lookup.selection_digest(state.room, item),
        )
    assert error.value.code == "file_unavailable"


def test_codes_bind_the_full_immutable_scope(file_state):
    state = file_state
    item = publish(state)
    code = lookup.selection_digest(state.room, item)
    assert len(code) == 64
    assert (
        lookup.resolve_file(state.backend, room=state.room, code=code[:8])[
            "attachment_id"
        ]
        == item["attachment_id"]
    )
    for change in (
        {"room_id": "other"},
        {"authority_gateway_id": "other"},
        {"authority_epoch": 2},
    ):
        assert lookup.selection_digest({**state.room, **change}, item) != code
    assert lookup.selection_digest(state.room, {**item, "event_id": "other"}) != code


def test_collision_refuses_first_match_and_longer_code_disambiguates(
    file_state, monkeypatch
):
    state = file_state
    first, second = publish(state), publish(state, "other.txt")

    def digest(room, item):
        return (
            "deadbeef" + hashlib.sha256(item["attachment_id"].encode()).hexdigest()[8:]
        )

    monkeypatch.setattr(lookup, "selection_digest", digest)
    with pytest.raises(FileAccessError) as error:
        lookup.resolve_file(state.backend, room=state.room, code="deadbeef")
    assert error.value.code == "file_code_ambiguous"
    assert {item["attachment_id"] for item in error.value.matches} == {
        first["attachment_id"],
        second["attachment_id"],
    }
    selected = lookup.resolve_file(
        state.backend, room=state.room, code=digest(state.room, second)[:16]
    )
    assert selected["attachment_id"] == second["attachment_id"]
    assert selected["attachment_id"] != first["attachment_id"]


def test_collision_does_not_reveal_another_members_file(file_state, monkeypatch):
    state = file_state
    allowed = publish(state)
    publish(state, "private-recipient.txt", recipients=["ops"])
    publish(state, "never-published.txt", published=False)
    monkeypatch.setattr(lookup, "selection_digest", lambda room, item: "a" * 64)
    selected = lookup.resolve_local_file(
        state.backend, room=state.room, code="aaaaaaaa", member_id="peer"
    )
    assert selected["attachment_id"] == allowed["attachment_id"]


def test_fifty_thousand_records_resolve_without_catalogue_pages(
    file_state, monkeypatch
):
    state = file_state
    _seed_events(
        state.db,
        total_events=50_000,
        records={index: [f"file-{index}.bin"] for index in range(1, 50_001)},
    )
    item = {"attachment_id": f"att_{50_000 * 16:032x}", "event_id": "event-50000"}
    code = lookup.selection_digest(state.room, item)
    monkeypatch.setattr(
        state.store,
        "list_published",
        lambda **kwargs: pytest.fail("resolver paged catalogue"),
    )
    monkeypatch.setattr(
        state.store, "_read_blob", lambda **kwargs: pytest.fail("resolver read bytes")
    )
    selected = lookup.resolve_file(state.backend, room=state.room, code=code)
    assert selected["event_id"] == "event-50000"
    monkeypatch.setattr(lookup, "MAX_LOOKUP_CANDIDATES", 10)
    with pytest.raises(FileAccessError) as error:
        lookup.resolve_file(state.backend, room=state.room, code=code)
    assert error.value.code == "file_lookup_limit"


def test_latest_reply_is_not_limited_to_eighty_recent_events(file_state):
    state = file_state

    def append(event_id, kind, actor, text):
        return hosted_rooms.append_event(
            state.db,
            room_id="room-1",
            event_id=event_id,
            kind=kind,
            actor=actor,
            payload={"text": text},
            authority_gateway_id=state.authority,
            authority_epoch=1,
        )

    append(
        "bot-original",
        "message.member",
        {"kind": "member", "id": "peer"},
        "# Complete reply\n" + "paragraph\n" * 100,
    )
    for index in range(100):
        append(
            f"ordinary-{index}",
            "message.user",
            {"kind": "user", "id": "desktop"},
            "ordinary",
        )
    assert (
        lookup.latest_reply(state.backend, room=state.room)["event_id"]
        == "bot-original"
    )
    append(
        "bot-latest", "message.member", {"kind": "member", "id": "ops"}, "latest work"
    )
    assert lookup.latest_reply(state.backend, room=state.room)["text"] == "latest work"


@pytest.mark.asyncio
async def test_remote_code_and_latest_reply_each_use_one_real_http_request(file_api):
    state = file_api
    item = publish(state, actor={"kind": "member", "id": "peer"})
    requests = []

    async def capture(request, response):
        requests.append(request.path)

    state.app.on_response_prepare.append(capture)
    async with TestServer(state.app) as server:
        backend, _link = peer_backend(state, str(server.make_url("")))
        room = {**state.room, "_room_mode": "remote", "_remote_member_id": "peer"}
        selected = await asyncio.to_thread(
            lookup.resolve_file,
            backend,
            room=room,
            code=lookup.selection_digest(room, item)[:8],
        )
        reply = await asyncio.to_thread(lookup.latest_reply, backend, room=room)
    assert selected["attachment_id"] == item["attachment_id"]
    assert reply["event_id"] == item["event_id"]
    assert requests == [
        "/v1/room-controls/room-1/files/resolve",
        "/v1/room-controls/room-1/latest-reply",
    ]


@pytest.mark.asyncio
async def test_remote_recipient_prefilter_returns_old_eligible_file_without_empty_pages(
    file_api,
):
    state = file_api
    old = publish(state, "old-peer.txt", recipients=["peer"])
    for index in range(300):
        publish(state, f"new-ops-{index}.txt", recipients=["ops"])
    async with TestServer(state.app) as server:
        backend, _link = peer_backend(state, str(server.make_url("")))
        room = {**state.room, "_room_mode": "remote", "_remote_member_id": "peer"}
        page = await asyncio.to_thread(backend.list_files, room=room)
    assert [item["attachment_id"] for item in page["items"]] == [old["attachment_id"]]
    assert page["has_more"] is False
    assert page["latest_seq"] == 1


@pytest.mark.asyncio
async def test_remote_collision_returns_only_authorized_exact_metadata(
    file_api, monkeypatch
):
    state = file_api
    first, second = publish(state), publish(state, "second.txt")
    publish(state, "ops-only.txt", recipients=["ops"])
    monkeypatch.setattr(lookup, "selection_digest", lambda room, item: "d" * 64)
    async with TestServer(state.app) as server:
        backend, _link = peer_backend(state, str(server.make_url("")))
        room = {**state.room, "_room_mode": "remote", "_remote_member_id": "peer"}
        with pytest.raises(FileAccessError) as error:
            await asyncio.to_thread(
                lookup.resolve_file, backend, room=room, code="dddddddd"
            )
    assert error.value.code == "file_code_ambiguous"
    assert {item["attachment_id"] for item in error.value.matches} == {
        first["attachment_id"],
        second["attachment_id"],
    }
