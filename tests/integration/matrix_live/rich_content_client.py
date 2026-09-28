"""Independent matrix-nio exchanges for native actions and sticker images."""

from __future__ import annotations

import asyncio
import base64
import io
from urllib.parse import quote

import aiohttp

from nio import (
    Event,
    JoinedMembersResponse,
    KeysQueryResponse,
    KeysUploadResponse,
    MegolmEvent,
    RoomMessageText,
    RoomPutStateResponse,
    RoomSendResponse,
    SyncResponse,
    UploadResponse,
)

from client import open_encrypted_client


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


async def open_room(room_id: str, bot_user: str, bot_device: str, *, encrypted: bool):
    client = open_encrypted_client()
    response = await client.sync(timeout=0, full_state=True)
    assert isinstance(response, SyncResponse), response
    if encrypted:
        uploaded = await client.keys_upload()
        assert isinstance(uploaded, KeysUploadResponse), uploaded
        encryption = await client.room_put_state(
            room_id,
            "m.room.encryption",
            {"algorithm": "m.megolm.v1.aes-sha2"},
        )
        assert isinstance(encryption, RoomPutStateResponse), encryption
        while not client.rooms[room_id].encrypted:
            response = await client.sync(timeout=250)
            assert isinstance(response, SyncResponse), response
        members = await client.joined_members(room_id)
        assert isinstance(members, JoinedMembersResponse), members
        queried = await client.keys_query()
        assert isinstance(queried, KeysQueryResponse), queried
        assert bot_device in client.device_store[bot_user], queried
    return client


async def send(client, room_id: str, event_type: str, content: dict) -> str:
    response = await client.room_send(
        room_id, event_type, content, ignore_unverified_devices=True
    )
    assert isinstance(response, RoomSendResponse), response
    return response.event_id


async def assert_native(
    client,
    room_id: str,
    event_id: str,
    event_type: str,
    content: dict,
    *,
    encrypted: bool,
) -> None:
    url = f"{client.homeserver}/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id, safe='')}"
    async with aiohttp.ClientSession(
        headers={"Authorization": f"Bearer {client.access_token}"}
    ) as http:
        async with http.get(url) as response:
            assert response.status == 200
            raw = await response.json()
    event = Event.parse_event(raw)
    assert isinstance(event, MegolmEvent) is encrypted
    if encrypted:
        event.room_id = room_id
        event = client.decrypt_event(event)
    assert event.source["type"] == event_type
    assert event.source["content"] == content


async def next_reply(
    client, room_id: str, bot_user: str, expected: str, *, encrypted: bool
):
    while True:
        response = await client.sync(
            timeout=250, full_state=room_id not in client.rooms
        )
        assert isinstance(response, SyncResponse), response
        joined = response.rooms.join.get(room_id)
        if joined is None:
            continue
        for event in joined.timeline.events:
            if (
                isinstance(event, RoomMessageText)
                and event.sender == bot_user
                and event.body == expected
            ):
                assert bool(event.decrypted) is encrypted
                return event


def relation(root: str, reply: str) -> dict:
    return {
        "rel_type": "m.thread",
        "event_id": root,
        "is_falling_back": False,
        "m.in_reply_to": {"event_id": reply},
    }


async def accepted_exchange(
    room_id: str, bot_user: str, bot_device: str, *, encrypted: bool
) -> dict:
    client = await open_room(room_id, bot_user, bot_device, encrypted=encrypted)
    try:
        root = await send(
            client,
            room_id,
            "m.room.message",
            {"msgtype": "m.notice", "body": "Rich content thread root"},
        )
        emote_content = {
            "msgtype": "m.emote",
            "body": "/new waves",
            "format": "org.matrix.custom.html",
            "formatted_body": "<i>/new waves</i>",
            "m.relates_to": relation(root, root),
        }
        emote = await send(client, room_id, "m.room.message", emote_content)
        await assert_native(
            client, room_id, emote, "m.room.message", emote_content, encrypted=encrypted
        )
        first = await next_reply(
            client, room_id, bot_user, "Matrix live reply", encrypted=encrypted
        )
        assert first.source["content"]["m.relates_to"] == relation(root, emote)
        uploaded, keys = await client.upload(
            io.BytesIO(PNG),
            content_type="image/png",
            filename="fox.png",
            filesize=len(PNG),
        )
        assert isinstance(uploaded, UploadResponse), uploaded
        assert keys is None
        sticker_content = {
            "body": "Friendly fox.png",
            "url": uploaded.content_uri,
            "info": {"mimetype": "image/png", "size": len(PNG), "w": 1, "h": 1},
            "m.relates_to": relation(root, emote),
        }
        sticker = await send(client, room_id, "m.sticker", sticker_content)
        await assert_native(
            client, room_id, sticker, "m.sticker", sticker_content, encrypted=encrypted
        )
        second = await next_reply(client, room_id, bot_user, "ok", encrypted=encrypted)
        assert second.source["content"]["m.relates_to"] == relation(root, sticker)
        return {"root": root, "sticker": sticker, "url": uploaded.content_uri}
    finally:
        await client.close()


async def paused_sticker(room_id: str, root: str, url: str, *, encrypted: bool) -> str:
    client = open_encrypted_client()
    try:
        response = await client.sync(timeout=0, full_state=True)
        assert isinstance(response, SyncResponse), response
        content = {
            "body": "Withdrawn fox @matrix-live:pause",
            "url": url,
            "info": {"mimetype": "image/png", "size": len(PNG), "w": 1, "h": 1},
            "m.relates_to": relation(root, root),
        }
        event_id = await send(client, room_id, "m.sticker", content)
        await assert_native(
            client, room_id, event_id, "m.sticker", content, encrypted=encrypted
        )
        return event_id
    finally:
        await client.close()


async def withdrawn_reply(room_id: str, bot_user: str, *, encrypted: bool) -> None:
    client = open_encrypted_client()
    try:
        await next_reply(client, room_id, bot_user, "ok", encrypted=encrypted)
    finally:
        await client.close()
