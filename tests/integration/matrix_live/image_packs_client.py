"""Independent Matrix client proof for image-pack discovery and sticker transport."""

from __future__ import annotations

import io
from urllib.parse import quote

import aiohttp
from nio import RoomPutStateResponse, SyncResponse, UploadResponse

from rich_content_client import (
    PNG,
    assert_native,
    next_reply,
    open_room,
    relation,
    send,
)


async def account_data(
    client, user_id: str, access_token: str, kind: str, content: dict
):
    url = (
        f"{client.homeserver}/_matrix/client/v3/user/{quote(user_id, safe='')}/account_data/"
        f"{quote(kind, safe='')}"
    )
    async with aiohttp.ClientSession(
        headers={"Authorization": f"Bearer {access_token}"}
    ) as http:
        async with http.put(url, json=content) as response:
            assert response.status == 200, await response.text()


def pack(url: str, body: str) -> dict:
    return {
        "pack": {"display_name": "/new duplicate label", "usage": ["sticker"]},
        "images": {
            "fox": {
                "url": url,
                "body": body,
                "info": {"mimetype": "image/png", "size": len(PNG), "w": 1, "h": 1},
            }
        },
    }


async def prepare(
    room: str, bot: str, device: str, bot_token: str, *, encrypted: bool
) -> dict:
    client = await open_room(room, bot, device, encrypted=encrypted)
    try:
        await send(
            client,
            room,
            "m.room.message",
            {"msgtype": "m.text", "body": "Pack handshake"},
        )
        await next_reply(client, room, bot, "Matrix live reply", encrypted=encrypted)
        uploaded, keys = await client.upload(
            io.BytesIO(PNG),
            content_type="image/png",
            filename="fox.png",
            filesize=len(PNG),
        )
        assert isinstance(uploaded, UploadResponse), uploaded
        assert keys is None
        url = uploaded.content_uri
        for event_type, key, body in [
            ("m.room.image_pack", "", "Room fox.png"),
            ("m.room.image_pack", "named", "Referenced fox.png"),
            ("im.ponies.room_emotes", "legacy", "Legacy fox.png"),
        ]:
            state = await client.room_put_state(
                room, event_type, pack(url, body), state_key=key
            )
            assert isinstance(state, RoomPutStateResponse), state
        await account_data(
            client,
            bot,
            bot_token,
            "m.image_pack.rooms",
            {"rooms": {room: {"named": {}}}},
        )
        await account_data(
            client,
            bot,
            bot_token,
            "im.ponies.emote_rooms",
            {"rooms": {room: {"legacy": {}}}},
        )
        await account_data(
            client,
            bot,
            bot_token,
            "im.ponies.user_emotes",
            pack(url, "Bot private fox.png"),
        )
        await account_data(
            client,
            client.user_id,
            client.access_token,
            "im.ponies.user_emotes",
            pack(url, "Requester private sentinel.png"),
        )
        root = await send(
            client,
            room,
            "m.room.message",
            {"msgtype": "m.notice", "body": "Image-pack thread root"},
        )
        return {"root": root, "url": url}
    finally:
        await client.close()


async def enable_encryption(room: str, bot: str, device: str) -> None:
    client = await open_room(room, bot, device, encrypted=True)
    await client.close()


async def ask(
    room: str,
    bot: str,
    root: str,
    body: str,
    expected: str,
    *,
    encrypted: bool,
    sticker_body: str | None = None,
    url: str | None = None,
) -> dict:
    from client import open_encrypted_client

    client = open_encrypted_client()
    try:
        synced = await client.sync(timeout=0, full_state=True)
        assert isinstance(synced, SyncResponse), synced
        question = await send(
            client,
            room,
            "m.room.message",
            {
                "msgtype": "m.text",
                "body": body,
                "m.relates_to": relation(root, root),
            },
        )
        sticker = None
        observed = []
        while True:
            response = await client.sync(timeout=250)
            assert isinstance(response, SyncResponse), response
            joined = response.rooms.join.get(room)
            if joined is None:
                continue
            done = False
            for event in joined.timeline.events:
                if event.sender != bot:
                    continue
                raw = event.source
                observed.append({
                    "event_id": event.event_id,
                    "type": raw.get("type"),
                    "decrypted": bool(event.decrypted),
                    "body": raw.get("content", {}).get("body"),
                    "relation": raw.get("content", {}).get("m.relates_to"),
                })
                if raw.get("type") == "m.sticker":
                    assert bool(event.decrypted) is encrypted
                    content = raw["content"]
                    assert url is not None and sticker_body is not None
                    expected_content = {
                        **pack(url, sticker_body)["images"]["fox"],
                        "m.relates_to": relation(root, question),
                    }
                    assert content == expected_content
                    await assert_native(
                        client,
                        room,
                        event.event_id,
                        "m.sticker",
                        expected_content,
                        encrypted=encrypted,
                    )
                    sticker = event.event_id
                if (
                    raw.get("type") == "m.room.message"
                    and raw["content"].get("body") == expected
                ):
                    assert bool(event.decrypted) is encrypted
                    done = True
            if done:
                assert (sticker is not None) is (sticker_body is not None), {
                    "expected_sticker": sticker_body,
                    "observed_bot_events": observed[-20:],
                }
                return {"question": question, "sticker": sticker}
    finally:
        await client.close()


async def reply_to_sticker(
    room: str, bot: str, root: str, sticker: str, *, encrypted: bool
):
    from client import open_encrypted_client

    client = open_encrypted_client()
    try:
        response = await client.sync(timeout=0, full_state=True)
        assert isinstance(response, SyncResponse), response
        await send(
            client,
            room,
            "m.room.message",
            {
                "msgtype": "m.text",
                "body": "Explain the selected sticker",
                "m.relates_to": relation(root, sticker),
            },
        )
        await next_reply(
            client, room, bot, "Sticker context checked", encrypted=encrypted
        )
    finally:
        await client.close()
