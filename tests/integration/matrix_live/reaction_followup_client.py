"""Drive encrypted reaction exchanges from the independent Linux client."""

from __future__ import annotations

import asyncio
from urllib.parse import quote

from nio import (
    AsyncClient,
    Event,
    KeysUploadResponse,
    MatrixRoom,
    RoomMessageText,
    RoomPutStateResponse,
    RoomSendResponse,
    SyncResponse,
)

from client import open_encrypted_client


async def _raw_event(client: AsyncClient, room_id: str, event_id: str) -> dict:
    path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/" + quote(
        event_id, safe=""
    )
    session = client.client_session
    assert session is not None
    async with session.get(
        client.homeserver + path,
        headers={"Authorization": f"Bearer {client.access_token}"},
    ) as response:
        assert response.status == 200, await response.text()
        return await response.json()


async def _send_text(
    client: AsyncClient,
    room_id: str,
    body: str,
    *,
    root: str | None = None,
    msgtype: str = "m.text",
) -> str:
    content = {"msgtype": msgtype, "body": body}
    if root is not None:
        content["m.relates_to"] = {
            "rel_type": "m.thread",
            "event_id": root,
            "is_falling_back": True,
            "m.in_reply_to": {"event_id": root},
        }
    sent = await client.room_send(
        room_id,
        "m.room.message",
        content,
        ignore_unverified_devices=True,
    )
    assert isinstance(sent, RoomSendResponse), sent
    raw = await _raw_event(client, room_id, sent.event_id)
    assert raw["type"] == "m.room.encrypted", raw
    return sent.event_id


async def _next_reply(
    client: AsyncClient,
    room_id: str,
    replies: asyncio.Queue[RoomMessageText],
    body: str,
    root: str,
) -> RoomMessageText:
    async def matching_reply() -> RoomMessageText:
        while True:
            reply = await replies.get()
            if reply.body == body:
                return reply

    reply = await asyncio.wait_for(matching_reply(), timeout=15)
    assert (reply.body, reply.decrypted) == (body, True)
    relation = reply.source["content"]["m.relates_to"]
    assert (relation["rel_type"], relation["event_id"]) == ("m.thread", root)
    raw = await _raw_event(client, room_id, reply.event_id)
    assert raw["type"] == "m.room.encrypted", raw
    assert raw["content"]["algorithm"] == "m.megolm.v1.aes-sha2"
    assert raw["content"]["ciphertext"]
    assert "body" not in raw["content"]
    return reply


async def _exchange(
    room_id: str,
    bot_user_id: str,
    parts: list[str],
    *,
    root: str | None = None,
    target: str | None = None,
) -> dict:
    client = open_encrypted_client()
    replies: asyncio.Queue[RoomMessageText] = asyncio.Queue()
    sync_task = None
    try:
        synced = await client.sync(timeout=0, full_state=True)
        assert isinstance(synced, SyncResponse), synced
        if client.should_upload_keys:
            uploaded = await client.keys_upload()
            assert isinstance(uploaded, KeysUploadResponse), uploaded
        if target is None:
            encrypted = await client.room_put_state(
                room_id,
                "m.room.encryption",
                {"algorithm": "m.megolm.v1.aes-sha2"},
            )
            assert isinstance(encrypted, RoomPutStateResponse), encrypted
            synced = await client.sync(timeout=0)
            assert isinstance(synced, SyncResponse), synced
        assert client.rooms[room_id].encrypted

        async def receive(room: MatrixRoom, event: Event) -> None:
            if (
                room.room_id == room_id
                and isinstance(event, RoomMessageText)
                and event.sender == bot_user_id
            ):
                replies.put_nowait(event)

        client.add_event_callback(receive, RoomMessageText)
        sync_task = asyncio.create_task(client.sync_forever(timeout=250))
        if target is not None:
            assert root is not None
            content = {
                "m.relates_to": {
                    "rel_type": "m.annotation",
                    "event_id": target,
                    "key": "👍",
                }
            }
            reacted = await client.room_send(
                room_id,
                "m.reaction",
                content,
                ignore_unverified_devices=True,
            )
            assert isinstance(reacted, RoomSendResponse), reacted
            raw = await _raw_event(client, room_id, reacted.event_id)
            assert (raw["type"], raw["sender"], raw["content"]) == (
                "m.reaction",
                client.user_id,
                content,
            )
            await _next_reply(client, room_id, replies, "Reaction follow-up", root)
            return {"reaction_id": reacted.event_id}

        root = await _send_text(
            client, room_id, "Encrypted thread root", msgtype="m.notice"
        )
        await _send_text(client, room_id, "Prime encrypted thread", root=root)
        await _next_reply(client, room_id, replies, "Matrix live reply", root)
        await _send_text(client, room_id, "Watch the split answer", root=root)
        chunks = [
            await _next_reply(
                client, room_id, replies, f"{part} ({i + 1}/{len(parts)})", root
            )
            for i, part in enumerate(parts)
        ]
        return {"root": root, "event_ids": [chunk.event_id for chunk in chunks]}
    finally:
        if sync_task is not None:
            sync_task.cancel()
            await asyncio.gather(sync_task, return_exceptions=True)
        await client.close()
