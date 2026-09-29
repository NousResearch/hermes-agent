"""Select a reaction menu in an encrypted room from the independent Linux client."""

from __future__ import annotations

import asyncio
import json
from urllib.parse import quote

from nio import (
    AsyncClient,
    Event,
    KeysUploadResponse,
    MatrixRoom,
    MegolmEvent,
    ReactionEvent,
    RoomMessageText,
    RoomPutStateResponse,
    RoomSendResponse,
    SyncResponse,
)

from client import open_encrypted_client


async def _raw_event(client: AsyncClient, room_id: str, event_id: str) -> dict:
    path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id, safe='')}"
    session = client.client_session
    assert session is not None
    async with session.get(
        client.homeserver + path, headers={"Authorization": f"Bearer {client.access_token}"},
    ) as response:
        assert response.status == 200, await response.text()
        return await response.json()


async def _send_encrypted_text(
    client: AsyncClient, room_id: str, body: str, content: dict, msgtype: str = "m.text",
) -> str:
    sent = await client.room_send(
        room_id, "m.room.message", {"msgtype": msgtype, "body": body, **content},
        ignore_unverified_devices=True,
    )
    assert isinstance(sent, RoomSendResponse), sent
    raw = await _raw_event(client, room_id, sent.event_id)
    assert raw["type"] == "m.room.encrypted", raw
    return sent.event_id


async def _assert_encrypted_in_thread(client: AsyncClient, room_id: str, event: RoomMessageText, root: str) -> None:
    raw = await _raw_event(client, room_id, event.event_id)
    relation = event.source["content"].get("m.relates_to", {})
    assert (event.decrypted, raw["type"], "body" in raw["content"], relation.get("rel_type"), relation.get("event_id")) == (
        True, "m.room.encrypted", False, "m.thread", root), json.dumps({"decrypted": event.source, "raw": raw})


async def select_menu(room_id: str, bot_user_id: str) -> dict:
    client = open_encrypted_client()
    replies: asyncio.Queue[RoomMessageText] = asyncio.Queue()
    bot_reactions: list[tuple[str, str]] = []
    undecrypted: list[str] = []
    received: list[dict] = []
    sync_task = None
    try:
        def receive(room: MatrixRoom, event: Event) -> None:
            if room.room_id != room_id or event.sender != bot_user_id:
                return
            if isinstance(event, RoomMessageText):
                received.append(event.source)
                replies.put_nowait(event)
            elif isinstance(event, ReactionEvent):
                bot_reactions.append((event.reacts_to, event.key))
            elif isinstance(event, MegolmEvent):
                undecrypted.append(event.event_id)

        for event_type in (RoomMessageText, ReactionEvent, MegolmEvent):
            client.add_event_callback(receive, event_type)

        synced = await client.sync(timeout=0, full_state=True)
        assert isinstance(synced, SyncResponse), synced
        if client.should_upload_keys:
            uploaded = await client.keys_upload()
            assert isinstance(uploaded, KeysUploadResponse), uploaded
        encrypted = await client.room_put_state(room_id, "m.room.encryption", {"algorithm": "m.megolm.v1.aes-sha2"})
        assert isinstance(encrypted, RoomPutStateResponse), encrypted
        synced = await client.sync(timeout=0)
        assert isinstance(synced, SyncResponse), synced
        assert client.rooms[room_id].encrypted
        sync_task = asyncio.create_task(client.sync_forever(timeout=250))

        # The gateway ignores notices, so only the threaded request starts a turn.
        root = await _send_encrypted_text(client, room_id, "Encrypted menu thread root", {}, "m.notice")
        thread = {"m.relates_to": {"rel_type": "m.thread", "event_id": root, "is_falling_back": True,
                                   "m.in_reply_to": {"event_id": root}}}
        await _send_encrypted_text(client, room_id, "Offer a route [in:encrypted-menu]", thread)

        async def reply(prefix: str) -> RoomMessageText:
            while True:
                event = await replies.get()
                if event.body.startswith(prefix):
                    return event

        menu = await asyncio.wait_for(reply("Choose the next route"), timeout=15)
        await _assert_encrypted_in_thread(client, room_id, menu, root)

        async def seeded() -> None:
            while not {(menu.event_id, "✅"), (menu.event_id, "❌")} <= set(bot_reactions):
                await asyncio.sleep(0.1)

        await asyncio.wait_for(seeded(), timeout=15)
        reaction = {"m.relates_to": {"rel_type": "m.annotation", "event_id": menu.event_id, "key": "✅"}}
        reacted = await client.room_send(room_id, "m.reaction", reaction, ignore_unverified_devices=True)
        assert isinstance(reacted, RoomSendResponse), reacted
        followup = await asyncio.wait_for(reply("Selected the first route"), timeout=15)
        await _assert_encrypted_in_thread(client, room_id, followup, root)
        return {"menu_body": menu.body, "followup_body": followup.body, "undecrypted": undecrypted}
    except BaseException:
        print("Observer diagnostics: " + json.dumps({
            "bot_reactions": bot_reactions, "undecrypted": undecrypted, "received": received,
            "sync_task_error": repr(sync_task.exception())
            if sync_task is not None and sync_task.done() and not sync_task.cancelled() else None,
        }), flush=True)
        raise
    finally:
        if sync_task is not None:
            sync_task.cancel()
            await asyncio.gather(sync_task, return_exceptions=True)
        await client.close()
