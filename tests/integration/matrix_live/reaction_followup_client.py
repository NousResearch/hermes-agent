"""Drive encrypted reaction exchanges from the independent Linux client."""

from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
from pathlib import Path
from urllib.parse import quote

from nio import (
    AsyncClient,
    Event,
    JoinedMembersResponse,
    KeysQueryResponse,
    KeysUploadResponse,
    MatrixRoom,
    MegolmEvent,
    RoomMessageText,
    RoomPutStateResponse,
    RoomSendResponse,
    ShareGroupSessionResponse,
    SyncResponse,
    ToDeviceEvent,
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
    barrier_home: str | None = None,
) -> dict:
    home = Path(barrier_home) if barrier_home is not None else None
    client = open_encrypted_client()
    olm = client.olm
    assert olm is not None
    loaded_sessions = [
        session.id for session in olm.inbound_group_store if session.room_id == room_id
    ][:32]
    replies: asyncio.Queue[RoomMessageText] = asyncio.Queue()
    events: deque[dict] = deque(maxlen=32)
    key_events: deque[dict] = deque(maxlen=16)
    crypto_errors: deque[str] = deque(maxlen=32)

    class CryptoErrors(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            crypto_errors.append(record.getMessage()[:2000])

    handler = CryptoErrors(level=logging.WARNING)
    crypto_logger = logging.getLogger("nio.crypto")
    crypto_logger.addHandler(handler)
    sync_task = None
    try:

        def receive(room: MatrixRoom, event: Event) -> None:
            if room.room_id != room_id:
                return
            summary = {
                "event_id": event.event_id,
                "sender": event.sender,
                "type": type(event).__name__,
            }
            if isinstance(event, MegolmEvent):
                summary.update(
                    session_id=event.session_id,
                    device_id=event.device_id,
                    session_available=(
                        event.sender_key is not None
                        and event.session_id is not None
                        and olm.inbound_group_store.get(
                            room_id, event.sender_key, event.session_id
                        )
                        is not None
                    ),
                )
            events.append(summary)
            if isinstance(event, RoomMessageText) and event.sender == bot_user_id:
                replies.put_nowait(event)

        for event_type in (RoomMessageText, MegolmEvent):
            client.add_event_callback(receive, event_type)

        def receive_key(event: ToDeviceEvent) -> None:
            key_events.append({
                "type": type(event).__name__,
                "sender": event.sender,
                "session_id": getattr(event, "session_id", None),
                "room_id": getattr(event, "room_id", None),
            })

        client.add_to_device_callback(receive_key, ToDeviceEvent)
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

        if home is not None and target is None:
            members = await client.joined_members(room_id)
            assert isinstance(members, JoinedMembersResponse), members
            if client.should_query_keys:
                queried = await client.keys_query()
                assert isinstance(queried, KeysQueryResponse), queried
            shared = await client.share_group_session(
                room_id, ignore_unverified_devices=True
            )
            assert isinstance(shared, ShareGroupSessionResponse), shared
            assert bot_user_id in {user for user, _device in shared.users_shared_with}
            await asyncio.to_thread(
                (home / "key-shared").write_text,
                olm.outbound_group_sessions[room_id].id,
                encoding="utf-8",
            )
            while not (home / "room-input-release").exists():
                await asyncio.sleep(0.05)

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
        if home is not None:
            raw_root = await _raw_event(client, room_id, root)
            assert raw_root["content"]["session_id"] == (home / "key-shared").read_text(
                encoding="utf-8"
            )
            await asyncio.to_thread(
                (home / "room-input-published").write_text, root, encoding="utf-8"
            )
        await _send_text(client, room_id, "Prime encrypted thread", root=root)
        await _next_reply(client, room_id, replies, "Matrix live reply", root)
        intake_id = await _send_text(
            client, room_id, "Watch the split answer", root=root
        )
        chunks = [
            await _next_reply(
                client, room_id, replies, f"{part} ({i + 1}/{len(parts)})", root
            )
            for i, part in enumerate(parts)
        ]
        return {
            "root": root,
            "intake_id": intake_id,
            "event_ids": [chunk.event_id for chunk in chunks],
        }
    except BaseException:
        task_error = None
        if sync_task is not None and sync_task.done() and not sync_task.cancelled():
            task_error = repr(sync_task.exception())
        diagnostics = json.dumps({
            "root": root,
            "target": target,
            "device_id": client.device_id,
            "loaded_sync_token": client.loaded_sync_token,
            "next_batch": client.next_batch,
            "sync_task_error": task_error,
            "loaded_inbound_sessions": loaded_sessions,
            "inbound_sessions": [
                session.id
                for session in olm.inbound_group_store
                if session.room_id == room_id
            ][:32],
            "events": list(events),
            "key_events": list(key_events),
            "crypto_errors": list(crypto_errors),
        })
        print(
            "Observer diagnostics: "
            + diagnostics.replace(client.access_token, "<redacted>"),
            flush=True,
        )
        raise
    finally:
        crypto_logger.removeHandler(handler)
        if sync_task is not None:
            sync_task.cancel()
            await asyncio.gather(sync_task, return_exceptions=True)
        await client.close()
