"""Independent Matrix client operations for the administration contract."""

from __future__ import annotations

from nio import (
    JoinedMembersResponse, KeysQueryResponse, KeysUploadResponse, RoomGetStateEventResponse,
    RoomMessageText, RoomPutStateResponse, RoomSendResponse, SyncResponse,
)


async def send(client, room: str, body: str, *, notice: bool = False) -> str:
    if room not in client.rooms:
        response = await client.sync(timeout=0, full_state=True)
        assert isinstance(response, SyncResponse), response
    result = await client.room_send(
        room, "m.room.message", {"msgtype": "m.notice" if notice else "m.text", "body": body}, ignore_unverified_devices=True,
    )
    assert isinstance(result, RoomSendResponse), result
    return result.event_id


async def reply(client, room: str, bot: str, answer: str, *, encrypted: bool) -> str:
    while True:
        response = await client.sync(timeout=250)
        assert isinstance(response, SyncResponse), response
        joined = response.rooms.join.get(room)
        if joined is None:
            continue
        for event in joined.timeline.events:
            if isinstance(event, RoomMessageText) and event.sender == bot and event.body == answer:
                assert event.decrypted is encrypted, event.source
                return event.event_id


async def prepare(client, room: str, bot: str, bot_device: str, *, encrypted: bool) -> None:
    response = await client.sync(timeout=0, full_state=True)
    assert isinstance(response, SyncResponse), response
    if encrypted:
        if client.should_upload_keys:
            uploaded = await client.keys_upload()
            assert isinstance(uploaded, KeysUploadResponse), uploaded
        if not client.rooms[room].encrypted:
            state = await client.room_put_state(room, "m.room.encryption", {"algorithm": "m.megolm.v1.aes-sha2"})
            assert isinstance(state, RoomPutStateResponse), state
        while not client.rooms[room].encrypted:
            await client.sync(timeout=250)
        members = await client.joined_members(room)
        assert isinstance(members, JoinedMembersResponse), members
        if client.should_query_keys:
            queried = await client.keys_query()
            assert isinstance(queried, KeysQueryResponse), queried
        assert bot_device in client.device_store[bot], list(client.device_store[bot])


async def request(client, room: str, bot: str, body: str, answer: str, *, encrypted: bool) -> str:
    response = await client.sync(timeout=0, full_state=True)
    assert isinstance(response, SyncResponse), response
    await send(client, room, body)
    return await reply(client, room, bot, answer, encrypted=encrypted)


async def power(client, room: str, actor: str, bot: str, *, actor_level: int = 100, bot_level: int = 100) -> None:
    result = await client.room_put_state(room, "m.room.power_levels", {
        "users": {actor: actor_level, bot: bot_level}, "invite": 50,
    })
    assert isinstance(result, RoomPutStateResponse), result


async def state(client, room: str, kind: str, state_key: str = "") -> dict:
    result = await client.room_get_state_event(room, kind, state_key)
    assert isinstance(result, RoomGetStateEventResponse), result
    return result.content
