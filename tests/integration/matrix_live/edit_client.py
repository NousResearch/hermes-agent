"""Separate-client exchanges for edit follow-ups, with a persistent encryption store."""

from __future__ import annotations

from nio import (
    JoinedMembersResponse, KeysQueryResponse, KeysUploadResponse, RoomMessageText,
    RoomPutStateResponse, RoomSendResponse, SyncResponse,
)


async def send(client, room_id: str, content: dict) -> str:
    result = await client.room_send(
        room_id, "m.room.message", content, ignore_unverified_devices=True,
    )
    assert isinstance(result, RoomSendResponse), result
    return result.event_id


async def final(client, room_id: str, bot: str, expected: str, *, encrypted: bool) -> dict:
    while True:
        response = await client.sync(timeout=250)
        assert isinstance(response, SyncResponse), response
        joined = response.rooms.join.get(room_id)
        if joined is None:
            continue
        for event in joined.timeline.events:
            if isinstance(event, RoomMessageText) and event.sender == bot and event.body == expected:
                assert event.decrypted is encrypted, event.source
                content = event.source["content"]
                return content.get("m.relates_to", {})


async def begin(client, room_id: str, bot: str, bot_device: str, *, encrypted: bool) -> dict:
    await client.sync(timeout=0)
    if encrypted:
        uploaded = await client.keys_upload()
        assert isinstance(uploaded, KeysUploadResponse), uploaded
        changed = await client.room_put_state(
            room_id, "m.room.encryption", {"algorithm": "m.megolm.v1.aes-sha2"},
        )
        assert isinstance(changed, RoomPutStateResponse), changed
        while not client.rooms[room_id].encrypted:
            await client.sync(timeout=250)
        members = await client.joined_members(room_id)
        assert isinstance(members, JoinedMembersResponse), members
        queried = await client.keys_query()
        assert isinstance(queried, KeysQueryResponse), queried
        assert bot_device in client.device_store[bot], queried
    root = await send(client, room_id, {"msgtype": "m.notice", "body": "Original thread root"})
    forged = await send(client, room_id, {"msgtype": "m.notice", "body": "Unrelated thread root"})
    relation = {
        "rel_type": "m.thread", "event_id": root,
        "m.in_reply_to": {"event_id": root}, "is_falling_back": True,
    }
    original = await send(client, room_id, {
        "msgtype": "m.text", "body": "Original prompt [in:original]",
        "m.mentions": {"user_ids": [bot]}, "m.relates_to": relation,
    })
    reply = await final(client, room_id, bot, "Matrix live reply", encrypted=encrypted)
    expected_reply = {
        "rel_type": "m.thread", "event_id": root, "is_falling_back": False,
        "m.in_reply_to": {"event_id": original},
    }
    assert reply == expected_reply, {"actual": reply, "expected": expected_reply}
    blocked = await send(client, room_id, {
        "msgtype": "m.text", "body": "Second prompt @matrix-live:pause [in:second]",
        "m.mentions": {"user_ids": [bot]}, "m.relates_to": relation,
    })
    return {"root": root, "forged": forged, "original": original, "blocked": blocked}


async def edits(client, room_id: str, bot: str, ids: dict, foreign_original: str) -> list[str]:
    await client.sync(timeout=0)
    sent = []
    for target, body in [
        (foreign_original, "Rejected author forgery [in:forgery]"),
        (ids["original"], "First correction [in:obsolete]"),
        (ids["original"], "Latest correction [in:latest]"),
    ]:
        sent.append(await send(client, room_id, {
            "msgtype": "m.text", "body": f"* {body}",
            "m.relates_to": {"rel_type": "m.replace", "event_id": target},
            "m.new_content": {
                "msgtype": "m.text", "body": body,
                "m.mentions": {"user_ids": [bot]},
                "m.relates_to": {"rel_type": "m.thread", "event_id": ids["forged"]},
            },
        }))
    return sent


async def finish(client, room_id: str, bot: str, ids: dict, *, encrypted: bool, enabled: bool) -> list[dict]:
    expected = ["Second response"] + (["Correction response"] if enabled else [])
    relations = []
    observed = []
    while len(relations) < len(expected):
        response = await client.sync(timeout=250)
        assert isinstance(response, SyncResponse), response
        joined = response.rooms.join.get(room_id)
        if joined is None:
            continue
        for event in joined.timeline.events:
            if (not isinstance(event, RoomMessageText) or event.sender != bot
                    or event.body not in expected or event.event_id in observed):
                continue
            assert event.decrypted is encrypted, event.source
            relation = event.source["content"].get("m.relates_to", {})
            expected_relation = {
                "rel_type": "m.thread", "event_id": ids["root"], "is_falling_back": bool(observed),
                "m.in_reply_to": {"event_id": observed[-1] if observed else ids["blocked"]},
            }
            assert (event.body, relation) == (expected[len(observed)], expected_relation), {
                "response": event.body, "actual": relation, "expected": expected_relation,
            }
            observed.append(event.event_id)
            relations.append(relation)
    return relations
