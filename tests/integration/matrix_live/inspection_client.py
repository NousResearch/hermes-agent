"""Inspect effective pins through a separate Matrix client and captured model calls."""

from __future__ import annotations

import copy
import json
from textwrap import dedent

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import (
    LinuxNioObserver,
    LiveRoom,
)
from tests.integration.matrix_live.live_gateway import LiveGateway


def check_inspection(
    gateway: LiveGateway,
    room: LiveRoom,
    observer: LinuxNioObserver,
    *,
    encrypted: bool,
) -> None:
    code = (
        f"ROOM_ID = {room.room_id!r}\nBOT_USER = {room.bot.user_id!r}\n"
        f"BOT_DEVICE = {room.bot.device_id!r}\nENCRYPTED = {encrypted!r}\n"
        + dedent("""
            import asyncio
            import json
            from urllib.parse import quote
            import aiohttp
            from nio import (
                JoinedMembersResponse, KeysQueryResponse, KeysUploadResponse,
                RoomMessageText, RoomPutStateResponse, RoomRedactResponse,
                RoomSendResponse, SyncResponse,
            )
            from client import open_encrypted_client

            async def send(client, content):
                result = await client.room_send(
                    ROOM_ID, 'm.room.message', content, ignore_unverified_devices=True,
                )
                assert isinstance(result, RoomSendResponse), result
                return result.event_id

            async def next_reply(client, expected):
                while True:
                    response = await client.sync(timeout=250)
                    assert isinstance(response, SyncResponse), response
                    joined = response.rooms.join.get(ROOM_ID)
                    if joined is None:
                        continue
                    for event in joined.timeline.events:
                        if isinstance(event, RoomMessageText) and event.sender == BOT_USER and event.body == expected:
                            assert event.decrypted is ENCRYPTED, event.source
                            return

            async def raw_event(client, event_id):
                url = (f'{client.homeserver}/_matrix/client/v3/rooms/'
                       f'{quote(ROOM_ID, safe="")}/event/{quote(event_id, safe="")}')
                async with aiohttp.ClientSession(headers={'Authorization': f'Bearer {client.access_token}'}) as http:
                    async with http.get(url) as response:
                        assert response.status == 200
                        return await response.json()
        """)
    )
    output = observer.run_python(
        code
        + dedent("""
        async def prepare():
            client = open_encrypted_client()
            try:
                response = await client.sync(timeout=0, full_state=True)
                assert isinstance(response, SyncResponse), response
                if ENCRYPTED:
                    uploaded = await client.keys_upload()
                    assert isinstance(uploaded, KeysUploadResponse), uploaded
                    encryption = await client.room_put_state(
                        ROOM_ID, 'm.room.encryption', {'algorithm': 'm.megolm.v1.aes-sha2'},
                    )
                    assert isinstance(encryption, RoomPutStateResponse), encryption
                    while not client.rooms[ROOM_ID].encrypted:
                        response = await client.sync(timeout=250)
                        assert isinstance(response, SyncResponse), response
                    members = await client.joined_members(ROOM_ID)
                    assert isinstance(members, JoinedMembersResponse), members
                    queried = await client.keys_query()
                    assert isinstance(queried, KeysQueryResponse), queried
                    assert BOT_DEVICE in client.device_store[BOT_USER], queried
                await send(client, {'msgtype': 'm.text', 'body': 'Start room inspection'})
                await next_reply(client, 'Matrix live reply')
                topic = await client.room_put_state(ROOM_ID, 'm.room.topic', {'topic': 'Release notes'})
                assert isinstance(topic, RoomPutStateResponse), topic
                target = await send(client, {'msgtype': 'm.notice', 'body': 'Draft pinned plan'})
                edit = await send(client, {
                    'msgtype': 'm.notice', 'body': '* Final pinned plan',
                    'm.new_content': {'msgtype': 'm.notice', 'body': 'Final pinned plan'},
                    'm.relates_to': {'rel_type': 'm.replace', 'event_id': target},
                })
                withdrawn = await send(client, {'msgtype': 'm.notice', 'body': 'Withdrawn pinned plan'})
                pins = await client.room_put_state(ROOM_ID, 'm.room.pinned_events', {'pinned': [target, withdrawn]})
                assert isinstance(pins, RoomPutStateResponse), pins
                redaction = await client.room_redact(ROOM_ID, withdrawn)
                assert isinstance(redaction, RoomRedactResponse), redaction
                raw = await raw_event(client, target)
                replacement = raw['unsigned']['m.relations']['m.replace']
                expected_type = 'm.room.encrypted' if ENCRYPTED else 'm.room.message'
                assert (raw['event_id'], raw['type'], raw['sender']) == (target, expected_type, client.user_id), raw
                assert (replacement['event_id'], replacement['type'], replacement['sender']) == (edit, expected_type, client.user_id), replacement
                redacted = await raw_event(client, withdrawn)
                assert redacted['unsigned']['redacted_because']['sender'] == client.user_id, redacted
                url = f'{client.homeserver}/_matrix/client/v3/rooms/{quote(ROOM_ID, safe="")}/joined_members'
                async with aiohttp.ClientSession(headers={'Authorization': f'Bearer {client.access_token}'}) as http:
                    async with http.get(url) as response:
                        assert response.status == 200
                        profiles = (await response.json())['joined']
                assert set(profiles) == {client.user_id, BOT_USER}, profiles
                print(json.dumps({'target': target, 'withdrawn': withdrawn, 'members': profiles}))
            finally:
                await client.close()
        asyncio.run(asyncio.wait_for(prepare(), timeout=20))
    """)
    )
    targets = json.loads(output.strip().splitlines()[-1])
    established = copy.deepcopy(gateway.model.main_requests())
    assert len(established) == 1
    gateway.model.push(
        ToolCall(
            "tool_search", {"queries": ["Matrix room state members permissions pins"]}
        ),
        ToolCall(
            "tool_call",
            {"calls": [{"name": "matrix_read", "arguments": {"kind": "state"}}]},
            parallel=[
                (
                    "tool_call",
                    {"calls": [{"name": "matrix_read", "arguments": {"kind": kind}}]},
                )
                for kind in ("members", "permissions", "pins")
            ],
        ),
        Text("Inspection complete"),
    )
    observer.run_python(
        code
        + dedent("""
        async def inspect():
            client = open_encrypted_client()
            try:
                response = await client.sync(timeout=0, full_state=True)
                assert isinstance(response, SyncResponse), response
                await send(client, {'msgtype': 'm.text', 'body': 'Inspect this room'})
                await next_reply(client, 'Inspection complete')
            finally:
                await client.close()
        asyncio.run(asyncio.wait_for(inspect(), timeout=20))
    """)
    )
    requests = copy.deepcopy(gateway.model.main_requests())
    assert len(requests) == 4
    previous = established[0]["messages"]
    assert requests[1]["messages"][: len(previous)] == previous
    assert requests[1]["tools"] == established[0]["tools"]
    results = [
        json.loads(message["content"])
        for message in requests[3]["messages"]
        if message["role"] == "tool"
    ]
    assert len(results) == 5
    assert "matrix_read" in results[0]["tools"]
    state, members, permissions, pins = results[1:]
    assert state == {
        "room_id": room.room_id,
        "name": "Matrix live test",
        "topic": "Release notes",
        "canonical_alias": None,
        "join_rule": "invite",
        "history_visibility": "shared",
        "encryption": "m.megolm.v1.aes-sha2" if encrypted else None,
    }
    assert members == {
        "members": [
            {
                "user_id": user_id,
                "display_name": profile.get("display_name") or None,
                "avatar_url": profile.get("avatar_url") or None,
            }
            for user_id, profile in sorted(targets["members"].items())
        ],
        "total": 2,
        "truncated": False,
    }, members
    assert permissions == {
        "requester": {
            "user_id": room.observer.user_id,
            "level": 100,
            "creator_override": False,
        },
        "bot": {"user_id": room.bot.user_id, "level": 0, "creator_override": False},
        "required": {
            "send_message": 0,
            "send_event_type": "m.room.encrypted" if encrypted else "m.room.message",
            "edit_pins": 50,
            "invite": 0,
            "kick": 50,
            "ban": 50,
            "redact_other": 50,
            "send_redaction": 0,
        },
        "bot_can_edit_pins": False,
    }
    base_events = [
        {
            "event_id": event_id,
            "sender": room.observer.user_id,
            "body": "[redacted]",
            "msgtype": None,
            "thread_id": None,
            "timestamp": None,
            "sender_authorized": True,
        }
        for event_id in (targets["target"], targets["withdrawn"])
    ]
    assert all(isinstance(event["timestamp"], int) for event in pins["events"])
    assert {
        **pins,
        "events": [{**event, "timestamp": None} for event in pins["events"]],
    } == {
        "events": [
            {
                **base_events[0],
                "body": "[notice: Final pinned plan]",
                "msgtype": "m.notice",
                "edited": True,
            },
            {**base_events[1], "redacted": True},
        ],
        "total": 2,
        "truncated": False,
        "errors": [],
    }
    gateway.model.push(
        ToolCall(
            "tool_call",
            {"calls": [{"name": "matrix_read", "arguments": {"kind": "pins"}}]},
        ),
        Text("Redacted inspection complete"),
    )
    observer.run_python(
        f"TARGET = {targets['target']!r}\n"
        + code
        + dedent("""
        async def inspect_redacted():
            client = open_encrypted_client()
            try:
                response = await client.sync(timeout=0, full_state=True)
                assert isinstance(response, SyncResponse), response
                redaction = await client.room_redact(ROOM_ID, TARGET)
                assert isinstance(redaction, RoomRedactResponse), redaction
                raw = await raw_event(client, TARGET)
                assert raw['unsigned']['redacted_because']['sender'] == client.user_id, raw
                await send(client, {'msgtype': 'm.text', 'body': 'Inspect the withdrawn pins'})
                await next_reply(client, 'Redacted inspection complete')
            finally:
                await client.close()
        asyncio.run(asyncio.wait_for(inspect_redacted(), timeout=20))
    """)
    )
    final_requests = gateway.model.main_requests()
    assert len(final_requests) == 6
    assert gateway.model.aux_requests() == []
    previous = requests[3]["messages"]
    assert final_requests[4]["messages"][: len(previous)] == previous
    assert final_requests[4]["tools"] == requests[3]["tools"]
    final_tools = [
        json.loads(message["content"])
        for message in final_requests[5]["messages"]
        if message["role"] == "tool"
    ]
    assert len(final_tools) == 6
    result = final_tools[-1]
    assert all(isinstance(event["timestamp"], int) for event in result["events"])
    assert {
        **result,
        "events": [{**event, "timestamp": None} for event in result["events"]],
    } == {
        "events": [{**event, "redacted": True} for event in base_events],
        "total": 2,
        "truncated": False,
        "errors": [],
    }
