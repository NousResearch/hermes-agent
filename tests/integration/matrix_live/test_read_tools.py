"""A separate Matrix client verifies model-visible reads through the live gateway."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from textwrap import dedent

import pytest
from nio import RoomMessageText, RoomSendResponse

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import LinuxNioObserver, LiveGateway, LiveRoom
from tests.integration.matrix_live.inspection_client import check_inspection


def test_model_reads_an_event_and_the_room_from_its_live_matrix_session(
    gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        try:
            await asyncio.wait_for(client.sync(timeout=0), timeout=15)

            async def send_and_wait(body: str, reply: str) -> str:
                sent = await client.room_send(
                    live_room.room_id, "m.room.message", {"msgtype": "m.text", "body": body},
                )
                assert isinstance(sent, RoomSendResponse), sent

                while True:
                    response = await client.sync(timeout=250)
                    joined = response.rooms.join.get(live_room.room_id)
                    if not joined:
                        continue
                    for event in joined.timeline.events:
                        if (
                            isinstance(event, RoomMessageText)
                            and event.sender == live_room.bot.user_id
                            and event.body == reply
                        ):
                            return sent.event_id

            target = await asyncio.wait_for(
                send_and_wait("Read target [history:blue]", "Matrix live reply"), timeout=15,
            )
            gateway.model.push(
                ToolCall("tool_search", {"queries": ["Matrix read event"]}),
                ToolCall("tool_call", {"calls": [{
                    "name": "matrix_read", "arguments": {"kind": "event", "event_id": target},
                }]}),
                ToolCall("tool_call", {"calls": [{
                    "name": "matrix_read", "arguments": {"kind": "room", "limit": 10},
                }]}),
                Text("Read complete"),
            )
            await asyncio.wait_for(send_and_wait("Read the earlier Matrix event", "Read complete"), timeout=15)

            requests = gateway.model.main_requests()
            assert len(requests) == 5
            search_messages = [message for message in requests[2]["messages"] if message["role"] == "tool"]
            assert len(search_messages) == 1
            assert "matrix_read" in json.loads(search_messages[0]["content"])["tools"]
            tool_messages = [message for message in requests[4]["messages"] if message["role"] == "tool"]
            assert len(tool_messages) == 3
            room = json.loads(tool_messages[2]["content"])
            timestamps = [event["timestamp"] for event in room["events"]]
            observer_bodies = [
                event["body"] for event in room["events"] if event["sender"] == live_room.observer.user_id
            ]
            assert (observer_bodies, timestamps == sorted(timestamps), room["errors"], room["skipped"]) == (
                ["Read target [history:blue]", "Read the earlier Matrix event"], True, [], 0,
            )
            result = json.loads(tool_messages[1]["content"])
            assert isinstance(result["events"][0]["timestamp"], int)
            assert {**result, "events": [{**result["events"][0], "timestamp": None}]} == {
                "events": [{
                    "event_id": target,
                    "sender": live_room.observer.user_id,
                    "body": "Read target [history:blue]",
                    "msgtype": "m.text",
                    "thread_id": None,
                    "timestamp": None,
                    "sender_authorized": True,
                }],
                "errors": [],
                "skipped": 0,
            }
        finally:
            await client.close()

    started = time.monotonic()
    try:
        asyncio.run(exchange())
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


def test_model_reads_valid_encrypted_edit_and_ignores_edit_without_new_content(
    gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    record_property: Callable[[str, object], None],
) -> None:
    assert linux_nio_observer.account == live_room.observer
    assert live_room.observer.device_id != live_room.bot.device_id
    client_code = (
        f"ROOM_ID = {live_room.room_id!r}\nBOT_USER = {live_room.bot.user_id!r}\n"
        f"BOT_DEVICE = {live_room.bot.device_id!r}\n"
        + dedent("""
            import asyncio
            import json

            from nio import RoomMessageText, RoomSendResponse, SyncResponse
            from client import open_encrypted_client

            async def send(client, content):
                response = await client.room_send(
                    ROOM_ID, "m.room.message", content, ignore_unverified_devices=True,
                )
                assert isinstance(response, RoomSendResponse), response
                return response.event_id

            async def next_reply(client, expected):
                while True:
                    response = await client.sync(timeout=250)
                    assert isinstance(response, SyncResponse), response
                    joined = response.rooms.join.get(ROOM_ID)
                    if joined is None:
                        continue
                    for event in joined.timeline.events:
                        if (isinstance(event, RoomMessageText)
                                and event.sender == BOT_USER and event.body == expected):
                            assert event.decrypted, event.source
                            return
        """)
    )
    started = time.monotonic()
    try:
        output = linux_nio_observer.run_python(client_code + dedent("""
            from nio import (
                Event, JoinedMembersResponse, KeysQueryResponse, KeysUploadResponse,
                MegolmEvent, RoomPutStateResponse,
            )
            import aiohttp
            from urllib.parse import quote

            async def prepare():
                client = open_encrypted_client()
                try:
                    response = await client.sync(timeout=0)
                    assert isinstance(response, SyncResponse), response
                    uploaded = await client.keys_upload()
                    assert isinstance(uploaded, KeysUploadResponse), uploaded
                    encryption = await client.room_put_state(
                        ROOM_ID, "m.room.encryption", {"algorithm": "m.megolm.v1.aes-sha2"},
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

                    await send(client, {"msgtype": "m.text", "body": "Encrypted handshake"})
                    await next_reply(client, "Matrix live reply")

                    targets = []
                    headers = {"Authorization": f"Bearer {client.access_token}"}
                    async with aiohttp.ClientSession(headers=headers) as http:
                        for valid in (True, False):
                            original = "Draft encrypted decision" if valid else "Unchanged encrypted decision"
                            revised = "Final encrypted decision" if valid else "Rejected encrypted decision"
                            target = await send(client, {"msgtype": "m.notice", "body": original})
                            content = {
                                "msgtype": "m.notice", "body": f"* {revised}",
                                "m.relates_to": {"rel_type": "m.replace", "event_id": target},
                            }
                            if valid:
                                content["m.new_content"] = {"msgtype": "m.notice", "body": revised}
                            replacement = await send(client, content)
                            url = (f"{client.homeserver}/_matrix/client/v3/rooms/"
                                   f"{quote(ROOM_ID, safe='')}/event/{quote(target, safe='')}")
                            async with http.get(url) as response:
                                assert response.status == 200
                                raw = await response.json()
                            assert raw["type"] == "m.room.encrypted", raw
                            edit = raw["unsigned"]["m.relations"]["m.replace"]
                            assert (edit["event_id"], edit["type"], edit["sender"]) == (
                                replacement, "m.room.encrypted", client.user_id,
                            ), edit
                            encrypted = Event.parse_event(edit)
                            assert isinstance(encrypted, MegolmEvent), encrypted
                            encrypted.room_id = ROOM_ID
                            decrypted = client.decrypt_event(encrypted)
                            assert decrypted.source["content"] == content
                            targets.append(target)
                    print(json.dumps(targets))
                finally:
                    await client.close()

            asyncio.run(asyncio.wait_for(prepare(), timeout=20))
        """))
        targets = json.loads(output.strip().splitlines()[-1])
        gateway.model.push(
            ToolCall("tool_search", {"queries": ["Matrix read event"]}),
            *(ToolCall("tool_call", {"calls": [{
                "name": "matrix_read", "arguments": {"kind": "event", "event_id": target},
            }]}) for target in targets),
            Text("Encrypted reads complete"),
        )
        linux_nio_observer.run_python(client_code + dedent("""
            async def read():
                client = open_encrypted_client()
                try:
                    response = await client.sync(timeout=0)
                    assert isinstance(response, SyncResponse), response
                    await send(client, {"msgtype": "m.text", "body": "Read both encrypted decisions"})
                    await next_reply(client, "Encrypted reads complete")
                finally:
                    await client.close()

            asyncio.run(asyncio.wait_for(read(), timeout=15))
        """))

        requests = gateway.model.main_requests()
        assert len(requests) == 5
        tool_messages = [message for message in requests[-1]["messages"] if message["role"] == "tool"]
        results = [json.loads(message["content"]) for message in tool_messages[1:]]
        for result in results:
            assert isinstance(result["events"][0]["timestamp"], int)
            result["events"][0]["timestamp"] = None
        assert results == [{
            "events": [{
                "event_id": targets[0], "sender": live_room.observer.user_id,
                "body": "[notice: Final encrypted decision]", "msgtype": "m.notice",
                "thread_id": None, "timestamp": None, "sender_authorized": True,
                "edited": True,
            }],
            "errors": [],
            "skipped": 0,
        }, {
            "events": [{
                "event_id": targets[1], "sender": live_room.observer.user_id,
                "body": "[notice: Unchanged encrypted decision]", "msgtype": "m.notice",
                "thread_id": None, "timestamp": None, "sender_authorized": True,
            }],
            "errors": [],
            "skipped": 0,
        }]
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


@pytest.mark.parametrize("gateway", ["inspection"], indirect=True)
@pytest.mark.parametrize("encrypted", [False, True])
def test_model_inspects_live_room_state_members_permissions_and_pins(
    gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    record_property: Callable[[str, object], None],
    encrypted: bool,
) -> None:
    assert linux_nio_observer.account == live_room.observer
    assert live_room.observer.device_id != live_room.bot.device_id
    started = time.monotonic()
    try:
        check_inspection(gateway, live_room, linux_nio_observer, encrypted=encrypted)
    except Exception as exc:
        results = [message for request in gateway.model.main_requests()
                   for message in request["messages"] if message["role"] == "tool"]
        logs = gateway.container.get_wrapped_container().logs().decode(errors="replace")[-8000:]
        raise AssertionError(
            f"Room inspection failed: {exc}\nTool results: {results}\n"
            f"Auxiliary calls: {gateway.model.aux_requests()}\nGateway logs:\n{logs}"
        ) from exc
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
