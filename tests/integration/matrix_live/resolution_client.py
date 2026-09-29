"""Drive withdrawal from a separate encrypted client at an adapter barrier."""

from __future__ import annotations

import asyncio
import copy
import json
from textwrap import dedent

from nio import RoomRedactResponse

from tests.fakes.fake_llm_provider import Text, ToolCall


def write_json(path, value):
    """Replace a hand-off file in one rename so the other side never reads it half written."""
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value), encoding="utf-8")
    partial.replace(path)


def _client_code(room):
    return (
        f"ROOM_ID = {room.room_id!r}\nBOT_USER = {room.bot.user_id!r}\nBOT_DEVICE = {room.bot.device_id!r}\n"
        + dedent("""
        import asyncio
        import json
        from nio import RoomMessageText, RoomSendResponse, SyncResponse
        from client import open_encrypted_client

        async def send(client, content):
            sent = await client.room_send(ROOM_ID, 'm.room.message', content, ignore_unverified_devices=True)
            assert isinstance(sent, RoomSendResponse), sent
            return sent.event_id

        async def next_reply(client, expected):
            while True:
                response = await client.sync(timeout=250)
                assert isinstance(response, SyncResponse), response
                joined = response.rooms.join.get(ROOM_ID)
                if joined is None:
                    continue
                for event in joined.timeline.events:
                    if isinstance(event, RoomMessageText) and event.sender == BOT_USER and event.body == expected:
                        assert event.decrypted, event.source
                        return
    """)
    )


def check_resolution(tmp_path, gateway, room, observer, *, scope, barrier, replacement):
    code = _client_code(room)
    output = observer.run_python(
        f"SCOPE = {scope!r}\nREPLACEMENT = {replacement!r}\n"
        + code
        + dedent("""
        import aiohttp
        from urllib.parse import quote
        from nio import JoinedMembersResponse, KeysQueryResponse, KeysUploadResponse, RoomPutStateResponse

        async def prepare():
            client = open_encrypted_client()
            try:
                response = await client.sync(timeout=0, full_state=True)
                assert isinstance(response, SyncResponse), response
                uploaded = await client.keys_upload()
                assert isinstance(uploaded, KeysUploadResponse), uploaded
                encryption = await client.room_put_state(ROOM_ID, 'm.room.encryption', {'algorithm': 'm.megolm.v1.aes-sha2'})
                assert isinstance(encryption, RoomPutStateResponse), encryption
                while not client.rooms[ROOM_ID].encrypted:
                    response = await client.sync(timeout=250)
                    assert isinstance(response, SyncResponse), response
                members = await client.joined_members(ROOM_ID)
                assert isinstance(members, JoinedMembersResponse), members
                queried = await client.keys_query()
                assert isinstance(queried, KeysQueryResponse), queried
                assert BOT_DEVICE in client.device_store[BOT_USER], queried
                root = await send(client, {'msgtype': 'm.notice', 'body': 'Thread root'}) if SCOPE.startswith('thread') else None
                establish = {'msgtype': 'm.text', 'body': f'{BOT_USER} establish', 'm.mentions': {'user_ids': [BOT_USER]}}
                if root:
                    establish['m.relates_to'] = {'rel_type': 'm.thread', 'event_id': root,
                                                 'm.in_reply_to': {'event_id': root}, 'is_falling_back': True}
                await send(client, establish)
                await next_reply(client, 'Matrix live reply')
                original = {'msgtype': 'm.notice', 'body': 'WITHDRAWN ORIGINAL'}
                if SCOPE == 'thread-child':
                    original['m.relates_to'] = {'rel_type': 'm.thread', 'event_id': root,
                                                'm.in_reply_to': {'event_id': root}, 'is_falling_back': True}
                target = await send(client, original)
                if SCOPE == 'thread-root':
                    root = target
                latest = None
                if REPLACEMENT:
                    for text in ('SURVIVING EDIT', 'WITHDRAWN EDIT'):
                        latest = await send(client, {
                            'msgtype': 'm.notice', 'body': f'* {text}',
                            'm.new_content': {'msgtype': 'm.notice', 'body': text},
                            'm.relates_to': {'rel_type': 'm.replace', 'event_id': target},
                        })
                url = f'{client.homeserver}/_matrix/client/v3/rooms/{quote(ROOM_ID, safe="")}/event/{quote(target, safe="")}'
                async with aiohttp.ClientSession(headers={'Authorization': f'Bearer {client.access_token}'}) as http:
                    async with http.get(url) as response:
                        assert response.status == 200
                        raw = await response.json()
                assert (raw['event_id'], raw['type'], raw['sender']) == (target, 'm.room.encrypted', client.user_id), raw
                if REPLACEMENT:
                    edit = raw['unsigned']['m.relations']['m.replace']
                    assert (edit['event_id'], edit['type'], edit['sender']) == (latest, 'm.room.encrypted', client.user_id), edit
                print(json.dumps({'target': target, 'replacement': latest, 'root': root}))
            finally:
                await client.close()

        asyncio.run(asyncio.wait_for(prepare(), timeout=20))
    """)
    )
    targets = json.loads(output.strip().splitlines()[-1])
    home = tmp_path / "hermes"

    async def intake_observed():
        path = home / "resolution-observed-events.json"
        expected = {targets["target"]}
        if targets["replacement"]:
            expected.add(targets["replacement"])
        while not path.exists() or not expected.issubset(
            json.loads(path.read_text(encoding="utf-8"))
        ):
            await asyncio.sleep(0.01)

    asyncio.run(asyncio.wait_for(intake_observed(), timeout=5))
    established = copy.deepcopy(gateway.model.main_requests())
    withdraw = targets["replacement"] if replacement else targets["target"]
    write_json(home / "resolution-config.json", {
        **targets,
        "withdraw": withdraw,
        "scope": scope,
        "barrier": barrier,
    })
    if scope == "event":
        gateway.model.push(
            ToolCall("tool_search", {"queries": ["Matrix read event"]}),
            ToolCall(
                "tool_call",
                {
                    "calls": [
                        {
                            "name": "matrix_read",
                            "arguments": {
                                "kind": "event",
                                "event_id": targets["target"],
                            },
                        }
                    ]
                },
            ),
            Text("Resolution complete"),
        )
    else:
        gateway.model.push(Text("Resolution complete"))
    command = {
        "msgtype": "m.text",
        "body": "resolution question"
        if scope == "event"
        else f"{room.bot.user_id} resolution question",
        **(
            {"m.mentions": {"user_ids": [room.bot.user_id]}} if scope != "event" else {}
        ),
    }
    if targets["root"]:
        command["m.relates_to"] = {
            "rel_type": "m.thread",
            "event_id": targets["root"],
            "m.in_reply_to": {"event_id": targets["root"]},
            "is_falling_back": True,
        }

    async def exchange():
        pending = asyncio.create_task(
            asyncio.to_thread(
                observer.run_python,
                code
                + dedent(f"""
            async def question():
                client = open_encrypted_client()
                try:
                    response = await client.sync(timeout=0, full_state=True)
                    assert isinstance(response, SyncResponse), response
                    await send(client, {command!r})
                    await next_reply(client, 'Resolution complete')
                finally:
                    await client.close()
            asyncio.run(asyncio.wait_for(question(), timeout=20))
        """),
            )
        )
        try:
            while not (home / "resolution-started.json").exists():
                await asyncio.sleep(0.01)
            observed = json.loads(
                (home / "resolution-started.json").read_text(encoding="utf-8")
            )
            assert observed["event_id"] == (
                targets["replacement"]
                if barrier in {"replacement", "store"}
                else targets["target"]
            )
            assert observed["user_id"] == room.bot.user_id
            assert observed["home"] == "/opt/data"
            assert observed["adapter_module"] != "plugins.platforms.matrix.adapter", (
                observed
            )
            assert (
                observed["store_type"]
                == "mautrix.crypto.store.asyncpg.store.PgCryptoStore"
            ), observed
            client = room.observer.client(room.homeserver)
            try:
                redaction = await asyncio.wait_for(
                    client.room_redact(room.room_id, withdraw), timeout=5
                )
                assert isinstance(redaction, RoomRedactResponse), redaction
            finally:
                await client.close()
            while not (home / "resolution-redaction.json").exists():
                await asyncio.sleep(0.01)
            eviction = json.loads(
                (home / "resolution-redaction.json").read_text(encoding="utf-8")
            )
            assert eviction == {
                "event_id": withdraw,
                "room_id": room.room_id,
                "entries": 2,
                "limit": 2,
            }
        finally:
            (home / "resolution-release").write_text("release", encoding="utf-8")
            await pending

    asyncio.run(asyncio.wait_for(exchange(), timeout=20))
    requests = gateway.model.main_requests()
    assert len(requests) == (4 if scope == "event" else 2)
    assert requests[0] == established[0]
    previous = established[0]["messages"]
    if scope != "thread-root":
        assert requests[1]["messages"][: len(previous)] == previous
        assert requests[0]["messages"][0] == requests[-1]["messages"][0]
    current_messages = (
        requests[-1]["messages"][1:]
        if scope == "thread-root"
        else requests[-1]["messages"][len(previous) :]
    )
    current = json.dumps(current_messages)
    if scope == "thread-root":
        assert "establish" not in current
    assert "resolution question" in current
    assert "WITHDRAWN ORIGINAL" not in current
    assert "WITHDRAWN EDIT" not in current
    assert ("SURVIVING EDIT" if replacement else "[redacted]") in current
    roles = [message["role"] for message in requests[-1]["messages"]]
    assert all(left != right for left, right in zip(roles, roles[1:]))
    if scope == "event":
        tools = [
            message
            for message in requests[-1]["messages"]
            if message.get("role") == "tool"
        ]
        read = json.loads(tools[-1]["content"])
        event = read["events"][0]
        assert isinstance(event["timestamp"], int)
        event["timestamp"] = None
        assert read == {
            "events": [
                {
                    "event_id": targets["target"],
                    "sender": room.observer.user_id,
                    "body": "[notice: SURVIVING EDIT]" if replacement else "[redacted]",
                    "msgtype": "m.notice" if replacement else None,
                    "thread_id": None,
                    "timestamp": None,
                    "sender_authorized": True,
                    **({"edited": True} if replacement else {"redacted": True}),
                }
            ],
            "errors": [],
        }
