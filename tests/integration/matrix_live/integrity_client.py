"""Read encrypted reaction pages and replacements from an independent device."""

from __future__ import annotations

import asyncio
import copy
import json
from textwrap import dedent

from nio import RoomRedactResponse

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.resolution_client import _client_code, write_json


_PREPARE = dedent("""
    import aiohttp
    import uuid
    from urllib.parse import quote
    from nio import JoinedMembersResponse, KeysQueryResponse, KeysUploadResponse, RoomPutStateResponse

    async def encrypted_ready(client):
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
        await send(client, {'msgtype': 'm.text', 'body': f'{BOT_USER} establish',
                            'm.mentions': {'user_ids': [BOT_USER]}})
        await next_reply(client, 'Matrix live reply')

    async def raw_send(client, content):
        url = (f'{client.homeserver}/_matrix/client/v3/rooms/{quote(ROOM_ID, safe="")}'
               f'/send/m.room.encrypted/{uuid.uuid4().hex}')
        async with aiohttp.ClientSession(headers={'Authorization': f'Bearer {client.access_token}'}) as http:
            async with http.put(url, json=content) as response:
                assert response.status == 200
                return (await response.json())['event_id']
""")


def _queue_read(gateway, target):
    gateway.model.push(
        ToolCall("tool_search", {"queries": ["Matrix read event"]}),
        ToolCall(
            "tool_call",
            {
                "calls": [
                    {
                        "name": "matrix_read",
                        "arguments": {"kind": "event", "event_id": target},
                    }
                ]
            },
        ),
        Text("Integrity complete"),
    )


def _question(observer, code):
    return observer.run_python(
        code
        + dedent("""
        async def question():
            client = open_encrypted_client()
            try:
                response = await client.sync(timeout=0, full_state=True)
                assert isinstance(response, SyncResponse), response
                await send(client, {'msgtype': 'm.text', 'body': 'integrity question'})
                await next_reply(client, 'Integrity complete')
            finally:
                await client.close()
        asyncio.run(asyncio.wait_for(question(), timeout=20))
    """)
    )


def _read_result(gateway, established):
    requests = gateway.model.main_requests()
    assert len(requests) == 4
    assert requests[0] == established[0]
    previous = established[0]["messages"]
    assert requests[1]["messages"][: len(previous)] == previous
    assert requests[0]["messages"][0] == requests[-1]["messages"][0]
    roles = [message["role"] for message in requests[-1]["messages"]]
    assert all(left != right for left, right in zip(roles, roles[1:]))
    tools = [
        message for message in requests[-1]["messages"] if message["role"] == "tool"
    ]
    result = json.loads(tools[-1]["content"])
    assert isinstance(result["events"][0]["timestamp"], int)
    result["events"][0]["timestamp"] = None
    return result


def _intake_observed(home, event_ids):
    async def wait():
        path = home / "resolution-observed-events.json"
        async with asyncio.timeout(5):
            while not path.exists() or not event_ids.issubset(
                json.loads(path.read_text(encoding="utf-8"))
            ):
                await asyncio.sleep(0.01)

    asyncio.run(wait())


def check_reaction_page(tmp_path, gateway, room, observer, *, eviction):
    code = _client_code(room)
    output = observer.run_python(
        code
        + _PREPARE
        + dedent("""
        async def prepare():
            client = open_encrypted_client()
            try:
                await encrypted_ready(client)
                target = await send(client, {'msgtype': 'm.notice', 'body': 'Reaction snapshot target'})
                reactions = []
                for key in ('WITHDRAWN REACTION', 'RETAINED REACTION'):
                    kind, content = client.encrypt(ROOM_ID, 'm.reaction', {
                        'm.relates_to': {'rel_type': 'm.annotation', 'event_id': target, 'key': key},
                    })
                    assert kind == 'm.room.encrypted'
                    reactions.append(await raw_send(client, content))
                print(json.dumps({'target': target, 'withdraw': reactions[0], 'replacement': reactions[1]}))
            finally:
                await client.close()
        asyncio.run(asyncio.wait_for(prepare(), timeout=20))
    """)
    )
    targets = json.loads(output.strip().splitlines()[-1])
    home = tmp_path / "hermes"
    _intake_observed(home, {targets["target"]})
    established = copy.deepcopy(gateway.model.main_requests())
    write_json(home / "resolution-config.json", {
        **targets,
        "scope": "event",
        "barrier": "reaction",
        "eviction": eviction,
    })
    _queue_read(gateway, targets["target"])

    async def exchange():
        pending = asyncio.create_task(asyncio.to_thread(_question, observer, code))
        try:
            async with asyncio.timeout(5):
                while not (home / "resolution-started.json").exists():
                    await asyncio.sleep(0.01)
            observed = json.loads(
                (home / "resolution-started.json").read_text(encoding="utf-8")
            )
            assert observed["event_id"] == targets["replacement"]
            assert observed["reaction_page"] == [
                targets["replacement"],
                targets["withdraw"],
            ]
            assert (observed["user_id"], observed["home"], observed["store_type"]) == (
                room.bot.user_id,
                "/opt/data",
                "mautrix.crypto.store.asyncpg.store.PgCryptoStore",
            )
            client = room.observer.client(room.homeserver)
            try:
                redacted = await asyncio.wait_for(
                    client.room_redact(room.room_id, targets["withdraw"]), timeout=5
                )
                assert isinstance(redacted, RoomRedactResponse), redacted
            finally:
                await client.close()
            async with asyncio.timeout(5):
                while not (home / "resolution-redaction.json").exists():
                    await asyncio.sleep(0.01)
            redaction = json.loads(
                (home / "resolution-redaction.json").read_text(encoding="utf-8")
            )
            assert redaction == {
                "event_id": targets["withdraw"],
                "room_id": room.room_id,
                "entries": 2,
                "limit": 2,
            }
        finally:
            (home / "resolution-release").write_text("release", encoding="utf-8")
            await pending

    asyncio.run(asyncio.wait_for(exchange(), timeout=20))
    assert _read_result(gateway, established) == {
        "events": [
            {
                "event_id": targets["target"],
                "sender": room.observer.user_id,
                "body": "[notice: Reaction snapshot target]",
                "msgtype": "m.notice",
                "thread_id": None,
                "timestamp": None,
                "sender_authorized": True,
                "reactions": [
                    {
                        "event_id": targets["replacement"],
                        "sender": room.observer.user_id,
                        "emoji": "RETAINED REACTION",
                        "target_event_id": targets["target"],
                        "sender_authorized": True,
                    }
                ],
            }
        ],
        "errors": [],
    }


def check_replacement_relation(tmp_path, gateway, room, observer, *, relation_kind):
    code = _client_code(room)
    output = observer.run_python(
        f"RELATION_KIND = {relation_kind!r}\n"
        + code
        + _PREPARE
        + dedent("""
        from nio import Event, MegolmEvent

        async def prepare():
            client = open_encrypted_client()
            try:
                await encrypted_ready(client)
                target = await send(client, {'msgtype': 'm.notice', 'body': 'Unchanged decision'})
                other = await send(client, {'msgtype': 'm.notice', 'body': 'Other decision'})
                content = {'msgtype': 'm.notice', 'body': '* Replacement decision',
                           'm.new_content': {'msgtype': 'm.notice', 'body': 'Replacement decision'}}
                if RELATION_KIND != 'outer-only':
                    content['m.relates_to'] = {'rel_type': 'm.replace',
                                               'event_id': other if RELATION_KIND == 'wrong-target' else target}
                kind, encrypted = client.encrypt(ROOM_ID, 'm.room.message', content)
                assert kind == 'm.room.encrypted'
                encrypted['m.relates_to'] = {'rel_type': 'm.replace', 'event_id': target}
                replacement = await raw_send(client, encrypted)
                url = f'{client.homeserver}/_matrix/client/v3/rooms/{quote(ROOM_ID, safe="")}/event/{quote(target, safe="")}'
                async with aiohttp.ClientSession(headers={'Authorization': f'Bearer {client.access_token}'}) as http:
                    async with http.get(url) as response:
                        assert response.status == 200
                        raw = await response.json()
                edit = raw['unsigned']['m.relations']['m.replace']
                assert (edit['event_id'], edit['type'], edit['sender']) == (replacement, 'm.room.encrypted', client.user_id), edit
                assert edit['content']['m.relates_to'] == {'rel_type': 'm.replace', 'event_id': target}
                parsed = Event.parse_event(edit)
                assert isinstance(parsed, MegolmEvent), parsed
                parsed.room_id = ROOM_ID
                clear = client.decrypt_event(parsed).source['content']
                expected_target = other if RELATION_KIND == 'wrong-target' else target
                assert clear['m.relates_to'] == {'rel_type': 'm.replace', 'event_id': expected_target}, clear
                assert clear['m.new_content'] == content['m.new_content']
                print(json.dumps({'target': target, 'replacement': replacement, 'clear_target': expected_target}))
            finally:
                await client.close()
        asyncio.run(asyncio.wait_for(prepare(), timeout=20))
    """)
    )
    targets = json.loads(output.strip().splitlines()[-1])
    home = tmp_path / "hermes"
    _intake_observed(home, {targets["target"], targets["replacement"]})
    established = copy.deepcopy(gateway.model.main_requests())
    write_json(home / "resolution-config.json", {
        **targets,
        "scope": "event",
        "barrier": "relation",
        "withdraw": "",
    })
    _queue_read(gateway, targets["target"])
    _question(observer, code)
    decrypted = json.loads(
        (home / "resolution-decrypted.json").read_text(encoding="utf-8")
    )
    assert decrypted == {
        "event_id": targets["replacement"],
        "relation": {"rel_type": "m.replace", "event_id": targets["clear_target"]},
        "user_id": room.bot.user_id,
        "home": "/opt/data",
        "store_type": "mautrix.crypto.store.asyncpg.store.PgCryptoStore",
    }
    valid = relation_kind != "wrong-target"
    assert _read_result(gateway, established) == {
        "events": [
            {
                "event_id": targets["target"],
                "sender": room.observer.user_id,
                "body": "[notice: Replacement decision]"
                if valid
                else "[notice: Unchanged decision]",
                "msgtype": "m.notice",
                "thread_id": None,
                "timestamp": None,
                "sender_authorized": True,
                **({"edited": True} if valid else {}),
            }
        ],
        "errors": [],
    }
