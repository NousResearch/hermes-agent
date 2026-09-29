"""Independent clients verify native polls, replacement votes and passive context."""

from __future__ import annotations

import json
from textwrap import dedent

import pytest

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import LinuxNioObserver, LiveGateway, LiveRoom
from tests.integration.matrix_live.context_client import group_gateway as group_gateway
from tests.integration.matrix_live.context_client import group_member as group_member


def _call(name, arguments):
    return ToolCall("tool_call", {"calls": [{"name": name, "arguments": arguments}]})


def _tool_results(gateway):
    messages = gateway.model.main_requests()[-1]["messages"]
    return [json.loads(message["content"]) for message in messages if message["role"] == "tool"]


@pytest.mark.parametrize("gateway", ["polls"], indirect=True)
@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "encrypted"])
def test_native_poll_exchange_remains_passive(
    group_gateway: LiveGateway, live_room: LiveRoom, linux_nio_observer: LinuxNioObserver,
    encrypted: bool,
) -> None:
    gateway = group_gateway
    prefix = (
        f"ROOM_ID = {live_room.room_id!r}\nBOT_USER = {live_room.bot.user_id!r}\n"
        f"BOT_DEVICE = {live_room.bot.device_id!r}\nENCRYPTED = {encrypted!r}\n"
        + dedent('''
            import asyncio
            import json
            from urllib.parse import quote

            import aiohttp
            from nio import (
                Event, JoinedMembersResponse, KeysQueryResponse, KeysUploadResponse,
                MegolmEvent, RoomMessageText, RoomPutStateResponse, RoomRedactResponse,
                RoomSendResponse, SyncResponse,
            )
            from client import open_encrypted_client

            NATIVE = "org.matrix.msc3381.poll."

            async def send(client, kind, content):
                result = await client.room_send(
                    ROOM_ID, kind, content, ignore_unverified_devices=True,
                )
                assert isinstance(result, RoomSendResponse), result
                return result.event_id

            async def fetch(client, event_id):
                url = (f"{client.homeserver}/_matrix/client/v3/rooms/"
                       f"{quote(ROOM_ID, safe='')}/event/{quote(event_id, safe='')}")
                async with aiohttp.ClientSession(headers={"Authorization": f"Bearer {client.access_token}"}) as http:
                    async with http.get(url) as result:
                        assert result.status == 200
                        raw = await result.json()
                if not ENCRYPTED:
                    assert raw["type"] != "m.room.encrypted", raw
                    return raw
                assert raw["type"] == "m.room.encrypted", raw
                event = Event.parse_event(raw)
                assert isinstance(event, MegolmEvent), event
                event.room_id = ROOM_ID
                decrypted = client.decrypt_event(event)
                assert not isinstance(decrypted, MegolmEvent), decrypted
                return decrypted.source

            async def final(client, expected):
                while True:
                    result = await client.sync(timeout=250)
                    assert isinstance(result, SyncResponse), result
                    joined = result.rooms.join.get(ROOM_ID)
                    if joined is None:
                        continue
                    for event in joined.timeline.events:
                        if (isinstance(event, RoomMessageText)
                                and event.sender == BOT_USER and event.body == expected):
                            assert event.decrypted is ENCRYPTED
                            return

            async def ask(client, body, expected):
                await send(client, "m.room.message", {
                    "msgtype": "m.text", "body": body,
                    "m.mentions": {"user_ids": [BOT_USER]},
                })
                await final(client, expected)

            async def open_client():
                client = open_encrypted_client()
                result = await client.sync(timeout=0, full_state=True)
                assert isinstance(result, SyncResponse), result
                return client
        ''')
    )
    setup = prefix + dedent('''
        async def prepare():
            client = await open_client()
            try:
                if ENCRYPTED:
                    uploaded = await client.keys_upload()
                    assert isinstance(uploaded, KeysUploadResponse), uploaded
                    result = await client.room_put_state(
                        ROOM_ID, "m.room.encryption", {"algorithm": "m.megolm.v1.aes-sha2"},
                    )
                    assert isinstance(result, RoomPutStateResponse), result
                    while not client.rooms[ROOM_ID].encrypted:
                        result = await client.sync(timeout=250)
                        assert isinstance(result, SyncResponse), result
                    members = await client.joined_members(ROOM_ID)
                    assert isinstance(members, JoinedMembersResponse), members
                    queried = await client.keys_query()
                    assert isinstance(queried, KeysQueryResponse), queried
                    assert BOT_DEVICE in client.device_store[BOT_USER], queried
                await ask(client, "Poll handshake", "Matrix live reply")
                poll = await send(client, NATIVE + "start", {
                    NATIVE + "start": {
                        "question": {"org.matrix.msc1767.text": "Which native option?"},
                        "kind": NATIVE + "disclosed", "max_selections": 1,
                        "answers": [{"id": "a", "org.matrix.msc1767.text": "A"},
                                    {"id": "b", "org.matrix.msc1767.text": "B"}],
                    }, "org.matrix.msc1767.text": "Which native option? A or B",
                })
                first = await send(client, NATIVE + "response", {
                    NATIVE + "response": {"answers": ["a"]},
                    "m.relates_to": {"rel_type": "m.reference", "event_id": poll},
                })
                replacement = await send(client, NATIVE + "response", {
                    NATIVE + "response": {"answers": ["b"]},
                    "m.relates_to": {"rel_type": "m.reference", "event_id": poll},
                })
                native = await fetch(client, poll)
                assert native["type"] == NATIVE + "start", native
                for event_id, selection in ((first, "a"), (replacement, "b")):
                    native = await fetch(client, event_id)
                    assert native["type"] == NATIVE + "response", native
                    assert native["content"] == {
                        NATIVE + "response": {"answers": [selection]},
                        "m.relates_to": {"rel_type": "m.reference", "event_id": poll},
                    }, native
                print(json.dumps({"poll": poll, "replacement": replacement}))
            finally:
                await client.close()

        asyncio.run(asyncio.wait_for(prepare(), timeout=30))
    ''')
    external = json.loads(linux_nio_observer.run_python(setup).strip().splitlines()[-1])
    assert (len(gateway.model.main_requests()), gateway.model.aux_requests(), len(gateway.model.requests)) == (1, [], 1)
    poll = external["poll"]
    gateway.model.push(
        ToolCall("tool_search", {"queries": ["native Matrix poll results voting", "Matrix read event"]}),
        _call("matrix_poll_results", {"poll_id": poll}),
        _call("matrix_read", {"kind": "event", "event_id": poll}),
        _call("matrix_poll_vote", {"poll_id": poll, "answers": ["a"]}),
        _call("matrix_poll_results", {"poll_id": poll}),
        Text("Native poll read complete"),
    )
    linux_nio_observer.run_python(prefix + dedent('''
        async def read():
            client = await open_client()
            try:
                await ask(client, "Inspect the native poll", "Native poll read complete")
            finally:
                await client.close()
        asyncio.run(asyncio.wait_for(read(), timeout=30))
    '''))
    requests = gateway.model.main_requests()
    assert (len(requests), gateway.model.aux_requests(), len(gateway.model.requests)) == (7, [], 7)
    context = json.dumps(requests[1]["messages"])
    assert "[poll: Which native option?; answers: a: A; b: B]" in context
    results = _tool_results(gateway)
    assert results[1]["complete"] is True
    assert [(answer["id"], answer["votes"]) for answer in results[1]["answers"]] == [("a", 0), ("b", 1)]
    assert results[2]["events"][0]["body"] == "[poll: Which native option?; answers: a: A; b: B]"
    assert [(answer["id"], answer["votes"]) for answer in results[4]["answers"]] == [("a", 1), ("b", 1)]
    bot_vote = results[3]["event_id"]
    gateway.model.push(
        ToolCall("tool_search", {"queries": ["native Matrix poll results create"]}),
        _call("matrix_poll_results", {"poll_id": poll}),
        _call("matrix_poll_create", {"question": "Bot native poll?", "answers": ["Yes", "No"], "kind": "undisclosed"}),
        Text("Native poll created"),
    )
    linux_nio_observer.run_python(prefix + f"POLL = {poll!r}\nREPLACEMENT = {external['replacement']!r}\nBOT_VOTE = {bot_vote!r}\n" + dedent('''
        async def close_external():
            client = await open_client()
            try:
                vote = await fetch(client, BOT_VOTE)
                assert vote["type"] == NATIVE + "response", vote
                assert vote["sender"] == BOT_USER
                assert vote["content"] == {
                    NATIVE + "response": {"answers": ["a"]},
                    "m.relates_to": {"rel_type": "m.reference", "event_id": POLL},
                }
                redaction = await client.room_redact(ROOM_ID, REPLACEMENT)
                assert isinstance(redaction, RoomRedactResponse), redaction
                await send(client, NATIVE + "end", {
                    NATIVE + "end": {}, "org.matrix.msc1767.text": "Poll ended",
                    "m.relates_to": {"rel_type": "m.reference", "event_id": POLL},
                })
                await ask(client, "Inspect closed poll and create a poll", "Native poll created")
            finally:
                await client.close()
        asyncio.run(asyncio.wait_for(close_external(), timeout=30))
    '''))
    assert (len(gateway.model.main_requests()), gateway.model.aux_requests(), len(gateway.model.requests)) == (11, [], 11)
    results = _tool_results(gateway)
    restored = results[-2]
    assert (restored["complete"], restored["closed"], restored["voters"]) == (True, True, 2)
    assert [(answer["id"], answer["votes"]) for answer in restored["answers"]] == [("a", 2), ("b", 0)]
    created = results[-1]
    assert created["actor"] == live_room.bot.user_id
    bot_poll = created["poll_id"]
    first_id, second_id = [answer["id"] for answer in created["answers"]]
    gateway.model.push(
        ToolCall("tool_search", {"queries": ["native Matrix poll results vote close"]}),
        _call("matrix_poll_results", {"poll_id": bot_poll}),
        _call("matrix_poll_vote", {"poll_id": bot_poll, "answers": [second_id]}),
        _call("matrix_poll_close", {"poll_id": bot_poll}),
        _call("matrix_poll_results", {"poll_id": bot_poll}),
        Text("Native poll closed"),
    )
    linux_nio_observer.run_python(prefix + f"POLL = {bot_poll!r}\nFIRST = {first_id!r}\nSECOND = {second_id!r}\n" + dedent('''
        async def bot_poll_roundtrip():
            client = await open_client()
            try:
                native = await fetch(client, POLL)
                assert native["type"] == NATIVE + "start", native
                assert native["sender"] == BOT_USER
                assert native["content"]["ai.hermes.poll.requester"] == client.user_id, native
                assert native["content"][NATIVE + "start"] == {
                    "question": {"org.matrix.msc1767.text": "Bot native poll?"},
                    "kind": NATIVE + "undisclosed", "max_selections": 1,
                    "answers": [{"id": FIRST, "org.matrix.msc1767.text": "Yes"},
                                {"id": SECOND, "org.matrix.msc1767.text": "No"}],
                }
                await send(client, NATIVE + "response", {
                    NATIVE + "response": {"answers": [FIRST]},
                    "m.relates_to": {"rel_type": "m.reference", "event_id": POLL},
                })
                await ask(client, "Vote and close your native poll", "Native poll closed")
            finally:
                await client.close()
        asyncio.run(asyncio.wait_for(bot_poll_roundtrip(), timeout=30))
    '''))
    assert (len(gateway.model.main_requests()), gateway.model.aux_requests(), len(gateway.model.requests)) == (17, [], 17)
    results = _tool_results(gateway)
    hidden, vote, receipt, closed = results[-4:]
    for action, sent in (("vote", vote), ("close", receipt)):
        assert sent == {
            "poll_id": bot_poll, "event_id": sent.get("event_id"), "actor": live_room.bot.user_id,
            "action": action, "complete": True, "incomplete_reasons": [],
        }, results[-4:]
        assert isinstance(sent["event_id"], str) and sent["event_id"].startswith("$"), results[-4:]
    assert (hidden["complete"], [answer["votes"] for answer in hidden["answers"]], hidden["voters"],
            hidden["results_visible"]) == (True, [None, None], None, False), results[-4:]
    assert (closed["complete"], closed["closed"], closed["voters"]) == (True, True, 2), results[-4:]
    assert [(answer["id"], answer["votes"]) for answer in closed["answers"]] == [(first_id, 1), (second_id, 1)]
    linux_nio_observer.run_python(prefix + f"POLL = {bot_poll!r}\nEND = {results[-2]['event_id']!r}\n" + dedent('''
        async def inspect_end():
            client = await open_client()
            try:
                native = await fetch(client, END)
                assert native["type"] == NATIVE + "end", native
                assert native["sender"] == BOT_USER
                assert native["content"] == {
                    NATIVE + "end": {}, "org.matrix.msc1767.text": "Poll closed.",
                    "m.relates_to": {"rel_type": "m.reference", "event_id": POLL},
                }
            finally:
                await client.close()
        asyncio.run(asyncio.wait_for(inspect_end(), timeout=15))
    '''))
