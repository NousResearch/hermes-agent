"""A separate Matrix client verifies room and thread catch-up in model context."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from urllib.parse import quote

import aiohttp
import pytest
from nio import JoinResponse, RoomInviteResponse, RoomMessageText, RoomRedactResponse, RoomSendResponse

from tests.integration.matrix_live.conftest import LiveGateway, LiveRoom, MatrixAccount, _register


@pytest.fixture
def group_member(live_room: LiveRoom) -> MatrixAccount:
    async def join() -> MatrixAccount:
        bob = await _register(live_room.homeserver, f"bob-{uuid.uuid4().hex[:8]}")
        alice_client = live_room.observer.client(live_room.homeserver)
        bob_client = bob.client(live_room.homeserver)
        try:
            invited = await alice_client.room_invite(live_room.room_id, bob.user_id)
            assert isinstance(invited, RoomInviteResponse), invited
            joined = await bob_client.join(live_room.room_id)
            assert isinstance(joined, JoinResponse), joined
            return bob
        finally:
            await alice_client.close()
            await bob_client.close()

    return asyncio.run(join())


@pytest.fixture
def group_gateway(group_member: MatrixAccount, gateway: LiveGateway) -> LiveGateway:
    return gateway


async def _send(
    client, room_id: str, body: str, *, root: str | None = None, mention: str | None = None,
) -> str:
    content: dict = {"msgtype": "m.text", "body": body}
    if root is not None:
        content["m.relates_to"] = {
            "rel_type": "m.thread", "event_id": root,
            "m.in_reply_to": {"event_id": root}, "is_falling_back": True,
        }
    if mention is not None:
        content["m.mentions"] = {"user_ids": [mention]}
    sent = await client.room_send(room_id, "m.room.message", content)
    assert isinstance(sent, RoomSendResponse), sent
    return sent.event_id


async def _wait_for_final(client, room: LiveRoom, seen: set[str], expected: str) -> None:
    while True:
        response = await client.sync(timeout=250)
        joined = response.rooms.join.get(room.room_id)
        if joined is None:
            continue
        for event in joined.timeline.events:
            if not isinstance(event, RoomMessageText) or event.sender != room.bot.user_id:
                continue
            if event.event_id in seen or event.body != expected:
                continue
            seen.add(event.event_id)
            return


def test_room_mention_recovers_unaddressed_messages(
    group_gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
) -> None:
    stage = "initial reply"

    async def exchange() -> None:
        nonlocal stage
        client = live_room.observer.client(live_room.homeserver)
        seen: set[str] = set()
        try:
            await client.sync(timeout=0)
            await _send(client, live_room.room_id, f"{live_room.bot.user_id} establish",
                        mention=live_room.bot.user_id)
            await _wait_for_final(client, live_room, seen, "Matrix live reply")

            target = await _send(client, live_room.room_id, "Room decision alpha")
            await _send(client, live_room.room_id, "Room decision beta")
            reacted = await client.room_send(live_room.room_id, "m.reaction", {
                "m.relates_to": {"rel_type": "m.annotation", "event_id": target, "key": "👍"},
            })
            assert isinstance(reacted, RoomSendResponse), reacted
            await _send(client, live_room.room_id, f"{live_room.bot.user_id} catch up",
                        mention=live_room.bot.user_id)
            stage = "catch-up reply"
            await _wait_for_final(client, live_room, seen, "ok")

            requests = group_gateway.model.main_requests()
            assert len(requests) == 2
            prompt = json.dumps(requests[1]["messages"], ensure_ascii=False)
            assert "[Recent room messages]" in prompt
            assert "Room decision alpha" in prompt
            assert "Room decision beta" in prompt
            assert f"[reaction by {live_room.observer.user_id} to {target}] 👍" in prompt
            assert "[New message]" in prompt
        finally:
            await client.close()

    started = time.monotonic()
    try:
        try:
            asyncio.run(asyncio.wait_for(exchange(), timeout=15))
        except asyncio.TimeoutError:
            pytest.fail(
                f"Matrix room catch-up exceeded 15 seconds during {stage}. "
                f"Model requests: {len(group_gateway.model.main_requests())}. Gateway logs:\n"
                + group_gateway.container.get_wrapped_container().logs().decode(errors="replace")[-6000:]
            )
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


def test_thread_mention_recovers_only_its_earlier_messages(
    group_gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        try:
            await client.sync(timeout=0)
            root_a = await _send(client, live_room.room_id, "Thread A root")
            root_b = await _send(client, live_room.room_id, "Thread B root")
            await _send(client, live_room.room_id, "Thread A earlier", root=root_a)
            await _send(client, live_room.room_id, "Thread B earlier", root=root_b)
            await _send(client, live_room.room_id, f"{live_room.bot.user_id} thread question",
                        root=root_a, mention=live_room.bot.user_id)
            await _wait_for_final(client, live_room, set(), "Matrix live reply")

            requests = group_gateway.model.main_requests()
            assert len(requests) == 1
            prompt = json.dumps(requests[0]["messages"])
            assert "Thread A root" in prompt
            assert "Thread A earlier" in prompt
            assert "Thread B earlier" not in prompt
            assert "Thread B root" not in prompt
        finally:
            await client.close()

    started = time.monotonic()
    try:
        try:
            asyncio.run(asyncio.wait_for(exchange(), timeout=15))
        except asyncio.TimeoutError:
            pytest.fail(
                "Matrix thread catch-up exceeded 15 seconds. Gateway logs:\n"
                + group_gateway.container.get_wrapped_container().logs().decode(errors="replace")[-6000:]
            )
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


def test_room_catch_up_shows_edits_and_redactions_to_model(
    group_gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        seen: set[str] = set()
        try:
            await client.sync(timeout=0)
            await _send(client, live_room.room_id, f"{live_room.bot.user_id} establish",
                        mention=live_room.bot.user_id)
            await _wait_for_final(client, live_room, seen, "Matrix live reply")

            edited_target = await _send(client, live_room.room_id, "Draft room decision")
            replacement = await client.room_send(live_room.room_id, "m.room.message", {
                "msgtype": "m.text", "body": "* Final room decision",
                "m.new_content": {"msgtype": "m.text", "body": "Final room decision"},
                "m.relates_to": {"rel_type": "m.replace", "event_id": edited_target},
            })
            assert isinstance(replacement, RoomSendResponse), replacement
            withdrawn_edit = await client.room_send(live_room.room_id, "m.room.message", {
                "msgtype": "m.text", "body": "* Withdrawn edited room decision",
                "m.new_content": {"msgtype": "m.text", "body": "Withdrawn edited room decision"},
                "m.relates_to": {"rel_type": "m.replace", "event_id": edited_target},
            })
            assert isinstance(withdrawn_edit, RoomSendResponse), withdrawn_edit
            edit_redaction = await client.room_redact(live_room.room_id, withdrawn_edit.event_id)
            assert isinstance(edit_redaction, RoomRedactResponse), edit_redaction

            redacted_target = await _send(client, live_room.room_id, "Withdrawn room decision")
            redaction = await client.room_redact(live_room.room_id, redacted_target)
            assert isinstance(redaction, RoomRedactResponse), redaction

            await _send(client, live_room.room_id, f"{live_room.bot.user_id} catch up",
                        mention=live_room.bot.user_id)
            await _wait_for_final(client, live_room, seen, "ok")

            requests = group_gateway.model.main_requests()
            assert len(requests) == 2
            prompt = json.dumps(requests[1]["messages"])
            assert "[Recent room messages]" in prompt
            assert "Final room decision" in prompt
            assert "[redacted]" in prompt
            assert "Draft room decision" not in prompt
            assert "Withdrawn room decision" not in prompt
            assert "Withdrawn edited room decision" not in prompt
        finally:
            await client.close()

    started = time.monotonic()
    try:
        asyncio.run(asyncio.wait_for(exchange(), timeout=20))
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


def test_redacted_child_is_removed_from_thread_relations(live_room: LiveRoom) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        try:
            root = await _send(client, live_room.room_id, "Thread root")
            child = await _send(client, live_room.room_id, "Thread child", root=root)
            base = f"{live_room.homeserver}/_matrix/client/v1/rooms/{quote(live_room.room_id, safe='')}"
            relations_url = f"{base}/relations/{quote(root, safe='')}/m.thread"
            event_url = f"{live_room.homeserver}/_matrix/client/v3/rooms/{quote(live_room.room_id, safe='')}/event/{quote(child, safe='')}"
            headers = {"Authorization": f"Bearer {live_room.observer.access_token}"}
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(relations_url) as response:
                    assert response.status == 200
                    before = await response.json()

                redaction = await client.room_redact(live_room.room_id, child)
                assert isinstance(redaction, RoomRedactResponse), redaction

                async with session.get(relations_url) as response:
                    assert response.status == 200
                    after = await response.json()
                async with session.get(event_url) as response:
                    assert response.status == 200
                    event = await response.json()

            assert [item["event_id"] for item in before["chunk"]] == [child]
            assert after["chunk"] == []
            assert event["content"] == {}
            assert event["unsigned"]["redacted_because"]["redacts"] == child
        finally:
            await client.close()

    asyncio.run(asyncio.wait_for(exchange(), timeout=15))
