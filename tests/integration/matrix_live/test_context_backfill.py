"""A separate Matrix client verifies room and thread catch-up in model context."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable

import pytest
from nio import JoinResponse, RoomInviteResponse, RoomMessageText, RoomSendResponse

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


def _conversation_roles(request: dict) -> list[str]:
    return [message["role"] for message in request["messages"] if message["role"] != "system"]


def _last_user_text(request: dict) -> str:
    content = [message for message in request["messages"] if message["role"] == "user"][-1]["content"]
    return content if isinstance(content, str) else "".join(part.get("text", "") for part in content)


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

            await _send(client, live_room.room_id, "Room decision alpha")
            await _send(client, live_room.room_id, "Room decision beta")
            await _send(client, live_room.room_id, f"{live_room.bot.user_id} catch up",
                        mention=live_room.bot.user_id)
            stage = "catch-up reply"
            await _wait_for_final(client, live_room, seen, "ok")

            requests = group_gateway.model.main_requests()
            assert len(requests) == 2
            assert (_conversation_roles(requests[1]), _last_user_text(requests[1])) == (
                ["user", "assistant", "user"],
                "[Recent room messages]\n[alice] Room decision alpha\n[alice] Room decision beta\n\n"
                "[New message]\ncatch up",
            )
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
            seen: set[str] = set()
            await _wait_for_final(client, live_room, seen, "Matrix live reply")

            requests = group_gateway.model.main_requests()
            assert len(requests) == 1
            prompt = json.dumps(requests[0]["messages"])
            assert "Thread A root" in prompt
            assert "Thread A earlier" in prompt
            assert "Thread B earlier" not in prompt
            assert "Thread B root" not in prompt

            await _send(client, live_room.room_id, f"{live_room.bot.user_id} thread follow-up",
                        root=root_a, mention=live_room.bot.user_id)
            await _wait_for_final(client, live_room, seen, "ok")

            requests = group_gateway.model.main_requests()
            assert len(requests) == 2
            assert (_conversation_roles(requests[1]), _last_user_text(requests[1])) == (
                ["user", "assistant", "user"], "[alice] thread follow-up",
            )
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
