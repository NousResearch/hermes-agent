"""Separate-client helpers for Matrix catch-up and context lifetime cases."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from nio import JoinResponse, RoomInviteResponse, RoomMessageText, RoomSendResponse

from tests.integration.matrix_live.conftest import (
    LiveGateway,
    LiveRoom,
    MatrixAccount,
    _register,
)


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
    client,
    room_id: str,
    body: str,
    *,
    root: str | None = None,
    mention: str | None = None,
    reply: str | None = None,
) -> str:
    content: dict = {"msgtype": "m.text", "body": body}
    if root is not None:
        content["m.relates_to"] = {
            "rel_type": "m.thread",
            "event_id": root,
            "m.in_reply_to": {"event_id": root},
            "is_falling_back": True,
        }
    if reply is not None:
        content["m.relates_to"] = {"m.in_reply_to": {"event_id": reply}}
    if mention is not None:
        content["m.mentions"] = {"user_ids": [mention]}
    sent = await client.room_send(room_id, "m.room.message", content)
    assert isinstance(sent, RoomSendResponse), sent
    return sent.event_id


async def _wait_for_final(
    client,
    room: LiveRoom,
    seen: set[str],
    expected: str,
    *,
    count: int = 1,
) -> None:
    observed = 0
    while True:
        response = await client.sync(timeout=250)
        joined = response.rooms.join.get(room.room_id)
        if joined is None:
            continue
        for event in joined.timeline.events:
            if (
                not isinstance(event, RoomMessageText)
                or event.sender != room.bot.user_id
            ):
                continue
            if event.event_id in seen or event.body != expected:
                continue
            seen.add(event.event_id)
            observed += 1
            if observed == count:
                return
