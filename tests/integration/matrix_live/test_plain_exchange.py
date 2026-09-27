"""A separate Matrix client exchanges a plain-room message with a real gateway."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from nio import RoomMessageText, RoomSendResponse

from tests.integration.matrix_live.conftest import LinuxNioObserver, LiveGateway, LiveRoom


def test_plain_room_exchange(
    gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
) -> None:
    assert linux_nio_observer.account == live_room.observer
    url = live_room.homeserver
    room_id = live_room.room_id
    alice = live_room.observer

    async def exchange() -> None:
        client = alice.client(url)
        try:
            await client.sync(timeout=0)
            sent = await client.room_send(
                room_id,
                "m.room.message",
                {"msgtype": "m.text", "body": "Hello Hermes [in:plain]"},
            )
            assert isinstance(sent, RoomSendResponse), sent

            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                response = await client.sync(timeout=1000)
                joined = response.rooms.join.get(room_id)
                if not joined:
                    continue
                replies = [
                    (event.sender, event.body)
                    for event in joined.timeline.events
                    if isinstance(event, RoomMessageText) and event.sender != alice.user_id
                ]
                if replies:
                    assert replies == [("@hermes:matrix.test", "Matrix live reply")]
                    requests = gateway.model.main_requests()
                    assert len(requests) == 1
                    assert "Hello Hermes [in:plain]" in json.dumps(requests[0]["messages"])
                    return
            pytest.fail(
                "No Matrix reply after 60 seconds. Gateway logs:\n"
                + gateway.container.get_wrapped_container().logs().decode(errors="replace")[-6000:]
            )
        finally:
            await client.close()

    asyncio.run(exchange())
