"""A separate Matrix client exchanges a plain-room message with a real gateway."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from urllib.parse import quote

import pytest
from nio import RoomMessageText, RoomSendResponse

from tests.integration.matrix_live.conftest import LinuxNioObserver, LiveGateway, LiveRoom


def test_plain_room_exchange(
    gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    record_property: Callable[[str, object], None],
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

            deadline = time.monotonic() + 15
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    response = await asyncio.wait_for(client.sync(timeout=250), timeout=remaining)
                except asyncio.TimeoutError:
                    break
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
                    messages = requests[0]["messages"]
                    assert "Hello Hermes [in:plain]" in json.dumps(messages)
                    permalink = (
                        f"https://matrix.to/#/{quote(room_id, safe='!$:@')}/"
                        f"{quote(sent.event_id, safe='!$:@')}?via=matrix.test"
                    )
                    assert f"[Matrix source: {permalink}]" in json.dumps(messages)
                    assert all(
                        permalink not in json.dumps(message)
                        for message in messages if message["role"] == "system"
                    )
                    return
            pytest.fail(
                "No Matrix reply after 15 seconds. Gateway logs:\n"
                + gateway.container.get_wrapped_container().logs().decode(errors="replace")[-6000:]
            )
        finally:
            await client.close()

    started = time.monotonic()
    try:
        asyncio.run(exchange())
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
