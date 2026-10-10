"""A native location from an independent client reaches the live gateway model."""

from __future__ import annotations

import json

from tests.integration.matrix_live.conftest import (
    LinuxNioObserver,
    LiveGateway,
    LiveRoom,
)


def test_native_location_exchange(
    gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
) -> None:
    result = json.loads(
        linux_nio_observer.run_python(f"""
import asyncio
import json
from nio import RoomMessageText, RoomSendResponse
from client import open_encrypted_client

async def exchange():
    client = open_encrypted_client()
    received = asyncio.Event()
    replies = []
    sync_task = None

    async def on_reply(room, event):
        if room.room_id == {live_room.room_id!r} and event.sender == {live_room.bot.user_id!r}:
            replies.append(event.source)
            received.set()

    client.add_event_callback(on_reply, RoomMessageText)
    try:
        await client.sync(timeout=0)
        sent = await client.room_send({live_room.room_id!r}, "m.room.message", {{
            "msgtype": "m.location",
            "body": "Location",
            "geo_uri": "geo:1,2",
            "org.matrix.msc3488.location": {{
                "uri": "geo:40.5694,9.7845,-12.5;u=35",
                "description": "Meeting point [in:location]",
            }},
        }})
        assert isinstance(sent, RoomSendResponse), sent
        sync_task = asyncio.create_task(client.sync_forever(timeout=250))
        await asyncio.wait_for(received.wait(), timeout=15)
        print(json.dumps({{"event_id": sent.event_id, "replies": replies}}))
    finally:
        if sync_task is not None:
            sync_task.cancel()
            await asyncio.gather(sync_task, return_exceptions=True)
        await client.close()

asyncio.run(exchange())
""")
    )

    assert [
        (
            reply["sender"],
            reply["content"]["msgtype"],
            reply["content"]["body"],
            reply["content"]["m.relates_to"]["m.in_reply_to"]["event_id"],
        )
        for reply in result["replies"]
    ] == [(live_room.bot.user_id, "m.text", "Matrix live reply", result["event_id"])]
    requests = gateway.model.main_requests()
    assert len(requests) == 1
    messages = requests[0]["messages"]
    user_messages = [
        message["content"] for message in messages if message["role"] == "user"
    ]
    assert len(user_messages) == 1
    assert (
        "📍 Location: 40.5694, 9.7845; altitude: -12.5 m; uncertainty: 35.0 m "
        "(Meeting point [in:location])"
    ) in user_messages[0]
    context = json.dumps(messages)
    assert live_room.room_id in context
