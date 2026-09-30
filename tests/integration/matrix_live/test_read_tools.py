"""A separate Matrix client verifies model-visible reads through the live gateway."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

from nio import RoomMessageText, RoomSendResponse

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import LiveGateway, LiveRoom


def test_model_reads_an_event_and_the_room_from_its_live_matrix_session(
    gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        try:
            await asyncio.wait_for(client.sync(timeout=0), timeout=60)

            async def send_and_wait(body: str, reply: str) -> str:
                sent = await client.room_send(
                    live_room.room_id,
                    "m.room.message",
                    {"msgtype": "m.text", "body": body},
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
                send_and_wait("Read target [history:blue]", "Matrix live reply"),
                timeout=60,
            )
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
                ToolCall(
                    "tool_call",
                    {
                        "calls": [
                            {
                                "name": "matrix_read",
                                "arguments": {"kind": "room", "limit": 10},
                            }
                        ]
                    },
                ),
                Text("Read complete"),
            )
            await asyncio.wait_for(
                send_and_wait("Read the earlier Matrix event", "Read complete"),
                timeout=60,
            )

            requests = gateway.model.main_requests()
            assert len(requests) == 5
            search_messages = [
                message
                for message in requests[2]["messages"]
                if message["role"] == "tool"
            ]
            assert len(search_messages) == 1
            assert "matrix_read" in json.loads(search_messages[0]["content"])["tools"]
            tool_messages = [
                message
                for message in requests[4]["messages"]
                if message["role"] == "tool"
            ]
            assert len(tool_messages) == 3
            room = json.loads(tool_messages[2]["content"])
            timestamps = [event["timestamp"] for event in room["events"]]
            observer_bodies = [
                event["body"]
                for event in room["events"]
                if event["sender"] == live_room.observer.user_id
            ]
            assert (
                observer_bodies,
                timestamps == sorted(timestamps),
                room["errors"],
                room["skipped"],
            ) == (
                ["Read target [history:blue]", "Read the earlier Matrix event"],
                True,
                [],
                0,
            )
            result = json.loads(tool_messages[1]["content"])
            assert isinstance(result["events"][0]["timestamp"], int)
            assert {
                **result,
                "events": [{**result["events"][0], "timestamp": None}],
            } == {
                "events": [
                    {
                        "event_id": target,
                        "sender": live_room.observer.user_id,
                        "body": "Read target [history:blue]",
                        "msgtype": "m.text",
                        "thread_id": None,
                        "timestamp": None,
                        "sender_authorized": True,
                    }
                ],
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
