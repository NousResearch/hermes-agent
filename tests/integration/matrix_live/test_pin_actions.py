"""A separate Matrix client observes pin changes and server permission errors."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

import pytest
from nio import RoomGetStateEventResponse, RoomMessageText, RoomPutStateResponse, RoomSendResponse

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import LiveGateway, LiveRoom


@pytest.fixture
def gateway_extra_config() -> str:
    return "platform_toolsets:\n  matrix: [hermes-matrix, matrix_admin]\n"


def test_model_pins_and_unpins_with_homeserver_permission_checks(
    gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        bot_client = live_room.bot.client(live_room.homeserver)
        try:
            await client.sync(timeout=0)

            async def request(body: str, answer: str) -> RoomMessageText:
                sent = await client.room_send(
                    live_room.room_id, "m.room.message", {"msgtype": "m.text", "body": body},
                )
                assert isinstance(sent, RoomSendResponse), sent
                while True:
                    response = await client.sync(timeout=250)
                    joined = response.rooms.join.get(live_room.room_id)
                    if joined:
                        for event in joined.timeline.events:
                            if (isinstance(event, RoomMessageText)
                                    and event.sender == live_room.bot.user_id and event.body == answer):
                                return event

            target = await asyncio.wait_for(request("Start pin test", "Matrix live reply"), timeout=10)
            baseline = await bot_client.room_send(
                live_room.room_id, "m.room.message", {"msgtype": "m.text", "body": "Existing pin"},
            )
            assert isinstance(baseline, RoomSendResponse), baseline
            response = await client.room_put_state(
                live_room.room_id, "m.room.pinned_events", {"pinned": [baseline.event_id]},
            )
            assert isinstance(response, RoomPutStateResponse), response

            async def action(operation: str, answer: str) -> dict:
                gateway.model.push(
                    ToolCall("tool_search", {"queries": ["Matrix pin message"]}),
                    ToolCall("tool_call", {"calls": [{"name": "matrix_pin", "arguments": {
                        "action": operation, "event_id": target.event_id,
                    }}]}),
                    Text(answer),
                )
                await asyncio.wait_for(request(f"{operation} the first reply", answer), timeout=10)
                results = [json.loads(message["content"])
                           for message in gateway.model.main_requests()[-1]["messages"]
                           if message["role"] == "tool"]
                return results[-1]

            denied = await action("pin", "Pin denied")
            assert denied["errcode"] == "M_FORBIDDEN"
            assert denied["error"] == "Matrix pin update was rejected"

            power = await client.room_put_state(live_room.room_id, "m.room.power_levels", {
                "users": {live_room.observer.user_id: 100, live_room.bot.user_id: 100},
            })
            assert isinstance(power, RoomPutStateResponse), power
            added = await action("pin", "Pin complete")
            removed = await action("unpin", "Unpin complete")
            assert added["pinned"] == [baseline.event_id, target.event_id]
            assert removed["pinned"] == [baseline.event_id]

            demoted = await client.room_put_state(live_room.room_id, "m.room.power_levels", {
                "users": {live_room.bot.user_id: 100},
            })
            assert isinstance(demoted, RoomPutStateResponse), demoted
            refused = await action("pin", "Pin refused")
            assert refused == {
                "error": "Matrix requester lacks permission to change pins", "required": 50, "level": 0,
            }

            state = await client.room_get_state_event(live_room.room_id, "m.room.pinned_events")
            assert isinstance(state, RoomGetStateEventResponse), state
            assert state.content == {"pinned": [baseline.event_id]}
        finally:
            await client.close()
            await bot_client.close()

    started = time.monotonic()
    try:
        asyncio.run(asyncio.wait_for(exchange(), timeout=30))
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
