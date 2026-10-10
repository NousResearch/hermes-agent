"""A separate Matrix client verifies the model's native reaction actions."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

import pytest
from nio import ReactionEvent, RedactionEvent, RoomMessageText, RoomSendResponse

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import LiveRoom
from tests.integration.matrix_live.live_gateway import LiveGateway


@pytest.mark.parametrize("unreadable", [None, "missing-key", "malformed"])
def test_model_adds_and_removes_a_native_matrix_reaction(
    gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
    unreadable: str | None,
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        seen: set[str] = set()
        try:
            await client.sync(timeout=0)
            if unreadable is not None:
                sent = await client.room_send(
                    live_room.room_id,
                    "m.room.encrypted",
                    {
                        "algorithm": (
                            "m.megolm.v1.aes-sha2"
                            if unreadable == "missing-key"
                            else "example.unsupported"
                        ),
                        "ciphertext": "unreadable historical ciphertext",
                        "sender_key": "unavailable-key",
                        "device_id": live_room.observer.device_id,
                        "session_id": "unavailable-session",
                    },
                )
                assert isinstance(sent, RoomSendResponse), sent

            async def send_and_collect(
                body: str, expected_reply: str, event_type: type | None = None
            ):
                sent = await client.room_send(
                    live_room.room_id,
                    "m.room.message",
                    {"msgtype": "m.text", "body": body},
                )
                assert isinstance(sent, RoomSendResponse), sent

                reply = None
                related = None
                while reply is None or (event_type is not None and related is None):
                    response = await client.sync(timeout=250)
                    joined = response.rooms.join.get(live_room.room_id)
                    if not joined:
                        continue
                    for event in joined.timeline.events:
                        if event.event_id in seen:
                            continue
                        seen.add(event.event_id)
                        if event.sender != live_room.bot.user_id:
                            continue
                        if (
                            isinstance(event, RoomMessageText)
                            and event.body == expected_reply
                        ):
                            reply = event
                        if event_type is not None and isinstance(event, event_type):
                            related = event
                return sent.event_id, related

            await send_and_collect("Prime Matrix session", "Matrix live reply")

            gateway.model.push(
                ToolCall("tool_search", {"queries": ["Matrix reaction emoji"]}),
                ToolCall(
                    "tool_call",
                    {
                        "calls": [
                            {
                                "name": "matrix_reaction",
                                "arguments": {"action": "react", "emoji": "👍"},
                            }
                        ]
                    },
                ),
                Text("Reaction added"),
            )
            target, reaction = await send_and_collect(
                "React to this message",
                "Reaction added",
                ReactionEvent,
            )
            assert (reaction.sender, reaction.reacts_to, reaction.key) == (
                live_room.bot.user_id,
                target,
                "👍",
            )

            gateway.model.push(
                ToolCall("tool_search", {"queries": ["Matrix reaction emoji"]}),
                ToolCall(
                    "tool_call",
                    {
                        "calls": [
                            {
                                "name": "matrix_reaction",
                                "arguments": {
                                    "action": "unreact",
                                    "message_id": target,
                                },
                            }
                        ]
                    },
                ),
                Text("Reaction removed"),
            )
            _, redaction = await send_and_collect(
                "Remove that reaction",
                "Reaction removed",
                RedactionEvent,
            )
            assert (redaction.sender, redaction.redacts) == (
                live_room.bot.user_id,
                reaction.event_id,
            )

            requests = gateway.model.main_requests()
            assert len(requests) == 7
            for search_index, tool_index in ((2, 3), (5, 6)):
                search_result = [
                    message
                    for message in requests[search_index]["messages"]
                    if message["role"] == "tool"
                ]
                assert (
                    "matrix_reaction"
                    in json.loads(search_result[-1]["content"])["tools"]
                )
                tool_result = [
                    message
                    for message in requests[tool_index]["messages"]
                    if message["role"] == "tool"
                ]
                assert json.loads(tool_result[-1]["content"]) == {
                    "success": True,
                    "message_id": target,
                }
        finally:
            await client.close()

    started = time.monotonic()
    try:
        try:
            asyncio.run(asyncio.wait_for(exchange(), timeout=15))
        except asyncio.TimeoutError:
            pytest.fail(
                "Matrix reaction exchange exceeded 15 seconds. Gateway logs:\n"
                + gateway.container
                .get_wrapped_container()
                .logs()
                .decode(errors="replace")[-6000:]
            )
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
