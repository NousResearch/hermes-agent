"""A separate Matrix client resumes a watched reply inside its thread."""

from __future__ import annotations

import asyncio
import json

import pytest
from nio import RoomMessageText, RoomSendResponse

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import LiveRoom
from tests.integration.matrix_live.live_gateway import LiveGateway


def test_requester_reaction_resumes_the_matrix_thread(
    gateway: LiveGateway,
    live_room: LiveRoom,
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        seen: set[str] = set()
        try:
            await client.sync(timeout=0)

            async def send_message(body: str, *, thread_root: str | None = None) -> str:
                content = {"msgtype": "m.text", "body": body}
                if thread_root:
                    content["m.relates_to"] = {
                        "rel_type": "m.thread",
                        "event_id": thread_root,
                        "is_falling_back": True,
                        "m.in_reply_to": {"event_id": thread_root},
                    }
                sent = await client.room_send(
                    live_room.room_id, "m.room.message", content
                )
                assert isinstance(sent, RoomSendResponse), sent
                return sent.event_id

            async def next_reply(body: str) -> RoomMessageText:
                while True:
                    response = await client.sync(timeout=250)
                    joined = response.rooms.join.get(live_room.room_id)
                    if not joined:
                        continue
                    for event in joined.timeline.events:
                        if event.event_id in seen:
                            continue
                        seen.add(event.event_id)
                        if (
                            isinstance(event, RoomMessageText)
                            and event.sender == live_room.bot.user_id
                            and event.body == body
                        ):
                            return event

            await send_message("Prime Matrix session")
            await next_reply("Matrix live reply")
            root = await client.room_send(
                live_room.room_id,
                "m.room.message",
                {"msgtype": "m.notice", "body": "Follow-up thread root"},
            )
            assert isinstance(root, RoomSendResponse), root

            gateway.model.push(
                ToolCall("tool_search", {"queries": ["Matrix reaction follow-up"]}),
                ToolCall(
                    "tool_call",
                    {
                        "calls": [
                            {
                                "name": "matrix_followup",
                                "arguments": {"enabled": True, "emoji": ["👍"]},
                            }
                        ]
                    },
                ),
                Text("Watch this answer"),
                Text("Reaction follow-up"),
            )
            await send_message(
                "Watch the reply in this thread", thread_root=root.event_id
            )
            watched = await next_reply("Watch this answer")
            reacted = await client.room_send(
                live_room.room_id,
                "m.reaction",
                {
                    "m.relates_to": {
                        "rel_type": "m.annotation",
                        "event_id": watched.event_id,
                        "key": "👍",
                    }
                },
            )
            assert isinstance(reacted, RoomSendResponse), reacted
            resumed = await next_reply("Reaction follow-up")

            requests = gateway.model.main_requests()
            assert len(requests) == 5
            assert ("Matrix reaction by " + live_room.observer.user_id) in json.dumps(
                requests[-1]["messages"]
            )
            latest_user = [
                message
                for message in requests[-1]["messages"]
                if message.get("role") == "user"
            ][-1]
            assert "Replying to your previous message" in json.dumps(latest_user)
            assert "Watch this answer" in json.dumps(latest_user)
            assert (
                watched.source["content"]["m.relates_to"]["event_id"] == root.event_id
            )
            assert (
                resumed.source["content"]["m.relates_to"]["event_id"] == root.event_id
            )
        finally:
            await client.close()

    try:
        asyncio.run(asyncio.wait_for(exchange(), timeout=25))
    except asyncio.TimeoutError:
        pytest.fail(
            "Matrix reaction follow-up exceeded 25 seconds. Gateway logs:\n"
            + gateway.container
            .get_wrapped_container()
            .logs()
            .decode(errors="replace")[-6000:]
        )
