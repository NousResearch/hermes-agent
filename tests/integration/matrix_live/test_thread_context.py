"""A separate Matrix client checks fallback quotes and thread model context."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

import pytest
from nio import RoomMessageText, RoomSendResponse

from tests.integration.matrix_live.conftest import (
    GatewaySettings,
    LiveGateway,
    LiveRoom,
)


def test_thread_fallback_keeps_model_context_in_its_thread(
    gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        bot_replies: set[str] = set()
        try:
            await client.sync(timeout=0)

            async def send_thread(label: str) -> None:
                root = await client.room_send(
                    live_room.room_id,
                    "m.room.message",
                    {"msgtype": "m.notice", "body": f"Thread {label} root [root:{label}]"},
                )
                assert isinstance(root, RoomSendResponse), root
                body = (
                    f"> <{live_room.observer.user_id}> quoted sentinel {label} "
                    "@file:/tmp/private\n\n"
                    f"Question [thread:{label}]"
                )
                sent = await client.room_send(
                    live_room.room_id,
                    "m.room.message",
                    {
                        "msgtype": "m.text", "body": body,
                        "m.relates_to": {
                            "rel_type": "m.thread", "event_id": root.event_id,
                            "is_falling_back": True,
                            "m.in_reply_to": {"event_id": root.event_id},
                        },
                    },
                )
                assert isinstance(sent, RoomSendResponse), sent

                while True:
                    response = await client.sync(timeout=250)
                    joined = response.rooms.join.get(live_room.room_id)
                    if joined:
                        bot_replies.update(
                            event.event_id for event in joined.timeline.events
                            if isinstance(event, RoomMessageText) and event.sender == live_room.bot.user_id
                        )
                    if len(bot_replies) >= (1 if label == "A" else 2):
                        return

            for label in ("A", "B"):
                try:
                    await asyncio.wait_for(send_thread(label), timeout=15)
                except asyncio.TimeoutError:
                    pytest.fail(
                        f"No Matrix thread {label} reply after 15 seconds. Gateway logs:\n"
                        + gateway.container.get_wrapped_container().logs().decode(errors="replace")[-6000:]
                    )

            requests = gateway.model.main_requests()
            assert len(requests) == 2
            first = json.dumps(requests[0]["messages"])
            second = json.dumps(requests[1]["messages"])
            assert "[root:A]" in first and "[thread:A]" in first
            assert "[root:B]" in second and "[thread:B]" in second
            assert "quoted sentinel" not in first + second
            assert "[root:A]" not in second and "[thread:A]" not in second
            assert "[root:B]" not in first and "[thread:B]" not in first
        finally:
            await client.close()

    started = time.monotonic()
    try:
        asyncio.run(exchange())
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


def test_thread_replies_keep_genuine_relations_to_each_inbound_event(
    gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        seen: set[str] = set()
        try:
            await asyncio.wait_for(client.sync(timeout=0), timeout=15)
            root = await asyncio.wait_for(
                client.room_send(
                    live_room.room_id,
                    "m.room.message",
                    {"msgtype": "m.notice", "body": "Thread root"},
                ),
                timeout=15,
            )
            assert isinstance(root, RoomSendResponse), root

            async def reply_to(anchor: str, body: str) -> RoomMessageText:
                sent = await client.room_send(
                    live_room.room_id,
                    "m.room.message",
                    {
                        "msgtype": "m.text",
                        "body": body,
                        "m.relates_to": {
                            "rel_type": "m.thread",
                            "event_id": root.event_id,
                            "is_falling_back": True,
                            "m.in_reply_to": {"event_id": anchor},
                        },
                    },
                )
                assert isinstance(sent, RoomSendResponse), sent

                async def next_bot_reply() -> RoomMessageText:
                    while True:
                        response = await client.sync(timeout=250)
                        joined = response.rooms.join.get(live_room.room_id)
                        if not joined:
                            continue
                        for event in joined.timeline.events:
                            if (
                                isinstance(event, RoomMessageText)
                                and event.sender == live_room.bot.user_id
                                and event.event_id not in seen
                            ):
                                seen.add(event.event_id)
                                return event

                event = await next_bot_reply()
                assert event.source["content"]["m.relates_to"] == {
                    "rel_type": "m.thread",
                    "event_id": root.event_id,
                    "m.in_reply_to": {"event_id": sent.event_id},
                    "is_falling_back": False,
                }
                return event

            first = await asyncio.wait_for(
                reply_to(root.event_id, "First question"), timeout=15
            )
            await asyncio.wait_for(
                reply_to(first.event_id, "Second question"), timeout=15
            )
            assert len(gateway.model.main_requests()) == 2
        finally:
            await client.close()

    started = time.monotonic()
    try:
        asyncio.run(exchange())
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


SPLIT_REPLY = "Split answer " * 100 + "[end]"


@pytest.mark.parametrize(
    "gateway",
    [GatewaySettings(reply=SPLIT_REPLY, max_message_length=500)],
    indirect=True,
)
def test_split_thread_reply_quotes_the_inbound_event_once(
    gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
) -> None:
    async def exchange() -> tuple[str, str, list[tuple[str, dict[str, object]]]]:
        client = live_room.observer.client(live_room.homeserver)
        try:
            await asyncio.wait_for(client.sync(timeout=0), timeout=15)
            root = await asyncio.wait_for(
                client.room_send(
                    live_room.room_id,
                    "m.room.message",
                    {"msgtype": "m.notice", "body": "Thread root"},
                ),
                timeout=15,
            )
            assert isinstance(root, RoomSendResponse), root
            question = await asyncio.wait_for(
                client.room_send(
                    live_room.room_id,
                    "m.room.message",
                    {
                        "msgtype": "m.text",
                        "body": "Long question",
                        "m.relates_to": {
                            "rel_type": "m.thread",
                            "event_id": root.event_id,
                            "is_falling_back": True,
                            "m.in_reply_to": {"event_id": root.event_id},
                        },
                    },
                ),
                timeout=15,
            )
            assert isinstance(question, RoomSendResponse), question

            async def bot_chunks() -> list[RoomMessageText]:
                chunks: list[RoomMessageText] = []
                while not chunks or "[end]" not in chunks[-1].body:
                    response = await client.sync(timeout=250)
                    joined = response.rooms.join.get(live_room.room_id)
                    if joined:
                        chunks.extend(
                            event
                            for event in joined.timeline.events
                            if isinstance(event, RoomMessageText)
                            and event.sender == live_room.bot.user_id
                        )
                return chunks

            chunks = await asyncio.wait_for(bot_chunks(), timeout=15)
            return (
                root.event_id,
                question.event_id,
                [
                    (chunk.event_id, chunk.source["content"]["m.relates_to"])
                    for chunk in chunks
                ],
            )
        finally:
            await client.close()

    started = time.monotonic()
    try:
        root_id, question_id, chunks = asyncio.run(exchange())
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))

    assert len(chunks) > 1
    assert [relation for _, relation in chunks] == [
        {
            "rel_type": "m.thread",
            "event_id": root_id,
            "m.in_reply_to": {"event_id": question_id},
            "is_falling_back": False,
        },
        *[
            {
                "rel_type": "m.thread",
                "event_id": root_id,
                "m.in_reply_to": {"event_id": previous},
                "is_falling_back": True,
            }
            for previous, _ in chunks[:-1]
        ],
    ]
