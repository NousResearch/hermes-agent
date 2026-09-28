"""Independent Matrix observations of inline and queued completion boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from threading import Event

import pytest
from nio import (
    AsyncClient,
    ReactionEvent,
    ReceiptEvent,
    RoomMessageText,
    RoomSendResponse,
)

from tests.fakes.fake_llm_provider import Text
from tests.integration.matrix_live.conftest import (
    LiveGateway,
    LiveRoom,
    MatrixFeedbackSettings,
)


@pytest.fixture
def matrix_feedback() -> MatrixFeedbackSettings:
    return MatrixFeedbackSettings("after_processing", True)


@pytest.fixture
def gateway_busy_input_mode() -> str:
    return "queue"


@pytest.fixture
def turn_gates() -> Iterator[list[tuple[Event, Event]]]:
    gates = [(Event(), Event()), (Event(), Event())]
    try:
        yield gates
    finally:
        for _started, release in gates:
            release.set()


@pytest.fixture
def model_responder(turn_gates: list[tuple[Event, Event]]) -> Callable[[dict], Text]:
    calls = 0

    def respond(_request: dict) -> Text:
        nonlocal calls
        index = calls
        calls += 1
        assert index < len(turn_gates), "Unexpected extra model turn"
        started, release = turn_gates[index]
        started.set()
        assert release.wait(15), "Observer did not release the model turn"
        return Text(f"Completed Matrix turn {index + 1}")

    return respond


@dataclass
class FeedbackObserver:
    client: AsyncClient
    room: LiveRoom
    receipts: set[str] = field(default_factory=set)
    reactions: dict[str, list[str]] = field(default_factory=dict)
    replies: list[str] = field(default_factory=list)

    async def send(self, body: str) -> str:
        result = await self.client.room_send(
            self.room.room_id,
            "m.room.message",
            {"msgtype": "m.text", "body": body},
        )
        assert isinstance(result, RoomSendResponse), result
        return result.event_id

    async def observe_until(self, ready: Callable[[], bool]) -> None:
        async with asyncio.timeout(5):
            while True:
                response = await self.client.sync(timeout=250)
                joined = response.rooms.join.get(self.room.room_id)
                if joined is None:
                    continue
                for event in joined.timeline.events:
                    if event.sender != self.room.bot.user_id:
                        continue
                    if isinstance(event, ReactionEvent):
                        self.reactions.setdefault(event.reacts_to, []).append(event.key)
                    if isinstance(event, RoomMessageText):
                        self.replies.append(event.body)
                for event in joined.ephemeral:
                    if isinstance(event, ReceiptEvent):
                        self.receipts.update(
                            receipt.event_id
                            for receipt in event.receipts
                            if receipt.user_id == self.room.bot.user_id
                            and receipt.receipt_type == "m.read"
                        )
                if ready():
                    return


def test_queued_turn_receipt_precedes_cancellation_of_the_followup(
    gateway: LiveGateway,
    live_room: LiveRoom,
    turn_gates: list[tuple[Event, Event]],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        seen = FeedbackObserver(client, live_room)
        try:
            await client.sync(timeout=0)
            opening = await seen.send("First turn [in:receipt-boundary]")
            assert await asyncio.to_thread(turn_gates[0][0].wait, 5)
            await seen.observe_until(lambda: seen.reactions.get(opening) == ["👀"])
            followup = await seen.send("Queued follow-up [in:receipt-followup]")
            await seen.observe_until(
                lambda: any("Queued" in reply for reply in seen.replies)
            )
            assert seen.receipts == set()
            turn_gates[0][1].set()
            assert await asyncio.to_thread(turn_gates[1][0].wait, 5)
            await seen.observe_until(
                lambda: (
                    opening in seen.receipts
                    and seen.reactions.get(opening) == ["👀", "✅"]
                    and seen.reactions.get(followup) == ["👀"]
                    and "Completed Matrix turn 1" in seen.replies
                )
            )
            assert (seen.receipts, seen.reactions) == (
                {opening},
                {opening: ["👀", "✅"], followup: ["👀"]},
            )
            stopped = await seen.send("/stop")
            await seen.observe_until(lambda: stopped in seen.receipts)
            assert (seen.receipts, seen.reactions) == (
                {opening, stopped},
                {opening: ["👀", "✅"], followup: ["👀"]},
            )
        except TimeoutError:
            pytest.fail(
                "Feedback boundary timed out:\n"
                + gateway.container
                .get_wrapped_container()
                .logs()
                .decode(errors="replace")[-6000:]
            )
        finally:
            for _started, release in turn_gates:
                release.set()
            await client.close()

    asyncio.run(exchange())


def test_inline_status_receipt_does_not_change_turn_reactions(
    gateway: LiveGateway,
    live_room: LiveRoom,
    turn_gates: list[tuple[Event, Event]],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        seen = FeedbackObserver(client, live_room)
        try:
            await client.sync(timeout=0)
            opening = await seen.send("Active turn [in:inline-receipt]")
            assert await asyncio.to_thread(turn_gates[0][0].wait, 5)
            await seen.observe_until(lambda: seen.reactions.get(opening) == ["👀"])
            status = await seen.send("/status")
            await seen.observe_until(
                lambda: status in seen.receipts and bool(seen.replies)
            )
            assert (seen.receipts, seen.reactions) == ({status}, {opening: ["👀"]})
            stopped = await seen.send("/stop")
            await seen.observe_until(lambda: stopped in seen.receipts)
            assert (seen.receipts, seen.reactions) == (
                {status, stopped},
                {opening: ["👀"]},
            )
        except TimeoutError:
            pytest.fail(
                "Inline receipt timed out:\n"
                + gateway.container
                .get_wrapped_container()
                .logs()
                .decode(errors="replace")[-6000:]
            )
        finally:
            for _started, release in turn_gates:
                release.set()
            await client.close()

    asyncio.run(exchange())
