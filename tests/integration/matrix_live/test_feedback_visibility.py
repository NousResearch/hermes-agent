"""A separate client observes feedback before and after a real gateway turn."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from threading import Event

import pytest
from nio import (
    ReactionEvent,
    ReceiptEvent,
    RoomMessageText,
    RoomSendResponse,
    TypingNoticeEvent,
)

from tests.fakes.fake_llm_provider import Text
from tests.integration.matrix_live.conftest import (
    LinuxNioObserver,
    LiveGateway,
    LiveRoom,
    MatrixFeedbackSettings,
)


@pytest.fixture(
    params=[
        MatrixFeedbackSettings("immediate", True),
        MatrixFeedbackSettings("after_processing", True),
        MatrixFeedbackSettings("disabled", False),
    ],
    ids=["immediate", "after-processing", "disabled"],
)
def matrix_feedback(request: pytest.FixtureRequest) -> MatrixFeedbackSettings:
    return request.param


@pytest.fixture
def response_gate() -> Iterator[tuple[Event, Event]]:
    started, release = Event(), Event()
    try:
        yield started, release
    finally:
        release.set()


@pytest.fixture
def model_responder(response_gate: tuple[Event, Event]) -> Callable[[dict], Text]:
    started, release = response_gate

    def respond(_request: dict) -> Text:
        started.set()
        assert release.wait(15), "Matrix client did not release the model response"
        return Text("Matrix live reply")

    return respond


@dataclass
class ObservedFeedback:
    receipts: set[str] = field(default_factory=set)
    reactions: list[str] = field(default_factory=list)
    replies: list[tuple[str, str]] = field(default_factory=list)
    typing: bool = False
    redirected: bool = False


def test_feedback_visibility(
    gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    matrix_feedback: MatrixFeedbackSettings,
    response_gate: tuple[Event, Event],
) -> None:
    assert linux_nio_observer.account == live_room.observer
    started, release = response_gate

    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        seen = ObservedFeedback()
        try:
            await client.sync(timeout=0)
            sent = await client.room_send(
                live_room.room_id,
                "m.room.message",
                {
                    "msgtype": "m.text",
                    "body": "Hello Hermes [in:feedback]",
                },
            )
            assert isinstance(sent, RoomSendResponse), sent
            assert await asyncio.to_thread(started.wait, 5), (
                "Gateway did not start the model turn"
            )

            async def observe_until(ready: Callable[[], bool]) -> None:
                async with asyncio.timeout(5):
                    while True:
                        response = await client.sync(timeout=250)
                        joined = response.rooms.join.get(live_room.room_id)
                        if joined is None:
                            continue
                        for event in joined.timeline.events:
                            if event.sender != live_room.bot.user_id:
                                continue
                            if (
                                isinstance(event, ReactionEvent)
                                and event.reacts_to == sent.event_id
                            ):
                                seen.reactions.append(event.key)
                            if isinstance(event, RoomMessageText):
                                if event.body.startswith("↪ Redirected current run"):
                                    seen.redirected = True
                                else:
                                    seen.replies.append((event.sender, event.body))
                        for event in joined.ephemeral:
                            if isinstance(event, ReceiptEvent):
                                seen.receipts.update(
                                    receipt.event_id
                                    for receipt in event.receipts
                                    if receipt.user_id == live_room.bot.user_id
                                    and receipt.receipt_type == "m.read"
                                )
                            if isinstance(event, TypingNoticeEvent):
                                seen.typing = live_room.bot.user_id in event.users
                        if ready():
                            return

            await observe_until(
                lambda: (
                    seen.typing
                    and (not matrix_feedback.reactions or "👀" in seen.reactions)
                    and (
                        matrix_feedback.read_receipts != "immediate"
                        or sent.event_id in seen.receipts
                    )
                )
            )
            assert seen == ObservedFeedback(
                receipts={sent.event_id}
                if matrix_feedback.read_receipts == "immediate"
                else set(),
                reactions=["👀"] if matrix_feedback.reactions else [],
                typing=True,
            )

            correction = await client.room_send(
                live_room.room_id,
                "m.room.message",
                {
                    "msgtype": "m.text",
                    "body": "Please answer this correction [in:feedback-correction]",
                },
            )
            assert isinstance(correction, RoomSendResponse), correction
            await observe_until(
                lambda: (
                    seen.redirected
                    and (
                        matrix_feedback.read_receipts != "immediate"
                        or correction.event_id in seen.receipts
                    )
                )
            )
            assert seen == ObservedFeedback(
                receipts={sent.event_id, correction.event_id}
                if matrix_feedback.read_receipts == "immediate"
                else set(),
                reactions=["👀"] if matrix_feedback.reactions else [],
                typing=True,
                redirected=True,
            )

            release.set()
            await observe_until(
                lambda: (
                    bool(seen.replies)
                    and not seen.typing
                    and (not matrix_feedback.reactions or "✅" in seen.reactions)
                    and (
                        matrix_feedback.read_receipts == "disabled"
                        or correction.event_id in seen.receipts
                    )
                )
            )
            assert seen == ObservedFeedback(
                receipts={
                    "immediate": {sent.event_id, correction.event_id},
                    "after_processing": {correction.event_id},
                    "disabled": set(),
                }[matrix_feedback.read_receipts],
                reactions=["👀", "✅"] if matrix_feedback.reactions else [],
                replies=[(live_room.bot.user_id, "Matrix live reply")],
                redirected=True,
            )
        except TimeoutError:
            pytest.fail(
                "Matrix feedback did not arrive. Gateway logs:\n"
                + gateway.container
                .get_wrapped_container()
                .logs()
                .decode(errors="replace")[-6000:]
            )
        finally:
            release.set()
            await client.close()

    asyncio.run(exchange())
