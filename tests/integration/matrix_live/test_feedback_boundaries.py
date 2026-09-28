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

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import (
    LiveGateway,
    LiveRoom,
    MatrixFeedbackSettings,
)


@pytest.fixture
def matrix_feedback() -> MatrixFeedbackSettings:
    return MatrixFeedbackSettings("after_processing", True)


@pytest.fixture
def gateway_busy_input_mode(request: pytest.FixtureRequest) -> str:
    return getattr(request, "param", "queue")


@pytest.fixture
def turn_gates() -> Iterator[list[tuple[Event, Event]]]:
    gates = [(Event(), Event()) for _ in range(3)]
    try:
        yield gates
    finally:
        for _started, release in gates:
            release.set()


@pytest.fixture
def feedback_path(request: pytest.FixtureRequest) -> str:
    return getattr(request, "param", "ordinary")


@pytest.fixture
def model_responder(
    turn_gates: list[tuple[Event, Event]],
    gateway_busy_input_mode: str,
    feedback_path: str,
) -> Callable[[dict], Text | ToolCall]:
    calls = 0

    def respond(_request: dict) -> Text | ToolCall:
        nonlocal calls
        index = calls
        calls += 1
        if feedback_path == "ordinary" and gateway_busy_input_mode == "interrupt":
            index = min(index, 1)
        assert index < len(turn_gates), "Unexpected extra model turn"
        started, release = turn_gates[index]
        started.set()
        assert release.wait(15), "Observer did not release the model turn"
        if index == 0 and feedback_path == "consumed":
            return ToolCall("read_file", {"path": "/etc/hostname"})
        if index == 0 and feedback_path == "approval":
            return ToolCall(
                "terminal",
                {
                    "command": "rm -rf /tmp/hermes-receipt-approval-target",
                    "timeout": 10,
                },
            )
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


@pytest.mark.parametrize("gateway_busy_input_mode", ["interrupt"], indirect=True)
def test_queue_and_recursive_correction_receipts_wait_for_processing(
    gateway: LiveGateway,
    live_room: LiveRoom,
    turn_gates: list[tuple[Event, Event]],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        seen = FeedbackObserver(client, live_room)
        try:
            await client.sync(timeout=0)
            opening = await seen.send("First turn [in:recursive-opening]")
            assert await asyncio.to_thread(turn_gates[0][0].wait, 5)
            await seen.observe_until(lambda: seen.reactions.get(opening) == ["👀"])
            queued = await seen.send("/queue Follow-up [in:recursive-queue]")
            await seen.observe_until(
                lambda: any(
                    "Queued for the next turn" in reply for reply in seen.replies
                )
            )
            assert (seen.receipts, seen.reactions) == (set(), {opening: ["👀"]})
            turn_gates[0][1].set()
            assert await asyncio.to_thread(turn_gates[1][0].wait, 5)
            await seen.observe_until(
                lambda: (
                    opening in seen.receipts and seen.reactions.get(queued) == ["👀"]
                )
            )
            assert (seen.receipts, seen.reactions) == (
                {opening},
                {opening: ["👀", "✅"], queued: ["👀"]},
            )
            correction = await seen.send(
                "Please answer this correction [in:recursive-correction]"
            )
            await seen.observe_until(
                lambda: any("Redirected current run" in reply for reply in seen.replies)
            )
            assert (seen.receipts, seen.reactions) == (
                {opening},
                {opening: ["👀", "✅"], queued: ["👀"]},
            )
            turn_gates[1][1].set()
            await seen.observe_until(
                lambda: (
                    correction in seen.receipts
                    and seen.reactions.get(queued) == ["👀", "✅"]
                    and "Completed Matrix turn 2" in seen.replies
                )
            )
            assert (seen.receipts, seen.reactions) == (
                {opening, correction},
                {opening: ["👀", "✅"], queued: ["👀", "✅"]},
            )
            assert "Completed Matrix turn 2" in seen.replies
        except TimeoutError:
            pytest.fail(
                "Recursive correction receipt timed out:\n"
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


@pytest.mark.parametrize("feedback_path", ["fifo"], indirect=True)
def test_fifo_precedes_late_steering_without_acknowledging_it(
    gateway: LiveGateway,
    live_room: LiveRoom,
    turn_gates: list[tuple[Event, Event]],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        seen = FeedbackObserver(client, live_room)
        try:
            await client.sync(timeout=0)
            opening = await seen.send("Opening [in:fifo-opening]")
            assert await asyncio.to_thread(turn_gates[0][0].wait, 5)
            await seen.observe_until(lambda: seen.reactions.get(opening) == ["👀"])
            queued = await seen.send("/queue Queued [in:fifo-queued]")
            await seen.observe_until(
                lambda: any(
                    "Queued for the next turn" in reply for reply in seen.replies
                )
            )
            late = await seen.send("/steer Late correction [in:fifo-late]")
            await seen.observe_until(
                lambda: any(
                    "Steer queued into current" in reply for reply in seen.replies
                )
            )
            assert seen.receipts == set()
            turn_gates[0][1].set()
            assert await asyncio.to_thread(turn_gates[1][0].wait, 5)
            await seen.observe_until(
                lambda: (
                    opening in seen.receipts and seen.reactions.get(queued) == ["👀"]
                )
            )
            assert (seen.receipts, seen.reactions) == (
                {opening},
                {opening: ["👀", "✅"], queued: ["👀"]},
            )
            turn_gates[1][1].set()
            assert await asyncio.to_thread(turn_gates[2][0].wait, 5)
            await seen.observe_until(
                lambda: queued in seen.receipts and seen.reactions.get(late) == ["👀"]
            )
            assert (seen.receipts, seen.reactions) == (
                {opening, queued},
                {opening: ["👀", "✅"], queued: ["👀", "✅"], late: ["👀"]},
            )
            turn_gates[2][1].set()
            await seen.observe_until(
                lambda: (
                    late in seen.receipts and seen.reactions.get(late) == ["👀", "✅"]
                )
            )
            assert seen.receipts == {opening, queued, late}
            requests = gateway.model.main_requests()
            assert len(requests) == 3
            assert "[in:fifo-queued]" in str(requests[1]["messages"][-1])
            assert "[in:fifo-late]" in str(requests[2]["messages"][-1])
        except TimeoutError:
            pytest.fail(
                "FIFO steering timed out:\n"
                + f"Receipts: {seen.receipts!r}\nReactions: {seen.reactions!r}\nReplies: {seen.replies!r}\n"
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


@pytest.mark.parametrize("feedback_path", ["consumed"], indirect=True)
@pytest.mark.parametrize("completion", ["success", "cancelled"])
def test_consumed_steering_receipt_precedes_pending_steering(
    gateway: LiveGateway,
    live_room: LiveRoom,
    turn_gates: list[tuple[Event, Event]],
    completion: str,
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        seen = FeedbackObserver(client, live_room)
        try:
            await client.sync(timeout=0)
            opening = await seen.send("Opening [in:consumed-opening]")
            assert await asyncio.to_thread(turn_gates[0][0].wait, 5)
            await seen.observe_until(lambda: seen.reactions.get(opening) == ["👀"])
            consumed = await seen.send("/steer Process this correction [in:consumed-b]")
            await seen.observe_until(
                lambda: (
                    sum("Steer queued into current" in reply for reply in seen.replies)
                    == 1
                )
            )
            turn_gates[0][1].set()
            assert await asyncio.to_thread(turn_gates[1][0].wait, 5)
            messages = gateway.model.main_requests()[1]["messages"]
            assert any(
                message.get("role") == "user"
                and "[in:consumed-b]" in str(message.get("content"))
                for message in messages
            )
            pending = await seen.send("/steer Defer this correction [in:pending-c]")
            await seen.observe_until(
                lambda: (
                    sum("Steer queued into current" in reply for reply in seen.replies)
                    == 2
                )
            )
            assert seen.receipts == set()
            turn_gates[1][1].set()
            assert await asyncio.to_thread(turn_gates[2][0].wait, 5)
            await seen.observe_until(
                lambda: (
                    consumed in seen.receipts and seen.reactions.get(pending) == ["👀"]
                )
            )
            assert (seen.receipts, seen.reactions) == (
                {consumed},
                {opening: ["👀", "✅"], pending: ["👀"]},
            )
            if completion == "cancelled":
                stopped = await seen.send("/stop")
                await seen.observe_until(lambda: stopped in seen.receipts)
                assert (seen.receipts, seen.reactions) == (
                    {consumed, stopped},
                    {opening: ["👀", "✅"], pending: ["👀"]},
                )
                return
            turn_gates[2][1].set()
            await seen.observe_until(
                lambda: (
                    pending in seen.receipts
                    and seen.reactions.get(pending) == ["👀", "✅"]
                )
            )
            assert seen.receipts == {consumed, pending}
        except TimeoutError:
            pytest.fail(
                "Consumed steering timed out:\n"
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


@pytest.mark.parametrize("feedback_path", ["approval"], indirect=True)
@pytest.mark.parametrize("answer", ["yes", "no"])
def test_plaintext_approval_receipt_precedes_active_turn_completion(
    gateway: LiveGateway,
    live_room: LiveRoom,
    turn_gates: list[tuple[Event, Event]],
    answer: str,
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        seen = FeedbackObserver(client, live_room)
        try:
            await client.sync(timeout=0)
            opening = await seen.send("Approval [in:plaintext-approval]")
            assert await asyncio.to_thread(turn_gates[0][0].wait, 5)
            await seen.observe_until(lambda: seen.reactions.get(opening) == ["👀"])
            turn_gates[0][1].set()
            await seen.observe_until(
                lambda: any(
                    "needs your OK" in reply
                    and "rm -rf /tmp/hermes-receipt-approval-target" in reply
                    for reply in seen.replies
                )
            )
            approval = await seen.send(answer)
            assert await asyncio.to_thread(turn_gates[1][0].wait, 5)
            await seen.observe_until(lambda: approval in seen.receipts)
            assert (
                seen.receipts,
                seen.reactions.get(opening),
                seen.reactions.get(approval),
            ) == (
                {approval},
                ["👀"],
                None,
            )
            tool_results = [
                message["content"]
                for message in gateway.model.main_requests()[1]["messages"]
                if message.get("role") == "tool"
            ]
            assert tool_results
            assert any(
                "denied" in result.lower() or "blocked" in result.lower()
                for result in tool_results
            ) == (answer == "no")
            turn_gates[1][1].set()
            await seen.observe_until(
                lambda: (
                    "Completed Matrix turn 2" in seen.replies
                    and seen.reactions.get(opening) == ["👀", "✅"]
                )
            )
            assert (
                seen.receipts,
                seen.reactions.get(opening),
                seen.reactions.get(approval),
            ) == (
                {approval},
                ["👀", "✅"],
                None,
            )
        except TimeoutError:
            pytest.fail(
                "Approval receipt timed out:\n"
                + f"Receipts: {seen.receipts!r}\nReactions: {seen.reactions!r}\nReplies: {seen.replies!r}\n"
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
