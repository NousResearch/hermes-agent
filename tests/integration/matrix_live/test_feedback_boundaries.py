"""Independent Matrix observations of inline and queued completion boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from io import BytesIO
import wave
from threading import Event

import pytest
from nio import (
    AsyncClient,
    ReactionEvent,
    ReceiptEvent,
    RoomMessageText,
    RoomSendResponse,
    UploadResponse,
)

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import (
    LiveGateway,
    LiveRoom,
    MatrixFeedbackSettings,
    GatewayDeliveryProbe,
)


# Bounds only a hang. Every wait below is for an event (a model request, a Matrix event or a file
# that the gateway writes) and returns as soon as it happens; a loaded runner can take tens of
# seconds between them.
_HANG_TIMEOUT = 60.0


@pytest.fixture
def matrix_feedback() -> MatrixFeedbackSettings:
    return MatrixFeedbackSettings("after_processing", True)


@pytest.fixture
def gateway_busy_input_mode(request: pytest.FixtureRequest) -> str:
    return getattr(request, "param", "queue")


@pytest.fixture
def turn_gates() -> Iterator[list[tuple[Event, Event]]]:
    gates = [(Event(), Event()) for _ in range(4)]
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
        assert release.wait(_HANG_TIMEOUT), "Observer did not release the model turn"
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

    async def send_voice(self) -> str:
        audio = BytesIO()
        with wave.open(audio, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 320)
        payload = audio.getvalue()
        uploaded, _encryption = await self.client.upload(
            BytesIO(payload),
            content_type="audio/wav",
            filename="voice.wav",
            filesize=len(payload),
        )
        assert isinstance(uploaded, UploadResponse), uploaded
        result = await self.client.room_send(
            self.room.room_id,
            "m.room.message",
            {
                "msgtype": "m.audio",
                "body": "voice.wav",
                "url": uploaded.content_uri,
                "info": {"mimetype": "audio/wav"},
                "org.matrix.msc3245.voice": {},
                "m.mentions": {"user_ids": [self.room.bot.user_id]},
            },
        )
        assert isinstance(result, RoomSendResponse), result
        return result.event_id

    async def observe_until(self, ready: Callable[[], bool]) -> None:
        async with asyncio.timeout(_HANG_TIMEOUT):
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
            assert await asyncio.to_thread(turn_gates[0][0].wait, _HANG_TIMEOUT)
            await seen.observe_until(lambda: seen.reactions.get(opening) == ["👀"])
            followup = await seen.send("Queued follow-up [in:receipt-followup]")
            await seen.observe_until(
                lambda: any("Queued" in reply for reply in seen.replies)
            )
            assert seen.receipts == set()
            turn_gates[0][1].set()
            assert await asyncio.to_thread(turn_gates[1][0].wait, _HANG_TIMEOUT)
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
            assert await asyncio.to_thread(turn_gates[0][0].wait, _HANG_TIMEOUT)
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
            assert await asyncio.to_thread(turn_gates[0][0].wait, _HANG_TIMEOUT)
            await seen.observe_until(lambda: seen.reactions.get(opening) == ["👀"])
            queued = await seen.send("/queue Follow-up [in:recursive-queue]")
            await seen.observe_until(
                lambda: any(
                    "Queued for the next turn" in reply for reply in seen.replies
                )
            )
            assert (seen.receipts, seen.reactions) == (set(), {opening: ["👀"]})
            turn_gates[0][1].set()
            assert await asyncio.to_thread(turn_gates[1][0].wait, _HANG_TIMEOUT)
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
            assert await asyncio.to_thread(turn_gates[0][0].wait, _HANG_TIMEOUT)
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
            assert await asyncio.to_thread(turn_gates[1][0].wait, _HANG_TIMEOUT)
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
            assert await asyncio.to_thread(turn_gates[2][0].wait, _HANG_TIMEOUT)
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
            assert await asyncio.to_thread(turn_gates[0][0].wait, _HANG_TIMEOUT)
            await seen.observe_until(lambda: seen.reactions.get(opening) == ["👀"])
            consumed = await seen.send("/steer Process this correction [in:consumed-b]")
            await seen.observe_until(
                lambda: (
                    sum("Steer queued into current" in reply for reply in seen.replies)
                    == 1
                )
            )
            turn_gates[0][1].set()
            assert await asyncio.to_thread(turn_gates[1][0].wait, _HANG_TIMEOUT)
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
            assert await asyncio.to_thread(turn_gates[2][0].wait, _HANG_TIMEOUT)
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
            assert await asyncio.to_thread(turn_gates[0][0].wait, _HANG_TIMEOUT)
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
            assert await asyncio.to_thread(turn_gates[1][0].wait, _HANG_TIMEOUT)
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


@pytest.mark.parametrize("gateway_busy_input_mode", ["steer"], indirect=True)
@pytest.mark.parametrize("feedback_path", ["delivery"], indirect=True)
@pytest.mark.parametrize("gateway_delivery_probe", [True], indirect=True)
@pytest.mark.parametrize("voice", [False, True])
def test_steering_during_final_delivery_waits_for_its_own_turn(
    gateway: LiveGateway,
    live_room: LiveRoom,
    turn_gates: list[tuple[Event, Event]],
    gateway_delivery_probe: GatewayDeliveryProbe,
    voice: bool,
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        seen = FeedbackObserver(client, live_room)
        try:
            await client.sync(timeout=0)
            opening = await seen.send("Opening [in:delivery-opening]")
            assert await asyncio.to_thread(turn_gates[0][0].wait, _HANG_TIMEOUT)
            await seen.observe_until(lambda: seen.reactions.get(opening) == ["👀"])
            queued = await seen.send("/queue Queued [in:delivery-queued]")
            await seen.observe_until(
                lambda: any(
                    "Queued for the next turn" in reply for reply in seen.replies
                )
            )
            voice_id = None
            if voice:
                voice_id = await seen.send_voice()
                await seen.observe_until(
                    lambda: (
                        '🎙️ "Voice correction [in:delivery-voice]"' in seen.replies
                        and any(
                            "Steered into current run" in reply
                            for reply in seen.replies
                        )
                    )
                )
            turn_gates[0][1].set()
            async with asyncio.timeout(_HANG_TIMEOUT):
                while not gateway_delivery_probe.started.exists():
                    await asyncio.sleep(0.01)
            late = await seen.send("/steer Later correction [in:delivery-late]")
            await seen.observe_until(
                lambda: any(
                    "/steer queued for the next turn" in reply for reply in seen.replies
                )
            )
            assert (seen.receipts, seen.reactions) == (set(), {opening: ["👀"]})
            gateway_delivery_probe.release.touch()
            expected = [
                opening,
                queued,
                *([voice_id] if voice_id is not None else []),
                late,
            ]
            for index, event_id in enumerate(expected[1:], 1):
                assert await asyncio.to_thread(turn_gates[index][0].wait, _HANG_TIMEOUT)
                await seen.observe_until(
                    lambda: (
                        expected[index - 1] in seen.receipts
                        and seen.reactions.get(event_id) == ["👀"]
                        and f"Completed Matrix turn {index}" in seen.replies
                    )
                )
                assert (seen.receipts, seen.reactions) == (
                    set(expected[:index]),
                    {prior: ["👀", "✅"] for prior in expected[:index]}
                    | {event_id: ["👀"]},
                )
                turn_gates[index][1].set()
            await seen.observe_until(
                lambda: (
                    late in seen.receipts
                    and seen.reactions.get(late) == ["👀", "✅"]
                    and f"Completed Matrix turn {len(expected)}" in seen.replies
                )
            )
            requests = gateway.model.main_requests()
            inputs = [request["messages"][-1]["content"] for request in requests]
            assert len(inputs) == len(expected)
            for text, marker in zip(
                inputs, ["opening", "queued", *(["voice"] if voice else []), "late"]
            ):
                assert text.count(f"[in:delivery-{marker}]") == 1
            assert [
                reply
                for reply in seen.replies
                if reply.startswith("Completed Matrix turn")
            ] == [
                f"Completed Matrix turn {index}"
                for index in range(1, len(expected) + 1)
            ]
            assert (seen.receipts, seen.reactions) == (
                set(expected),
                {event_id: ["👀", "✅"] for event_id in expected},
            )
            if voice:
                assert (
                    len(
                        gateway_delivery_probe.transcriptions.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    )
                    == 1
                )
                assert (
                    seen.replies.count('🎙️ "Voice correction [in:delivery-voice]"') == 1
                )
        except TimeoutError:
            pytest.fail(
                "Delivery-time steering timed out:\n"
                + f"Receipts: {seen.receipts!r}\nReactions: {seen.reactions!r}\nReplies: {seen.replies!r}\n"
                + gateway.container
                .get_wrapped_container()
                .logs()
                .decode(errors="replace")[-6000:]
            )
        finally:
            gateway_delivery_probe.release.touch()
            for _started, release in turn_gates:
                release.set()
            await client.close()

    asyncio.run(exchange())
