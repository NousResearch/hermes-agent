from __future__ import annotations
import asyncio

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent import realtime_voice_registry
from agent.realtime_voice import (
    HeardAudioBoundary,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeSession,
    RealtimeVoiceProvider,
)
from agent.realtime_voice_coordinator import RealtimeVoiceCoordinator


class FakeSession(RealtimeSession):
    def __init__(self, events: list[RealtimeEvent]) -> None:
        self._events = events
        self.audio: list[bytes] = []
        self.tool_results: list[tuple[str, str]] = []
        self.context: list[tuple[str, str]] = []
        self.cancelled = False
        self.cancellation_boundaries: list[HeardAudioBoundary | None] = []
        self.cancellation_operations: list[str] = []
        self.closed = False

    async def send_audio(self, pcm: bytes) -> None:
        self.audio.append(pcm)

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        for event in self._events:
            yield event

    async def submit_tool_result(self, call_id: str, output: str) -> None:
        self.tool_results.append((call_id, output))

    async def add_context(self, item_id: str, text: str) -> None:
        self.context.append((item_id, text))

    async def cancel_response(self) -> None:
        self.cancelled = True
        self.cancellation_operations.append("cancel")

    async def close(self) -> None:
        self.closed = True


class FakeProvider(RealtimeVoiceProvider):
    def __init__(self, name: str, session: FakeSession) -> None:
        self._name = name
        self.session = session
        self.opened_with: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self._name

    async def open_session(self, *, instructions, tools, voice=None):
        self.opened_with = {"instructions": instructions, "tools": tools, "voice": voice}
        return self.session


class BoundarySession(FakeSession):
    async def truncate_response(self, boundary: HeardAudioBoundary) -> None:
        self.cancellation_boundaries.append(boundary)
        self.cancellation_operations.append("truncate")


class LegacyCancelSession(FakeSession):
    async def cancel_response(self) -> None:
        self.cancelled = True
        self.cancellation_boundaries.append(None)


@pytest.fixture(autouse=True)
def reset_registry():
    realtime_voice_registry._reset_for_tests()
    yield
    realtime_voice_registry._reset_for_tests()

def async_dispatch(result: str):
    async def dispatch(_name: str, _arguments: dict[str, Any]) -> str:
        return result

    return dispatch


def test_registry_is_profile_scoped_and_accepts_two_providers():
    alpha = FakeProvider("alpha", FakeSession([]))
    beta = FakeProvider("beta", FakeSession([]))
    realtime_voice_registry.register_provider(alpha, scope="home-a")
    realtime_voice_registry.register_provider(beta, scope="home-a")

    assert realtime_voice_registry.get_provider(" ALPHA ", scope="home-a") is alpha
    assert [provider.name for provider in realtime_voice_registry.list_providers(scope="home-a")] == [
        "alpha",
        "beta",
    ]
    assert realtime_voice_registry.get_provider("alpha", scope="home-b") is None


def test_registry_rejects_invalid_provider_and_empty_name():
    with pytest.raises(TypeError, match="RealtimeVoiceProvider"):
        realtime_voice_registry.register_provider(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        realtime_voice_registry.register_provider(FakeProvider(" ", FakeSession([])))


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["grok-plugin", "second-provider"])
async def test_coordinator_keeps_tool_dispatch_in_hermes(provider_name: str):
    session = FakeSession(
        [
            RealtimeEvent.audio(b"reply-pcm"),
            RealtimeEvent.transcript("hello", final=True, role="user"),
            RealtimeEvent.tool_call("call-1", "terminal", {"command": "pwd"}),
        ]
    )
    provider = FakeProvider(provider_name, session)
    dispatched: list[tuple[str, dict[str, Any]]] = []

    async def dispatch(name: str, arguments: dict[str, Any]) -> str:
        dispatched.append((name, arguments))
        return "/safe/workspace"

    coordinator = RealtimeVoiceCoordinator(provider, dispatch_tool=dispatch)
    await coordinator.open(instructions="Hermes owns tools", tools=[{"name": "terminal"}], voice="eve")
    await coordinator.send_audio(b"user-pcm")
    await coordinator.add_context("progress-1", "checked repository")
    observed = [event async for event in coordinator.events()]
    await asyncio.gather(*coordinator._tool_tasks.values())
    await coordinator.close()

    assert provider.opened_with == {
        "instructions": "Hermes owns tools",
        "tools": [{"name": "terminal"}],
        "voice": "eve",
    }
    assert session.audio == [b"user-pcm"]
    assert session.context == [("progress-1", "checked repository")]
    assert dispatched == [("terminal", {"command": "pwd"})]
    assert session.tool_results == [("call-1", "/safe/workspace")]
    assert [event.type for event in observed] == [
        RealtimeEventType.AUDIO,
        RealtimeEventType.TRANSCRIPT,
        RealtimeEventType.TOOL_CALL,
    ]
    assert observed[1].role == "user"
    assert session.closed is True


@pytest.mark.asyncio
async def test_coordinator_cancels_at_the_latest_heard_output_boundary_once():
    session = BoundarySession([RealtimeEvent.audio(b"reply-pcm", item_id="item-1")])
    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", session), dispatch_tool=async_dispatch("ok")
    )
    await coordinator.open(instructions="", tools=[])
    [output] = [event async for event in coordinator.events()]

    assert output.item_id == "item-1"
    assert coordinator.report_audio_heard(output, audio_end_ms=240) is True
    await coordinator.cancel_response()

    assert session.cancellation_boundaries == [
        HeardAudioBoundary(item_id="item-1", audio_end_ms=240)
    ]
    assert session.cancellation_operations == ["cancel", "truncate"]


@pytest.mark.asyncio
async def test_coordinator_rejects_foreign_stale_and_regressing_heard_boundaries():
    first = RealtimeEvent.audio(b"first", item_id="item-1")
    second = RealtimeEvent.audio(b"second", item_id="item-2")
    session = FakeSession([first, second])
    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", session), dispatch_tool=async_dispatch("ok")
    )
    await coordinator.open(instructions="", tools=[])
    observed = [event async for event in coordinator.events()]

    assert coordinator.report_audio_heard(first, audio_end_ms=100) is False
    assert coordinator.report_audio_heard(
        RealtimeEvent.audio(b"foreign", item_id="item-2"), audio_end_ms=100
    ) is False
    assert coordinator.report_audio_heard(observed[1], audio_end_ms=100) is True
    assert coordinator.report_audio_heard(observed[1], audio_end_ms=90) is False


@pytest.mark.asyncio
async def test_coordinator_rejects_foreign_event_with_the_same_content():
    emitted = RealtimeEvent.audio(b"emitted", item_id="item-1")
    foreign = RealtimeEvent(
        type=RealtimeEventType.AUDIO,
        audio_bytes=b"emitted",
        item_id="item-1",
    )
    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", FakeSession([emitted])),
        dispatch_tool=async_dispatch("ok"),
    )
    await coordinator.open(instructions="", tools=[])
    [observed] = [event async for event in coordinator.events()]

    assert coordinator.report_audio_heard(observed, audio_end_ms=100) is True
    assert coordinator.report_audio_heard(foreign, audio_end_ms=120) is False


@pytest.mark.asyncio
async def test_coordinator_retains_only_latest_audio_event_identity():
    events = [
        RealtimeEvent.audio(b"x" * 4_800, item_id="item-1")
        for _ in range(10_000)
    ]
    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", FakeSession(events)),
        dispatch_tool=async_dispatch("ok"),
    )
    await coordinator.open(instructions="", tools=[])
    observed = [event async for event in coordinator.events()]

    assert coordinator._current_audio_event is observed[-1]
    assert coordinator.report_audio_heard(observed[-1], audio_end_ms=100) is True
    assert coordinator.report_audio_heard(observed[-2], audio_end_ms=100) is False


@pytest.mark.asyncio
async def test_zero_heard_and_legacy_cancel_remain_compatible_across_reconnect():
    old_output = RealtimeEvent.audio(b"old", item_id="reused-item")
    session = LegacyCancelSession([old_output])
    provider = FakeProvider("legacy", session)
    coordinator = RealtimeVoiceCoordinator(
        provider, dispatch_tool=async_dispatch("ok")
    )
    await coordinator.open(instructions="", tools=[])
    [observed_old] = [event async for event in coordinator.events()]
    await coordinator.close()

    replacement = LegacyCancelSession([])
    provider.session = replacement
    await coordinator.open(instructions="", tools=[])
    assert coordinator.report_audio_heard(observed_old, audio_end_ms=0) is False
    await coordinator.cancel_response()

    assert replacement.cancellation_boundaries == [None]


@pytest.mark.asyncio
async def test_zero_heard_boundary_cancels_before_truncating_to_start():
    session = BoundarySession([RealtimeEvent.audio(b"reply", item_id="item-zero")])
    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", session), dispatch_tool=async_dispatch("ok")
    )
    await coordinator.open(instructions="", tools=[])
    [output] = [event async for event in coordinator.events()]
    assert coordinator.report_audio_heard(output, audio_end_ms=0) is True

    await coordinator.cancel_response()

    assert session.cancellation_boundaries == [HeardAudioBoundary("item-zero", 0)]
    assert session.cancellation_operations == ["cancel", "truncate"]


@pytest.mark.asyncio
async def test_coordinator_logs_dispatch_failures_with_tool_context(
    caplog: pytest.LogCaptureFixture,
):
    session = FakeSession([RealtimeEvent.tool_call("call-2", "browser", {})])

    async def dispatch(_name: str, _arguments: dict[str, Any]) -> str:
        raise RuntimeError("approval denied")

    coordinator = RealtimeVoiceCoordinator(FakeProvider("fake", session), dispatch_tool=dispatch)
    await coordinator.open(instructions="", tools=[])
    events = [event async for event in coordinator.events()]
    await asyncio.gather(*coordinator._tool_tasks.values())

    assert len(events) == 1
    assert session.tool_results == [("call-2", "Error: approval denied")]
    [record] = [
        record
        for record in caplog.records
        if record.getMessage() == "Realtime voice tool dispatch failed"
    ]
    assert record.__dict__["tool_name"] == "browser"
    assert record.__dict__["call_id"] == "call-2"
    assert record.exc_info is not None


@pytest.mark.asyncio
async def test_coordinator_requires_open_session_and_closes_idempotently():
    session = FakeSession([])
    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", session), dispatch_tool=async_dispatch("ok")
    )

    with pytest.raises(RuntimeError, match="not open"):
        await coordinator.send_audio(b"pcm")
    await coordinator.close()
    await coordinator.open(instructions="", tools=[])
    await coordinator.close()
    await coordinator.close()
    assert session.closed is True


@pytest.mark.asyncio
async def test_tool_dispatch_does_not_block_later_realtime_events():
    release = asyncio.Event()
    session = FakeSession(
        [
            RealtimeEvent.tool_call("call-1", "terminal", {"command": "pwd"}),
            RealtimeEvent.transcript("still listening"),
        ]
    )

    async def dispatch(_name: str, _arguments: dict[str, Any]) -> str:
        await release.wait()
        return "done"

    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", session), dispatch_tool=dispatch
    )
    await coordinator.open(instructions="", tools=[])
    events = coordinator.events()

    assert (await anext(events)).type is RealtimeEventType.TOOL_CALL
    assert (
        await asyncio.wait_for(anext(events), timeout=0.1)
    ).type is RealtimeEventType.TRANSCRIPT
    release.set()
    with pytest.raises(StopAsyncIteration):
        await anext(events)
    async def wait_for_result() -> None:
        while not session.tool_results:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_result(), timeout=0.1)
    assert session.tool_results == [("call-1", "done")]


@pytest.mark.asyncio
async def test_duplicate_tool_call_is_dispatched_exactly_once():
    duplicate = RealtimeEvent.tool_call("call-1", "terminal", {"command": "pwd"})
    session = FakeSession([duplicate, duplicate])
    dispatched = 0

    async def dispatch(_name: str, _arguments: dict[str, Any]) -> str:
        nonlocal dispatched
        dispatched += 1
        return "done"

    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", session), dispatch_tool=dispatch
    )
    await coordinator.open(instructions="", tools=[])
    assert len([event async for event in coordinator.events()]) == 2
    await asyncio.gather(*coordinator._tool_tasks.values())

    assert dispatched == 1
    assert session.tool_results == [("call-1", "done")]


@pytest.mark.asyncio
async def test_tool_call_id_reuse_with_different_arguments_fails_closed():
    session = FakeSession(
        [
            RealtimeEvent.tool_call("call-1", "terminal", {"command": "pwd"}),
            RealtimeEvent.tool_call("call-1", "terminal", {"command": "whoami"}),
        ]
    )
    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", session), dispatch_tool=async_dispatch("done")
    )
    await coordinator.open(instructions="", tools=[])

    with pytest.raises(ValueError, match="reused with different arguments"):
        _ = [event async for event in coordinator.events()]


def test_coordinator_requires_cancellable_async_dispatch():
    with pytest.raises(TypeError, match="async callable"):
        RealtimeVoiceCoordinator(
            FakeProvider("fake", FakeSession([])),
            dispatch_tool=lambda _name, _arguments: "done",
        )


@pytest.mark.asyncio
async def test_provider_eof_does_not_wait_for_pending_tool():
    release = asyncio.Event()
    session = FakeSession([RealtimeEvent.tool_call("pending", "terminal", {})])

    async def dispatch(_name: str, _arguments: dict[str, Any]) -> str:
        await release.wait()
        return "done"

    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", session), dispatch_tool=dispatch
    )
    await coordinator.open(instructions="", tools=[])
    assert len(await asyncio.wait_for(
        _collect_events(coordinator), timeout=0.1
    )) == 1
    assert session.tool_results == []
    await coordinator.close()


async def _collect_events(
    coordinator: RealtimeVoiceCoordinator,
) -> list[RealtimeEvent]:
    return [event async for event in coordinator.events()]


@pytest.mark.asyncio
async def test_close_rejects_buffered_events_and_stale_tool_effects():
    release_event = asyncio.Event()
    release_tool = asyncio.Event()

    class BufferedSession(FakeSession):
        async def events(self) -> AsyncIterator[RealtimeEvent]:
            await release_event.wait()
            yield RealtimeEvent.audio(b"stale", item_id="old")

    session = BufferedSession([])

    async def dispatch(_name: str, _arguments: dict[str, Any]) -> str:
        await release_tool.wait()
        return "stale result"

    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", session), dispatch_tool=dispatch
    )
    await coordinator.open(instructions="", tools=[])
    events = coordinator.events()
    pending_event = asyncio.create_task(anext(events))
    await asyncio.sleep(0)
    coordinator._start_tool_dispatch(
        RealtimeEvent.tool_call("stale-call", "terminal", {}),
        session,
        coordinator._generation,
    )
    await coordinator.close()
    release_event.set()
    release_tool.set()
    with pytest.raises(StopAsyncIteration):
        await pending_event
    assert session.tool_results == []


@pytest.mark.asyncio
async def test_completed_duplicate_is_not_resubmitted():
    duplicate = RealtimeEvent.tool_call("duplicate", "terminal", {})
    session = FakeSession([duplicate, duplicate])
    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", session), dispatch_tool=async_dispatch("done")
    )
    await coordinator.open(instructions="", tools=[])
    events = coordinator.events()
    await anext(events)
    await asyncio.gather(*coordinator._tool_tasks.values())
    await anext(events)
    assert session.tool_results == [("duplicate", "done")]
    await coordinator.close()


@pytest.mark.asyncio
async def test_result_submission_failure_reaches_event_consumer(
    caplog: pytest.LogCaptureFixture,
):
    class FailingSubmissionSession(FakeSession):
        async def events(self) -> AsyncIterator[RealtimeEvent]:
            yield RealtimeEvent.tool_call("failed-submit", "terminal", {})
            await asyncio.Event().wait()

        async def submit_tool_result(self, call_id: str, output: str) -> None:
            raise RuntimeError("provider disconnected")

    session = FailingSubmissionSession([])
    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", session), dispatch_tool=async_dispatch("done")
    )
    await coordinator.open(instructions="", tools=[])
    events = coordinator.events()

    assert (await anext(events)).type is RealtimeEventType.TOOL_CALL
    failure = await asyncio.wait_for(anext(events), timeout=0.1)

    assert failure.type is RealtimeEventType.ERROR
    assert failure.text == (
        "realtime tool result submission failed: provider disconnected"
    )
    assert "failed-submit" not in coordinator._completed_tool_calls
    assert any(
        record.getMessage() == "Realtime voice tool result submission failed"
        and record.__dict__["call_id"] == "failed-submit"
        and record.exc_info is not None
        for record in caplog.records
    )
    await coordinator.close()


@pytest.mark.asyncio
async def test_completed_tool_dedupe_and_output_are_bounded():
    session = FakeSession(
        [RealtimeEvent.tool_call(f"call-{index}", "terminal", {}) for index in range(8)]
    )
    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", session),
        dispatch_tool=async_dispatch("x" * 1000),
        max_in_flight_tool_calls=8,
        max_completed_tool_calls=3,
    )
    await coordinator.open(instructions="", tools=[])
    await _collect_events(coordinator)
    await asyncio.gather(*coordinator._tool_tasks.values())
    assert len(coordinator._completed_tool_calls) == 3
    assert list(coordinator._completed_tool_calls) == ["call-5", "call-6", "call-7"]
    assert "x" * 1000 not in repr(coordinator._completed_tool_calls)
    assert coordinator._tool_calls == {}
    await coordinator.close()


@pytest.mark.asyncio
async def test_overload_is_bounded_and_fails_the_session():
    release = asyncio.Event()
    session = FakeSession(
        [
            RealtimeEvent.tool_call("active", "terminal", {}),
            *[
                RealtimeEvent.tool_call(f"overflow-{index}", "terminal", {})
                for index in range(1_000)
            ],
        ]
    )

    async def dispatch(_name: str, _arguments: dict[str, Any]) -> str:
        await release.wait()
        return "done"

    coordinator = RealtimeVoiceCoordinator(
        FakeProvider("fake", session),
        dispatch_tool=dispatch,
        max_in_flight_tool_calls=1,
    )
    await coordinator.open(instructions="", tools=[])
    observed = [event async for event in coordinator.events()]

    assert len(coordinator._tool_tasks) == 1
    assert list(coordinator._tool_calls) == ["active"]
    assert observed[-1].type is RealtimeEventType.ERROR
    assert observed[-1].text == (
        "too many realtime voice tool calls are already in flight"
    )
    release.set()
    await coordinator.close()
