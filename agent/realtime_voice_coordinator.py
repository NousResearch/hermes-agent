"""Hermes-owned coordination for provider-neutral realtime voice sessions."""

from __future__ import annotations
import asyncio
import inspect
import logging
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from agent.realtime_voice import (
    HeardAudioBoundary,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeSession,
    RealtimeVoiceProvider,
)

ToolDispatcher = Callable[[str, dict[str, Any]], Awaitable[str]]
logger = logging.getLogger(__name__)


class RealtimeVoiceCoordinator:
    """Relay audio while dispatching every tool call through the Hermes host."""

    def __init__(
        self,
        provider: RealtimeVoiceProvider,
        *,
        dispatch_tool: ToolDispatcher,
        max_in_flight_tool_calls: int = 16,
        max_completed_tool_calls: int = 256,
    ) -> None:
        if max_in_flight_tool_calls < 1:
            raise ValueError("max_in_flight_tool_calls must be positive")
        if max_completed_tool_calls < 1:
            raise ValueError("max_completed_tool_calls must be positive")
        if not (
            inspect.iscoroutinefunction(dispatch_tool)
            or inspect.iscoroutinefunction(getattr(dispatch_tool, "__call__", None))
        ):
            raise TypeError("dispatch_tool must be an async callable")
        self._provider = provider
        self._dispatch_tool = dispatch_tool
        self._session: RealtimeSession | None = None
        self._current_item_id: str | None = None
        self._current_audio_event: RealtimeEvent | None = None
        self._heard_boundary: HeardAudioBoundary | None = None
        self._max_in_flight_tool_calls = max_in_flight_tool_calls
        self._max_completed_tool_calls = max_completed_tool_calls
        self._generation = 0
        self._tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}
        self._completed_tool_calls: OrderedDict[
            str, tuple[str, dict[str, Any]]
        ] = OrderedDict()
        self._tool_tasks: dict[str, asyncio.Task[None]] = {}
        self._errors: asyncio.Queue[RealtimeEvent] = asyncio.Queue(maxsize=1)

    async def open(
        self,
        *,
        instructions: str,
        tools: list[dict[str, Any]],
        voice: str | None = None,
    ) -> None:
        if self._session is not None:
            raise RuntimeError("Realtime voice session is already open")
        session = await self._provider.open_session(
            instructions=instructions, tools=tools, voice=voice
        )
        self._generation += 1
        self._session = session
        self._tool_calls.clear()
        self._completed_tool_calls.clear()
        self._errors = asyncio.Queue(maxsize=1)
        self._reset_output_state()

    def _require_session(self) -> RealtimeSession:
        if self._session is None:
            raise RuntimeError("Realtime voice session is not open")
        return self._session

    async def send_audio(self, pcm: bytes) -> None:
        await self._require_session().send_audio(pcm)

    async def add_context(self, item_id: str, text: str) -> None:
        """Append silent progress context to the active provider session."""

        if not item_id or not text.strip():
            return
        await self._require_session().add_context(item_id, text)

    def report_audio_heard(self, event: RealtimeEvent, *, audio_end_ms: int) -> bool:
        """Record playback progress only for audio emitted by this open epoch."""
        if (
            self._session is None
            or event.type is not RealtimeEventType.AUDIO
            or not event.item_id
            or event.item_id != self._current_item_id
            or self._current_audio_event is not event
            or audio_end_ms < 0
        ):
            return False
        boundary = HeardAudioBoundary(event.item_id, audio_end_ms)
        if self._heard_boundary and audio_end_ms < self._heard_boundary.audio_end_ms:
            return False
        self._heard_boundary = boundary
        return True

    async def cancel_response(self) -> None:
        session = self._require_session()
        boundary, self._heard_boundary = self._heard_boundary, None
        await session.cancel_response()
        if boundary is not None:
            await session.truncate_response(boundary)

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        session = self._require_session()
        generation = self._generation
        provider_events = session.events().__aiter__()
        provider_next = asyncio.create_task(anext(provider_events))
        error_next = asyncio.create_task(self._errors.get())
        try:
            while True:
                done, _ = await asyncio.wait(
                    (provider_next, error_next),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if error_next in done:
                    provider_next.cancel()
                    await asyncio.gather(provider_next, return_exceptions=True)
                    yield error_next.result()
                    return
                try:
                    event = provider_next.result()
                except StopAsyncIteration:
                    return
                if not self._is_current_session(session, generation):
                    return
                if event.type is RealtimeEventType.AUDIO and event.item_id:
                    if event.item_id != self._current_item_id:
                        self._current_item_id = event.item_id
                        self._heard_boundary = None
                    self._current_audio_event = event
                if event.type is RealtimeEventType.TOOL_CALL:
                    self._start_tool_dispatch(event, session, generation)
                if not self._is_current_session(session, generation):
                    return
                yield event
                provider_next = asyncio.create_task(anext(provider_events))
        finally:
            provider_next.cancel()
            error_next.cancel()
            await asyncio.gather(provider_next, error_next, return_exceptions=True)

    def _start_tool_dispatch(
        self, event: RealtimeEvent, session: RealtimeSession, generation: int
    ) -> None:
        if not event.call_id or not event.tool_name:
            raise ValueError("Realtime tool_call events require call_id and tool_name")
        call = (event.tool_name, dict(event.arguments))
        active = self._tool_calls.get(event.call_id)
        if active is not None:
            if active != call:
                raise ValueError(
                    f"Realtime tool call {event.call_id!r} was reused with different arguments"
                )
            return
        completed = self._completed_tool_calls.get(event.call_id)
        if completed is not None:
            if completed != call:
                raise ValueError(
                    f"Realtime tool call {event.call_id!r} was reused with different arguments"
                )
            self._completed_tool_calls.move_to_end(event.call_id)
            return

        if len(self._tool_tasks) >= self._max_in_flight_tool_calls:
            self._publish_error(
                "too many realtime voice tool calls are already in flight",
                session,
                generation,
            )
            return
        self._tool_calls[event.call_id] = call
        task = asyncio.create_task(self._dispatch(event, session, generation))
        self._tool_tasks[event.call_id] = task
        task.add_done_callback(
            lambda completed_task, call_id=event.call_id: self._tool_task_done(
                call_id,
                completed_task,
                session,
                generation,
            )
        )

    def _tool_task_done(
        self,
        call_id: str,
        task: asyncio.Task[None],
        session: RealtimeSession,
        generation: int,
    ) -> None:
        if self._tool_tasks.get(call_id) is task:
            self._tool_tasks.pop(call_id, None)
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            logger.error(
                "Realtime voice tool result submission failed",
                extra={"call_id": call_id},
                exc_info=(type(exception), exception, exception.__traceback__),
            )
            self._publish_error(
                f"realtime tool result submission failed: {exception}",
                session,
                generation,
            )

    async def _dispatch(
        self, event: RealtimeEvent, session: RealtimeSession, generation: int
    ) -> None:
        if not event.call_id or not event.tool_name:
            raise ValueError("Realtime tool_call events require call_id and tool_name")
        try:
            result = await self._dispatch_tool(
                event.tool_name,
                dict(event.arguments),
            )
            output = str(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Realtime voice tool dispatch failed",
                extra={"tool_name": event.tool_name, "call_id": event.call_id},
                exc_info=True,
            )
            output = f"Error: {exc}"
        if not self._is_current_session(session, generation):
            return
        call = (event.tool_name, dict(event.arguments))
        await self._submit_result(event.call_id, output, session, generation)
        if self._is_current_session(session, generation):
            self._complete_call(event.call_id, call)

    async def _submit_result(
        self, call_id: str, output: str, session: RealtimeSession, generation: int
    ) -> None:
        if self._is_current_session(session, generation):
            await session.submit_tool_result(call_id, output)

    def _publish_error(
        self,
        message: str,
        session: RealtimeSession,
        generation: int,
    ) -> None:
        if not self._is_current_session(session, generation) or self._errors.full():
            return
        self._errors.put_nowait(
            RealtimeEvent(type=RealtimeEventType.ERROR, text=message)
        )

    def _complete_call(
        self, call_id: str, call: tuple[str, dict[str, Any]]
    ) -> None:
        self._tool_calls.pop(call_id, None)
        self._completed_tool_calls[call_id] = call
        self._completed_tool_calls.move_to_end(call_id)
        while len(self._completed_tool_calls) > self._max_completed_tool_calls:
            self._completed_tool_calls.popitem(last=False)

    def _is_current_session(
        self, session: RealtimeSession, generation: int
    ) -> bool:
        return self._session is session and self._generation == generation

    async def close(self) -> None:
        session, self._session = self._session, None
        self._generation += 1
        tasks = tuple(self._tool_tasks.values())
        self._tool_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tool_calls.clear()
        self._completed_tool_calls.clear()
        self._reset_output_state()
        if session is not None:
            await session.close()

    def _reset_output_state(self) -> None:
        self._current_item_id = None
        self._current_audio_event = None
        self._heard_boundary = None
