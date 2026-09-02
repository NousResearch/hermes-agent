"""
Realtime Voice Orchestrator
===========================

Drives one :class:`agent.realtime_voice_provider.RealtimeVoiceSession` on
behalf of a Hermes-owned host. The split is fixed:

* the **provider** owns audio transport and turn-taking (server VAD,
  response lifecycle, wire protocol);
* **Hermes** owns tools, approvals, history, and memory. Function calls the
  provider emits are executed here through a host-supplied executor and
  answered exactly once per ``call_id``.

What this module guarantees, independent of provider:

Response identity
    A cancelled response keeps emitting audio and transcript deltas for a
    moment. Every delta is fenced by ``response_id`` against a bounded ledger
    of settled responses, so a barge-in never plays the tail of the answer
    it interrupted over the next one.
Barge-in
    ``InputSpeechStarted`` drains local playback, sends one cancel per
    in-flight response (when the provider supports it), truncates the
    provider's copy of the interrupted item at the millisecond the operator
    actually heard (when supported — otherwise the local drop is the whole
    degrade, never a faked truncation), and cancels in-flight tool tasks.
Tool dispatch
    Tool calls start immediately on their own cancellable tasks. Every call
    is bounded by ``tool_timeout_s`` — a wedged tool yields an honest timeout
    result instead of holding the turn forever — and settles to exactly one
    outcome: result, error, timeout, interrupted, or provider-cancelled (no
    output submitted). Redelivered call ids are ignored. Results for one
    response are submitted as one ordered batch with one continuation after
    the provider reports the response complete, which is when the wire is
    ready for them.
Clean close
    Leaving :meth:`RealtimeVoiceOrchestrator.run` cancels every task it
    started and closes the session, on success, failure, or cancellation.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agent.realtime_voice_provider import (
    InputSpeechStarted,
    InputTranscript,
    OutputAudio,
    OutputTranscript,
    RealtimeCapability,
    RealtimeToolResult,
    RealtimeVoiceEvent,
    RealtimeVoiceSession,
    ResponseCompleted,
    ResponseStarted,
    SessionClosed,
    SessionFailure,
    SessionReady,
    SessionResumptionUpdate,
    ToolCall,
    ToolCallCancelled,
    UnsupportedRealtimeCapability,
)

logger = logging.getLogger(__name__)

ToolExecutor = Callable[[str, Mapping[str, Any]], Awaitable[Any]]

DEFAULT_TOOL_TIMEOUT_S = 60.0
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 8_000
#: How many settled response ids the ledger remembers. Late deltas arrive
#: within seconds of a cancel, so a short memory is enough — and a long
#: session must not grow it without bound.
MAX_SETTLED_RESPONSES = 32
#: Remembered call ids so a provider redelivery after reconnect cannot run a
#: tool twice. Bounded for the same reason.
MAX_REMEMBERED_CALL_IDS = 256
SHUTDOWN_TIMEOUT_S = 5.0

UNPARSEABLE_ARGS_TEXT = (
    "I couldn't run that tool: its arguments were not a valid JSON object. "
    "Say what you wanted and I'll try again."
)
TOOLS_UNAVAILABLE_TEXT = "Tools are not available in this voice session."
INTERRUPTED_TEXT = "The user interrupted, so this tool call was cancelled before it finished."


class RealtimeVoiceError(RuntimeError):
    """The session ended with a terminal provider failure."""


# -- host seam ---------------------------------------------------------------


class RealtimeVoiceHost:
    """Callbacks the owning Hermes surface receives; every default is a no-op.

    Callbacks run on the orchestrator's event loop and must return quickly.
    Only :meth:`on_barge_in` returns a value.
    """

    def on_session_ready(self, session_id: str) -> None:
        return None

    def on_session_closed(self, reason: str) -> None:
        return None

    def on_error(self, message: str, *, terminal: bool) -> None:
        return None

    def on_resumption_update(self, handle: str | None) -> None:
        return None

    def on_input_transcript(self, text: str, final: bool) -> None:
        return None

    def on_output_transcript(self, text: str, final: bool) -> None:
        return None

    def on_output_item_started(self, item_id: str) -> None:
        """Audio for a new assistant item is about to arrive (reset played-ms)."""
        return None

    def on_output_audio(self, pcm: bytes) -> None:
        """Queue assistant audio for playback (session ``output_audio_format``)."""
        return None

    def on_response_started(self, response_id: str | None) -> None:
        return None

    def on_response_completed(self, response_id: str | None) -> None:
        return None

    def on_barge_in(self) -> int | None:
        """Stop local playback now.

        Return how many milliseconds of the item currently being spoken the
        operator actually heard, or ``None`` when nothing was mid-playback.
        The orchestrator truncates the provider's copy at that point when the
        provider supports it.
        """
        return None

    def on_tool_call(self, call: ToolCall) -> None:
        return None

    def on_tool_result(self, call: ToolCall, output: str, outcome: str) -> None:
        """``outcome`` is one of ``ok``, ``error``, ``timeout``, ``interrupted``,
        ``provider_cancelled``, ``refused``."""
        return None


# -- blocking tool bridge ----------------------------------------------------


async def run_blocking_tool(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking callable on a daemon thread and await its result.

    Unlike :func:`asyncio.to_thread`, a wedged tool cannot hold process exit:
    the thread is a daemon, and the awaiting task can be cancelled or timed
    out independently (the thread then finishes on its own and its result is
    discarded). ``contextvars`` propagate like ``to_thread``.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()
    context = contextvars.copy_context()

    def _deliver(setter: Callable[[Any], None], value: Any) -> None:
        if future.done():
            return
        setter(value)

    def _worker() -> None:
        try:
            result = context.run(fn, *args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — the Future preserves it
            outcome = (future.set_exception, exc)
        else:
            outcome = (future.set_result, result)
        try:
            loop.call_soon_threadsafe(_deliver, *outcome)
        except RuntimeError:
            # The loop is closed; nobody is waiting for this result.
            logger.debug("realtime voice tool finished after its loop closed")

    threading.Thread(target=_worker, name="realtime-voice-tool", daemon=True).start()
    return await future


# -- response identity ledger ------------------------------------------------


class _ResponseLedger:
    """Which response may speak, as three named states rather than a boolean.

    Ported from the field-tested hermes-talk relay: a response stays *in
    flight* from ``ResponseStarted`` until its terminal event even after we
    asked the provider to cancel it, only its own ``response_id`` decides
    whether a delta is played, and an already-cancelled response that gets
    interrupted a second time is treated as having lost its terminal event.
    """

    def __init__(self, max_settled: int = MAX_SETTLED_RESPONSES) -> None:
        self._max_settled = max_settled
        self.in_flight = False
        self.active_id: str | None = None
        self._settled: OrderedDict[str, None] = OrderedDict()
        self._unnamed_cancelled = False

    def may_speak(self, response_id: str | None) -> bool:
        """Ambiguity fails open: muting real speech is the worse failure."""
        if response_id is None:
            return not self._unnamed_cancelled
        if response_id in self._settled:
            return False
        return self.active_id is None or response_id == self.active_id

    def _settle(self, response_id: str) -> None:
        if response_id in self._settled:
            return
        self._settled[response_id] = None
        while len(self._settled) > self._max_settled:
            self._settled.popitem(last=False)

    def start(self, response_id: str | None) -> None:
        if response_id is not None and response_id in self._settled:
            return  # a replayed start cannot revive a cancelled response
        if response_id is None and self.in_flight and self.active_id is not None:
            return  # an unnamed start cannot un-name a live response
        self._unnamed_cancelled = False
        self.in_flight = True
        self.active_id = response_id

    def cancel_active(self) -> bool:
        """Settle the interrupted response; ``False`` when nothing needs a cancel."""
        if not self.in_flight:
            return False
        if self.active_id is not None:
            if self.active_id in self._settled:
                self._release_lost_terminal()
                return False
            self._settle(self.active_id)
        elif self._unnamed_cancelled:
            self._release_lost_terminal()
            return False
        else:
            self._unnamed_cancelled = True
        return True

    def _release_lost_terminal(self) -> None:
        self.in_flight = False
        self.active_id = None
        self._unnamed_cancelled = False

    def finish(self, response_id: str | None) -> bool:
        """Close a response; ``False`` when the terminal named some other response."""
        if response_id is not None:
            self._settle(response_id)
        if (
            response_id is not None
            and self.active_id is not None
            and response_id != self.active_id
        ):
            return False
        self.in_flight = False
        self.active_id = None
        self._unnamed_cancelled = False
        return True


# -- tool ledger -------------------------------------------------------------


def _observe_task(task: asyncio.Task[None]) -> None:
    """Retrieve a finished task's exception so it is logged, never orphaned."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning(
            "realtime voice task %s failed", task.get_name(), exc_info=(type(exc), exc, exc.__traceback__)
        )


@dataclass
class _ToolCallState:
    call: ToolCall
    task: asyncio.Task[None] | None = None
    output: str | None = None
    outcome: str = "pending"
    submit: bool = True

    def settle(self, output: str, outcome: str) -> None:
        if self.output is None:
            self.output = output
            self.outcome = outcome


@dataclass
class _ToolBatch:
    call_ids: list[str] = field(default_factory=list)
    closed: bool = False


# -- orchestrator ------------------------------------------------------------


class RealtimeVoiceOrchestrator:
    """Run one session end to end for a :class:`RealtimeVoiceHost`."""

    def __init__(
        self,
        session: RealtimeVoiceSession,
        host: RealtimeVoiceHost,
        *,
        tool_executor: ToolExecutor | None = None,
        tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
        max_tool_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS,
        cancel_tools_on_barge_in: bool = True,
        shutdown_timeout_s: float = SHUTDOWN_TIMEOUT_S,
    ) -> None:
        if not isinstance(session, RealtimeVoiceSession):
            raise TypeError("session must be a RealtimeVoiceSession")
        if not isinstance(host, RealtimeVoiceHost):
            raise TypeError("host must be a RealtimeVoiceHost")
        if tool_timeout_s <= 0:
            raise ValueError("tool_timeout_s must be positive")
        if max_tool_output_chars <= 0:
            raise ValueError("max_tool_output_chars must be positive")
        self._session = session
        self._host = host
        self._tool_executor = tool_executor
        self._tool_timeout_s = float(tool_timeout_s)
        self._max_tool_output_chars = int(max_tool_output_chars)
        self._cancel_tools_on_barge_in = bool(cancel_tools_on_barge_in)
        self._shutdown_timeout_s = float(shutdown_timeout_s)
        self._responses = _ResponseLedger()
        self._calls: dict[str, _ToolCallState] = {}
        self._seen_call_ids: OrderedDict[str, None] = OrderedDict()
        self._batches: dict[str, _ToolBatch] = {}
        self._settle_tasks: set[asyncio.Task[None]] = set()
        self._last_audio_item_id: str | None = None
        self._failure: SessionFailure | None = None
        self._started = False
        self._closing = False

    # -- public surface ----------------------------------------------------

    @property
    def session(self) -> RealtimeVoiceSession:
        return self._session

    @property
    def response_active(self) -> bool:
        """Whether the provider still has a response open."""
        return self._responses.in_flight

    @property
    def pending_tool_calls(self) -> int:
        return sum(1 for state in self._calls.values() if state.output is None)

    async def send_audio(self, pcm: bytes) -> None:
        """Forward operator audio (``session.input_audio_format``) to the provider."""
        await self._session.send_audio(pcm)

    async def run(self) -> None:
        """Consume the session until it closes. Raises on terminal failure."""
        if self._started:
            raise RuntimeError("RealtimeVoiceOrchestrator.run() may only be called once")
        self._started = True
        try:
            async with self._session:
                try:
                    async for event in self._session.events():
                        await self._handle(event)
                finally:
                    await self._shutdown()
        finally:
            self._closing = True
        if self._failure is not None:
            raise RealtimeVoiceError(
                f"{self._failure.code}: {self._failure.message}"
            )

    # -- event dispatch ----------------------------------------------------

    async def _handle(self, event: RealtimeVoiceEvent) -> None:
        if isinstance(event, OutputAudio):
            if not self._responses.may_speak(event.response_id):
                return  # tail audio from a cancelled or superseded response
            if event.item_id is not None and event.item_id != self._last_audio_item_id:
                self._last_audio_item_id = event.item_id
                self._host.on_output_item_started(event.item_id)
            self._host.on_output_audio(event.data)
        elif isinstance(event, OutputTranscript):
            if self._responses.may_speak(event.response_id):
                self._host.on_output_transcript(event.text, event.final)
        elif isinstance(event, InputTranscript):
            self._host.on_input_transcript(event.text, event.final)
        elif isinstance(event, InputSpeechStarted):
            await self._barge_in()
        elif isinstance(event, ResponseStarted):
            self._responses.start(event.response_id)
            # Playback of the previous answer may still be draining, and a
            # barge-in during that tail must truncate the item the operator
            # heard, so the item id survives ResponseCompleted and resets here.
            self._last_audio_item_id = None
            self._host.on_response_started(event.response_id)
        elif isinstance(event, ResponseCompleted):
            self._responses.finish(event.response_id)
            if event.response_id is not None:
                self._close_batch(f"response:{event.response_id}")
            self._host.on_response_completed(event.response_id)
        elif isinstance(event, ToolCall):
            self._admit_tool_call(event)
        elif isinstance(event, ToolCallCancelled):
            self._provider_cancelled(event.call_ids)
        elif isinstance(event, SessionReady):
            self._host.on_session_ready(event.session_id)
        elif isinstance(event, SessionResumptionUpdate):
            self._host.on_resumption_update(event.handle if event.resumable else None)
        elif isinstance(event, SessionFailure):
            if event.terminal:
                self._failure = event
            self._host.on_error(event.message, terminal=event.terminal)
        elif isinstance(event, SessionClosed):
            self._host.on_session_closed(event.reason)

    # -- barge-in ----------------------------------------------------------

    async def _barge_in(self) -> None:
        played_ms = self._host.on_barge_in()
        item_id = self._last_audio_item_id
        active_id = self._responses.active_id
        needs_cancel = self._responses.cancel_active()
        if self._cancel_tools_on_barge_in:
            self._cancel_running_tools()
        if needs_cancel and self._session.supports(RealtimeCapability.RESPONSE_CANCELLATION):
            await self._wire_call("cancel", self._session.cancel_response(active_id))
        if item_id is not None and played_ms is not None:
            # One truncation per item: the drained item is spent either way.
            self._last_audio_item_id = None
            if self._session.supports(RealtimeCapability.OUTPUT_TRUNCATION):
                await self._wire_call(
                    "truncate", self._session.truncate_output(item_id, max(0, int(played_ms)))
                )
        # Without OUTPUT_TRUNCATION the local drain above is the entire
        # degrade: the provider keeps its full transcript and nothing is faked.

    async def _wire_call(self, what: str, operation: Awaitable[None]) -> None:
        """Run one best-effort provider write; the event stream owns terminal state."""
        try:
            await operation
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a failed write is reported, not fatal
            logger.warning("realtime voice: %s failed", what, exc_info=True)
            self._host.on_error(
                f"Could not {what} the response: {type(exc).__name__}: {exc}", terminal=False
            )

    # -- tools -------------------------------------------------------------

    def _admit_tool_call(self, call: ToolCall) -> None:
        if call.call_id in self._seen_call_ids:
            logger.debug("realtime voice: ignoring redelivered tool call %s", call.call_id)
            return
        self._seen_call_ids[call.call_id] = None
        while len(self._seen_call_ids) > MAX_REMEMBERED_CALL_IDS:
            self._seen_call_ids.popitem(last=False)
        state = _ToolCallState(call=call)
        self._calls[call.call_id] = state
        batch_key = self._batch_key(call)
        batch = self._batches.setdefault(batch_key, _ToolBatch())
        batch.call_ids.append(call.call_id)
        state.task = asyncio.create_task(
            self._run_tool_call(state), name=f"realtime-voice-tool:{call.call_id}"
        )
        state.task.add_done_callback(_observe_task)
        if call.response_id is None:
            # No response to wait for: the call is its own batch.
            self._close_batch(batch_key)

    @staticmethod
    def _batch_key(call: ToolCall) -> str:
        if call.response_id is not None:
            return f"response:{call.response_id}"
        return f"call:{call.call_id}"

    def _close_batch(self, batch_key: str) -> None:
        batch = self._batches.pop(batch_key, None)
        if batch is None or batch.closed:
            return
        batch.closed = True
        task = asyncio.create_task(
            self._settle_batch(list(batch.call_ids)),
            name=f"realtime-voice-settle:{batch_key}",
        )
        self._settle_tasks.add(task)
        task.add_done_callback(self._settle_tasks.discard)
        task.add_done_callback(_observe_task)

    async def _run_tool_call(self, state: _ToolCallState) -> None:
        call = state.call
        try:
            output, outcome = await self._execute_tool_call(call)
        except asyncio.CancelledError:
            state.settle(INTERRUPTED_TEXT, "interrupted")
            self._host.on_tool_result(call, state.output or "", state.outcome)
            raise
        # First settlement wins: a provider-side cancellation that landed
        # while the task was unwinding keeps its verdict.
        state.settle(output, outcome)
        self._host.on_tool_result(call, state.output or "", state.outcome)

    async def _execute_tool_call(self, call: ToolCall) -> tuple[str, str]:
        """Resolve one call to ``(output, outcome)``; only cancellation escapes."""
        arguments = self._parse_arguments(call.arguments)
        if arguments is None:
            return UNPARSEABLE_ARGS_TEXT, "error"
        if self._tool_executor is None:
            return TOOLS_UNAVAILABLE_TEXT, "refused"
        try:
            self._host.on_tool_call(call)
            raw = await asyncio.wait_for(
                self._tool_executor(call.name, arguments),
                timeout=self._tool_timeout_s,
            )
        except TimeoutError:
            return (
                f"The {call.name} tool did not finish within "
                f"{self._tool_timeout_s:g} seconds, so I stopped waiting for it. "
                "It may still be running in the background.",
                "timeout",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — every failure becomes a spoken result
            logger.warning("realtime voice tool %s failed", call.name, exc_info=True)
            return f"The {call.name} tool failed: {type(exc).__name__}: {exc}", "error"
        return self._coerce_output(raw), "ok"

    @staticmethod
    def _parse_arguments(raw: str) -> dict[str, Any] | None:
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _coerce_output(self, raw: Any) -> str:
        if isinstance(raw, str):
            text = raw
        else:
            try:
                text = json.dumps(raw, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                text = str(raw)
        if len(text) > self._max_tool_output_chars:
            text = text[: self._max_tool_output_chars] + " …[truncated]"
        return text

    async def _settle_batch(self, call_ids: list[str]) -> None:
        states = [self._calls[call_id] for call_id in call_ids if call_id in self._calls]
        running = [
            state.task
            for state in states
            if state.task is not None and not state.task.done()
        ]
        if running:
            await asyncio.wait(running)
        for state in states:
            self._calls.pop(state.call.call_id, None)
        results = [
            RealtimeToolResult(state.call.call_id, state.output)
            for state in states
            if state.submit and state.output is not None
        ]
        if not results or self._closing:
            return
        interrupted = any(state.outcome == "interrupted" for state in states)
        try:
            await self._session.submit_tool_results(
                results, continue_response=not interrupted
            )
        except asyncio.CancelledError:
            raise
        except UnsupportedRealtimeCapability:
            logger.warning(
                "realtime voice provider emitted tool calls without tool_calling capability"
            )
        except Exception as exc:  # noqa: BLE001 — a lost result must not kill the call
            logger.warning("realtime voice: tool results were not delivered", exc_info=True)
            self._host.on_error(
                f"Tool results could not be delivered: {type(exc).__name__}: {exc}",
                terminal=False,
            )

    def _cancel_running_tools(self) -> None:
        for state in self._calls.values():
            if state.task is not None and not state.task.done():
                state.task.cancel()

    def _provider_cancelled(self, call_ids: tuple[str, ...]) -> None:
        for call_id in call_ids:
            state = self._calls.get(call_id)
            if state is None:
                continue
            state.submit = False
            state.settle("", "provider_cancelled")
            if state.task is not None and not state.task.done():
                state.task.cancel()

    # -- shutdown ----------------------------------------------------------

    async def _shutdown(self) -> None:
        self._closing = True
        tasks: set[asyncio.Task[None]] = set(self._settle_tasks)
        for state in self._calls.values():
            if state.task is not None:
                tasks.add(state.task)
        for task in tasks:
            task.cancel()
        if not tasks:
            return
        _done, pending = await asyncio.wait(tasks, timeout=self._shutdown_timeout_s)
        if pending:
            logger.warning(
                "realtime voice: %d task(s) did not stop within %.1fs",
                len(pending),
                self._shutdown_timeout_s,
            )


__all__ = [
    "DEFAULT_MAX_TOOL_OUTPUT_CHARS",
    "DEFAULT_TOOL_TIMEOUT_S",
    "INTERRUPTED_TEXT",
    "MAX_REMEMBERED_CALL_IDS",
    "MAX_SETTLED_RESPONSES",
    "TOOLS_UNAVAILABLE_TEXT",
    "UNPARSEABLE_ARGS_TEXT",
    "RealtimeVoiceError",
    "RealtimeVoiceHost",
    "RealtimeVoiceOrchestrator",
    "ToolExecutor",
    "run_blocking_tool",
]
