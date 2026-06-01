"""StreamingCallSession — the transport-agnostic reflex core (WP7).

This is the heart of the streaming voice slice.  It wires together the seven
ports (transport, STT, turn-detection, TTS, brain, interruption policy, tracer)
into a single cooperative-async reflex loop.

Design notes / concurrency model
---------------------------------
asyncio is single-threaded with cooperative scheduling, so shared mutable state
(``_speaking`` / ``_thinking`` / ``_active_scope``) is safe to read and write
without locks *as long as* we never ``await`` between a read and a dependent
write in a way that could interleave with the other task.  The two concurrent
contexts are:

  1. The inbound loop (``run``): drains ``transport.inbound()`` frame-by-frame,
     pushes each frame to STT + turn detection, and dispatches turn events.  It
     must NEVER block on the assistant turn — brain latency or TTS playback are
     run on a *separate* task (``_assistant_task``) so the inbound loop keeps
     draining (Scenario E).

  2. The assistant turn task (``_do_assistant_turn``): thinks (brain) then
     speaks (TTS), committing a CallTurnRecord at the end.  Barge-in is a fast
     reflex performed *by the inbound loop* against the active scope/transport
     while this task is still running; the task observes the cancelled scope
     cooperatively and finalizes as BARGED_IN.

No ``time.*`` / ``asyncio.sleep`` appears here: all timing is delegated to the
injected clock and to the fakes/engines that own the generators.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from .cancellation import CancellationScope
from .clock import Clock
from .ledger import HeardSpanLedger
from .ports import (
    AudioTransportPort,
    HermesBrainPort,
    InterruptionPolicyPort,
    SpeechToTextPort,
    StreamingCallTracerPort,
    TextToSpeechPort,
    TurnDetectionPort,
)
from .types import (
    BrainEventKind,
    CallTurnRecord,
    InterruptionAction,
    InterruptionDecision,
    InterruptionSignal,
    PlaybackMark,
    StreamingCallContext,
    TranscriptEvent,
    TtsEventKind,
    TurnEndReason,
    TurnEvent,
    TurnEventKind,
)

logger = logging.getLogger(__name__)

LedgerFactory = Callable[[str, int], HeardSpanLedger]
BrainFactory = Callable[[], HermesBrainPort]


def _default_ledger_factory(call_id: str, turn_index: int) -> HeardSpanLedger:
    return HeardSpanLedger(call_id, turn_index=turn_index)


class StreamingCallSession:
    """Transport-agnostic reflex core for a single streaming voice call."""

    def __init__(
        self,
        ctx: StreamingCallContext,
        *,
        transport: AudioTransportPort,
        stt: SpeechToTextPort,
        turns: TurnDetectionPort,
        tts: TextToSpeechPort,
        brain_factory: BrainFactory,
        policy: InterruptionPolicyPort,
        tracer: StreamingCallTracerPort,
        clock: Clock,
        ledger_factory: LedgerFactory | None = None,
    ) -> None:
        self.ctx = ctx
        self.transport = transport
        self.stt = stt
        self.turns = turns
        self.tts = tts
        self.brain_factory = brain_factory
        self.policy = policy
        self.tracer = tracer
        self.clock = clock
        self.ledger_factory = ledger_factory or _default_ledger_factory

        # Committed turns — exposed for assertions.
        self.records: list[CallTurnRecord] = []

        # Mutable reflex state.
        self._latest_partial: TranscriptEvent | None = None
        self._speaking: bool = False
        self._thinking: bool = False
        self._speech_start_ms: int | None = None
        self._active_scope: CancellationScope | None = None
        self._active_ledger: HeardSpanLedger | None = None
        self._full_text: str = ""
        self._turn_index: int = 0
        self._last_mark: PlaybackMark | None = None

        self._assistant_task: asyncio.Task[None] | None = None
        # A handle to the most recent assistant-turn task that survives
        # _cleanup_turn() (which nulls _assistant_task).  run() uses this to
        # await the task and observe an unexpected exception even after the
        # turn body has cleaned up its own reflex state (Fix B).
        self._pending_assistant_task: asyncio.Task[None] | None = None
        self._stt_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the call to completion (until inbound stream ends)."""
        self._stt_task = asyncio.create_task(self._consume_stt_events())
        # Fix C: surface any failure in the background STT consumer instead of
        # letting it be swallowed silently by the event loop.
        self._stt_task.add_done_callback(self._on_stt_task_done)
        try:
            async for frame in self.transport.inbound():
                await self.stt.push(frame)
                for te in await self.turns.observe(frame):
                    await self._on_turn_event(te)
        finally:
            # Inbound ended: let any in-flight assistant turn finish, then stop
            # the STT consumer.  The STT cancellation lives in a nested finally
            # so that an exception raised while awaiting the assistant turn
            # cannot leak the STT task (Fix B).
            try:
                # Await via the surviving handle (_cleanup_turn nulls
                # _assistant_task), so an unexpected turn exception propagates
                # out of run() instead of being stranded on an orphaned task.
                if self._pending_assistant_task is not None:
                    await self._pending_assistant_task
            finally:
                if self._stt_task is not None:
                    self._stt_task.cancel()
                    try:
                        await self._stt_task
                    except asyncio.CancelledError:
                        pass

    def _on_stt_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("STT consumer task failed", exc_info=exc)

    # ------------------------------------------------------------------
    # Background STT consumer
    # ------------------------------------------------------------------

    async def _consume_stt_events(self) -> None:
        async for event in self.stt.events():
            self._latest_partial = event
            self.tracer.transcript(event)

    # ------------------------------------------------------------------
    # Turn-event dispatch
    # ------------------------------------------------------------------

    @property
    def _in_assistant_turn(self) -> bool:
        return self._thinking or self._speaking

    async def _on_turn_event(self, te: TurnEvent) -> None:
        if not self._in_assistant_turn:
            self.tracer.turn(te)
            if te.kind is TurnEventKind.USER_SPEECH_STARTED:
                self._speech_start_ms = self.clock.now_ms()
            elif te.kind is TurnEventKind.ENDPOINT_DETECTED:
                final = await self.stt.finalize()
                if final is not None and final.text:
                    self._launch_assistant_turn(final)
            return

        # --- In an assistant turn: barge-in evaluation ---
        if te.kind is TurnEventKind.USER_SPEECH_STARTED:
            # Fast reflex: stamp speech start and flush queued audio immediately.
            # This does NOT cancel the brain (spec §5.3 flush-vs-interrupt order).
            self._speech_start_ms = self.clock.now_ms()
            result = await self.transport.flush_outbound("vad_trigger")
            if self._active_ledger is not None and self._full_text:
                self._active_ledger.note_flush(result, self._full_text)

        now = self.clock.now_ms()
        signal = InterruptionSignal(
            call_id=self.ctx.call_id,
            at_ms=now,
            # The assistant "holds the floor" while thinking OR speaking; both
            # states are interruptible by a qualifying barge-in (Scenario E).
            assistant_speaking=self._in_assistant_turn,
            turn_event=te,
            latest_partial=self._latest_partial,
            playhead=self._last_mark,
            params=self.ctx.interruption,
            ms_since_speech_start=now - (self._speech_start_ms or now),
        )
        decision = self.policy.decide(signal)

        if decision.action is InterruptionAction.INTERRUPT:
            await self._barge_in()
            self._trace_interruption(decision)
        elif decision.action is InterruptionAction.RESUME:
            # Scenario G: false positive. The brain was never interrupted; we do
            # not cancel and do not commit a BARGED_IN record.  Resuming playback
            # from the last mark is a no-op in this fake-driven slice.
            self._trace_interruption(decision)
        # IGNORE / WAIT → do nothing (keep speaking / thinking).

    def _trace_interruption(self, decision: InterruptionDecision) -> None:
        heard = self._last_mark.text_so_far if self._last_mark is not None else ""
        abandoned = self._full_text[len(heard):] if self._full_text else ""
        self.tracer.interruption(decision, heard, abandoned)

    async def _barge_in(self) -> None:
        """Real barge-in: flush, cancel TTS, cancel the scope (fires brain interrupt)."""
        result = await self.transport.flush_outbound("barge_in")
        if self._active_ledger is not None and self._full_text:
            self._active_ledger.note_flush(result, self._full_text)
        await self.tts.cancel()
        if self._active_scope is not None:
            self._active_scope.cancel("barge_in")

    # ------------------------------------------------------------------
    # Assistant turn
    # ------------------------------------------------------------------

    def _launch_assistant_turn(self, final: TranscriptEvent) -> None:
        task = asyncio.create_task(self._do_assistant_turn(final))
        self._assistant_task = task
        self._pending_assistant_task = task

    def _cleanup_turn(self) -> None:
        self._active_scope = None
        self._active_ledger = None
        self._speaking = False
        self._thinking = False
        self._last_mark = None
        self._speech_start_ms = None
        self._assistant_task = None

    def _commit(self, rec: CallTurnRecord) -> None:
        self.records.append(rec)
        self.tracer.turn_committed(rec)

    async def _do_assistant_turn(self, final: TranscriptEvent) -> None:
        # Fix B: the normal paths below each call _cleanup_turn() exactly once.
        # If brain.respond()/tts.synthesize() raises an UNEXPECTED exception
        # (i.e. not the handled BrainEventKind.ERROR path), we clean up the turn
        # state here so no _thinking/_speaking/_active_scope/_active_ledger flag
        # is left dirty, then re-raise so the failure is not silently swallowed
        # (run()'s finally still tears down the STT task).  CancelledError is
        # NOT caught — cooperative cancellation must propagate untouched.
        try:
            await self._run_assistant_turn(final)
        except Exception:
            self._cleanup_turn()
            raise

    async def _run_assistant_turn(self, final: TranscriptEvent) -> None:
        self._turn_index += 1
        scope = CancellationScope()
        self._active_scope = scope
        ledger = self.ledger_factory(self.ctx.call_id, self._turn_index)
        self._active_ledger = ledger
        brain = self.brain_factory()

        self._thinking = True
        self._full_text = ""
        self._last_mark = None

        # --- Think (brain) ---
        async for ev in brain.respond(final, self.ctx, scope):
            self.tracer.brain(ev)
            if ev.kind is BrainEventKind.FINAL_TEXT:
                self._full_text = ev.text
            elif ev.kind is BrainEventKind.ERROR:
                self._thinking = False
                rec = ledger.record(
                    user_transcript=final.text,
                    full_text=self._full_text,
                    reason=TurnEndReason.BRAIN_ERROR,
                )
                self._commit(rec)
                self._cleanup_turn()
                return

        self._thinking = False

        if scope.cancelled:
            # Barge-in arrived while the brain was still thinking.
            rec = ledger.record(
                user_transcript=final.text,
                full_text=self._full_text,
                reason=TurnEndReason.BARGED_IN,
            )
            self._commit(rec)
            self._cleanup_turn()
            return

        if not self._full_text:
            # Nothing to say: intentionally commit no record (Slice-1 behavior).
            self._cleanup_turn()
            return

        # --- Speak (TTS) ---
        self._speaking = True
        async for tev in self.tts.synthesize(self._full_text, self.ctx, scope):
            if tev.kind is TtsEventKind.AUDIO and tev.frame is not None:
                await self.transport.emit_outbound(tev.frame)
            elif tev.kind is TtsEventKind.MARK and tev.mark is not None:
                ledger.note_mark(tev.mark)
                self._last_mark = tev.mark
                self.tracer.playback(tev.mark)
            elif tev.kind is TtsEventKind.CANCELLED:
                break
            elif tev.kind is TtsEventKind.DONE:
                break
            if scope.cancelled:
                break

        self._speaking = False

        # The flush already happened in _barge_in; do NOT double-flush here.
        # The ledger already has the last confirmed mark via note_mark.
        reason = TurnEndReason.BARGED_IN if scope.cancelled else TurnEndReason.COMPLETED
        rec = ledger.record(
            user_transcript=final.text,
            full_text=self._full_text,
            reason=reason,
        )
        self._commit(rec)
        self._cleanup_turn()
