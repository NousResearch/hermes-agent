"""Kanban auto-react debouncer.

When the kanban notifier delivers an "interesting" event (blocked / gave_up
/ crashed / timed_out) to a thread, the agent should also *react* — not
just see the notification text in the thread but actually get a turn to
do something about it (unblock, retry, escalate, document, whatever).

The naive design (one turn per event) explodes on fan-outs. The chosen
design is a state machine per session_key:

  Idle:
    event arrives → buffer it, arm a 10s debounce timer.
    More events within those 10s → join the same batch (timer NOT reset).
    User message arrives → cancel timer, piggyback buffered events as a
      preamble on the user's turn (zero extra turns).
    Timer fires → flush as a synthetic internal turn. State → InFlight.

  InFlight (agent is processing a turn — user or synthetic):
    Event arrives → just buffer, no timer.
    Turn finishes:
      buffer empty → state → Idle.
      buffer non-empty → arm fresh 10s debounce (state stays InFlight
        until that fires, but a *new* timer starts so any stragglers
        during the next 10s also batch in).

  Throughout: all events still post to the thread via the notifier's
  adapter.send() call. This module only controls when the *agent*
  gets a turn about them. The user keeps seeing every notification.

Loop prevention:
  * Synthetic turns use user_id="system:kanban-react" + internal=True.
  * A synthetic turn that creates more kanban cards will auto-sub the
    same thread (per the 2026-05-16 PR) so further blocks/crashes will
    queue. That's by design — that's the feedback loop we want. The
    runaway protection is that "completed" events never trigger reacts,
    so a healthy fan-out is silent.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Reacts only on events the agent could meaningfully *do* something about.
# Completed events post to the thread for visibility but never wake the agent.
REACT_KINDS = frozenset({"blocked", "gave_up", "crashed", "timed_out"})

# Debounce window: time after the first event in a batch before flushing.
# Additional events arriving within this window join the same batch but do
# NOT extend the timer — bounded latency from first-event-arrival to flush.
DEFAULT_DEBOUNCE_SECONDS = 10.0


@dataclass
class KanbanReactEvent:
    """One event tagged for the debouncer."""
    task_id: str
    kind: str           # "blocked" / "gave_up" / "crashed" / "timed_out"
    board: str
    summary: str        # human-facing text (matches what was posted to thread)
    received_at: float  # monotonic


@dataclass
class _SessionState:
    """Per-session_key state machine."""
    buffer: list[KanbanReactEvent] = field(default_factory=list)
    timer: Optional[asyncio.Task] = None      # the 10s debounce task
    in_flight: bool = False                   # agent is processing a turn
    # Snapshot of the source we should use when forging the synthetic
    # MessageEvent. Captured from the first event in a fresh batch — all
    # events in one batch share the same session_key (and thus the same
    # source), so a single snapshot suffices.
    source_proto: Optional[dict] = None


class KanbanReactCoordinator:
    """State machine + render logic. Thread-safe for asyncio (single loop)."""

    def __init__(
        self,
        flush_callback,
        *,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
    ) -> None:
        """
        flush_callback: async fn(session_key, events, source_proto) → None
            Called when a batch is ready to dispatch as a synthetic turn.
            The coordinator marks state as in_flight before calling, and
            the caller must call mark_turn_done(session_key) when the
            agent turn finishes (success or failure) so the next batch
            can start.
        """
        self._flush_cb = flush_callback
        self._debounce = float(debounce_seconds)
        self._states: dict[str, _SessionState] = {}
        # One lock per coordinator. All transitions are short and
        # asyncio-bound; coarse locking is simpler than per-key locks.
        self._lock = asyncio.Lock()

    async def record_event(
        self,
        session_key: str,
        event: KanbanReactEvent,
        source_proto: dict,
    ) -> None:
        """Notifier calls this after a successful adapter.send()."""
        if event.kind not in REACT_KINDS:
            return  # completed → post-only, no react
        async with self._lock:
            st = self._states.setdefault(session_key, _SessionState())
            st.buffer.append(event)
            if st.source_proto is None:
                st.source_proto = source_proto
            if st.in_flight:
                # An agent turn is running — just queue. mark_turn_done
                # will (re-)arm the debounce when the turn ends if the
                # buffer is non-empty at that point.
                logger.debug(
                    "kanban-react: queued (in-flight) sk=%s task=%s kind=%s",
                    session_key, event.task_id, event.kind,
                )
                return
            if st.timer is None or st.timer.done():
                # Idle → arm a fresh 10s timer.
                st.timer = asyncio.create_task(
                    self._debounce_then_flush(session_key)
                )
                logger.debug(
                    "kanban-react: armed debounce sk=%s task=%s kind=%s "
                    "(buffer=1, %.1fs)",
                    session_key, event.task_id, event.kind, self._debounce,
                )
            else:
                # Timer already running, just joined the batch.
                logger.debug(
                    "kanban-react: joined batch sk=%s task=%s kind=%s buffer=%d",
                    session_key, event.task_id, event.kind, len(st.buffer),
                )

    async def consume_for_user_turn(self, session_key: str) -> list[KanbanReactEvent]:
        """Called by _handle_message when a real user message lands.

        If there are buffered events, returns them (so the caller can
        prepend a preamble to the user's text) and cancels any pending
        debounce timer. Marks the session as in_flight so events arriving
        *during* the user's turn batch into the next round.
        """
        async with self._lock:
            st = self._states.setdefault(session_key, _SessionState())
            drained = st.buffer
            st.buffer = []
            if st.timer is not None and not st.timer.done():
                st.timer.cancel()
            st.timer = None
            # Always mark in_flight on a user turn so events during the
            # turn join the next batch (whether or not we drained any).
            st.in_flight = True
            if drained:
                logger.info(
                    "kanban-react: piggybacked %d event(s) onto user turn sk=%s",
                    len(drained), session_key,
                )
            return drained

    async def mark_turn_start(self, session_key: str) -> None:
        """Called when a synthetic flush turn starts dispatching.
        Sets in_flight so concurrent events queue rather than racing."""
        async with self._lock:
            st = self._states.setdefault(session_key, _SessionState())
            st.in_flight = True
            # Cancel any leftover timer — flush is happening now.
            if st.timer is not None and not st.timer.done():
                st.timer.cancel()
            st.timer = None

    async def mark_turn_done(self, session_key: str) -> None:
        """Caller invokes this after _handle_message returns (success OR
        failure). If events accumulated during the turn, arm a fresh
        debounce; otherwise return to idle.
        """
        async with self._lock:
            st = self._states.get(session_key)
            if st is None:
                return
            st.in_flight = False
            if st.buffer:
                # Stragglers from during the turn — give them a 10s
                # settle window for any final events, then flush.
                st.timer = asyncio.create_task(
                    self._debounce_then_flush(session_key)
                )
                logger.debug(
                    "kanban-react: post-turn re-armed debounce sk=%s buffer=%d",
                    session_key, len(st.buffer),
                )
            else:
                # Clean state — drop the entry so the dict stays small.
                if st.timer is None or st.timer.done():
                    self._states.pop(session_key, None)

    async def _debounce_then_flush(self, session_key: str) -> None:
        """Background task: sleep `debounce`, then call the flush
        callback with the drained buffer. Cancellable."""
        try:
            await asyncio.sleep(self._debounce)
        except asyncio.CancelledError:
            return
        async with self._lock:
            st = self._states.get(session_key)
            if st is None or not st.buffer:
                return
            events = st.buffer
            source_proto = st.source_proto
            st.buffer = []
            st.timer = None
            st.in_flight = True  # turn about to dispatch
        try:
            logger.info(
                "kanban-react: flushing %d event(s) sk=%s",
                len(events), session_key,
            )
            await self._flush_cb(session_key, events, source_proto)
        except Exception as exc:
            logger.warning(
                "kanban-react: flush callback failed sk=%s: %s",
                session_key, exc, exc_info=True,
            )
        finally:
            # Whether the synthetic turn succeeded or raised, release
            # the in_flight flag so future events can flow.
            await self.mark_turn_done(session_key)


def render_events_preamble(events: list[KanbanReactEvent]) -> str:
    """Format a batch of events into a short markdown preamble suitable
    for prepending to a user message OR as the body of a synthetic turn.

    Kept compact — every byte here is a token. One bullet per event,
    ordered by arrival time (which is also the order the user saw them
    posted in their thread).
    """
    if not events:
        return ""
    lines = [
        f"[Kanban auto-react — {len(events)} event(s) on board(s) "
        f"{', '.join(sorted({e.board for e in events}))}]",
    ]
    for ev in events:
        lines.append(f"  • {ev.task_id} **{ev.kind}**: {ev.summary}")
    return "\n".join(lines)
