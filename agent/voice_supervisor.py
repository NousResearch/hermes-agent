"""Surface-agnostic supervisor brain for realtime voice.

One brain, many surfaces: the CLI, Discord voice channels, and browser
clients each provide a :class:`TurnRunner` + an event callback; the
consult/steer lifecycle, instant acknowledgments, progress narration, and
stale-consult self-healing live here exactly once.

The controller is transport-dumb: it talks to a realtime session
(function calls in, ``send_function_output`` / ``speak_acknowledgment`` /
``speak_verbatim`` out) and to a surface-supplied turn runner.

Mirror contract with ``apps/shared/src/voice-supervisor.ts`` (browser
surfaces hold their own socket, so the machine exists once per language —
keep these in lockstep):

* tool names (``consult_hermes`` / ``steer_hermes``), busy/steer/failure
  reply texts, and the ack-phrase list (tools/voice_realtime_config.py ↔
  apps/shared/src/realtime-voice.ts)
* MAX_CONSULT_OUTPUT_CHARS=6000, STALE_CONSULT_MIN_AGE=30s,
  NARRATE_INTERVAL=12s (and the session's 20s quiet-wait before
  ``response.create``)
* deliberate asymmetry: here a lock spans the whole steer, so a turn
  cannot complete mid-steer; the TS controller awaits gateway RPCs
  mid-steer and needs its explicit ``steering`` guard instead.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Protocol

from tools.voice_realtime_config import CONSULT_TOOL_NAME, STEER_TOOL_NAME

logger = logging.getLogger(__name__)

# A consult whose turn is provably dead (runner idle, queue empty) must be
# older than this before it is failed out — rules out the enqueue→run window.
STALE_CONSULT_MIN_AGE_S = 30.0

# Minimum gap between spoken tool-progress lines.
NARRATE_INTERVAL_S = 12.0

# Function-output payload cap (the full text lives in session history).
MAX_CONSULT_OUTPUT_CHARS = 6000


class TurnRunner(Protocol):
    """The surface's agent-turn seam."""

    def submit(self, task: str) -> bool:
        """Queue *task* as a normal agent turn. True if it was accepted."""

    def interrupt(self) -> None:
        """Interrupt the currently running turn (steering)."""

    def is_busy(self) -> bool:
        """True while an agent turn is executing."""

    def is_queue_empty(self) -> bool:
        """True when no turn is queued waiting to run."""


def _owns_turn_text(task: str, message: str) -> bool:
    """Match a consult to a finished turn without substring false positives.

    Equality covers the normal path. Line equality covers pending-message
    coalescing (``task\\n\\nsteer``) without treating ``ls`` as a match for
    ``please list files``.
    """
    if message == task:
        return True
    for line in message.splitlines():
        if line.strip() == task:
            return True
    return False


class VoiceSupervisorController:
    """Consult/steer lifecycle between a realtime voice session and a surface.

    Events (``on_event(kind, text)``) let each surface render its own UI:
    ``consult`` / ``steer`` — a task/instruction was accepted from voice.
    """

    def __init__(
        self,
        session: Any,
        runner: TurnRunner,
        *,
        narrate: bool = True,
        on_event: Optional[Callable[[str, str], None]] = None,
    ):
        self._session = session
        self._runner = runner
        self._narrate = narrate
        self._on_event = on_event
        self._consult: Optional[Dict[str, Any]] = None
        self._last_narrate = 0.0
        # Consult transitions run on different threads (function calls on the
        # session's recv thread, completions on the surface's turn thread);
        # the lock keeps e.g. a failed steer's revert from clobbering a
        # consult that replaced it in between.
        self._lock = threading.Lock()

    @property
    def session(self) -> Any:
        """The realtime session this controller is bound to."""
        return self._session

    @property
    def consult_active(self) -> bool:
        return self._consult is not None

    def reset(self) -> None:
        """Drop tracked state (session teardown)."""
        with self._lock:
            self._consult = None

    def fail_active_consult(self, reason: str) -> None:
        """Fail the in-flight tool call (session restart / reconnect)."""
        with self._lock:
            consult = self._consult
            self._consult = None
        if consult is None or not self._session.alive:
            return
        try:
            self._session.send_function_output(consult["call_id"], reason)
        except Exception:
            logger.debug("voice supervisor fail_active_consult failed", exc_info=True)

    def _emit(self, kind: str, text: str) -> None:
        if self._on_event is not None:
            try:
                self._on_event(kind, text)
            except Exception:
                logger.debug("voice supervisor event callback failed", exc_info=True)

    def on_function_call(self, name: str, call_id: str, args_json: str) -> None:
        """Dispatch a tool call from the voice model."""
        try:
            args = json.loads(args_json) or {}
        except (ValueError, TypeError):
            args = {}
        if name == STEER_TOOL_NAME:
            with self._lock:
                self._on_steer(call_id, args)
            return
        if name != CONSULT_TOOL_NAME:
            self._session.send_function_output(call_id, f"Unknown tool: {name}")
            return
        task = str(args.get("task") or "").strip()
        if not task:
            self._session.send_function_output(call_id, "No task provided.")
            return
        with self._lock:
            stale = self._take_stale_consult()
            if stale is not None:
                self._session.send_function_output(
                    stale["call_id"], "That task failed without producing a result."
                )
            if self._consult is not None:
                self._session.send_function_output(
                    call_id,
                    "Hermes is still working on the previous task. This new "
                    "request was NOT queued. If it corrects, redirects, or "
                    "extends the active task, call steer_hermes now. Otherwise "
                    "wait for completion, then call consult_hermes again.",
                )
                return
            try:
                accepted = bool(self._runner.submit(task))
            except Exception:
                logger.debug("voice consult submit failed", exc_info=True)
                accepted = False
            if not accepted:
                self._session.send_function_output(
                    call_id,
                    "Could not start that task (no speaker or the turn was dropped).",
                )
                return
            # ``tasks`` keeps every text this consult has run under (the
            # original plus each steer): a steered turn drained in-band by
            # the gateway completes under the ORIGINAL event's text.
            self._consult = {
                "call_id": call_id,
                "task": task,
                "tasks": [task],
                "at": time.monotonic(),
            }
        self._emit("consult", task)
        if not self._session.last_response_had_audio:
            self._session.speak_acknowledgment()

    def _on_steer(self, call_id: str, args: Dict[str, Any]) -> None:
        """Interrupt Hermes and continue with the user's instruction.
        Caller holds the lock."""
        instruction = str(args.get("instruction") or "").strip()
        if not instruction:
            self._session.send_function_output(call_id, "No steering instruction provided.")
            return
        if self._consult is None:
            self._session.send_function_output(
                call_id,
                "No Hermes task is running — use consult_hermes to start one.",
            )
            return
        self._emit("steer", instruction)
        steered = self._consult
        prev_task = steered["task"]
        steered["task"] = instruction
        steered["tasks"].append(instruction)
        steered["at"] = time.monotonic()
        try:
            if self._runner.is_busy():
                self._runner.interrupt()
        except Exception as e:
            logger.debug("voice steer interrupt failed: %s", e)
        try:
            accepted = bool(self._runner.submit(instruction))
        except Exception:
            logger.debug("voice steer submit failed", exc_info=True)
            accepted = False
        if not accepted:
            # Revert only the consult we steered — never a successor that
            # replaced it while the submit was in flight.
            if self._consult is steered:
                steered["task"] = prev_task
                steered["tasks"].pop()
            self._session.send_function_output(
                call_id, "Steering failed — Hermes could not queue the new instruction."
            )
            return
        self._session.send_function_output(
            call_id, "Steering applied — Hermes is adjusting course."
        )

    def _take_stale_consult(self) -> Optional[Dict[str, Any]]:
        """Caller holds the lock."""
        consult = self._consult
        if (
            consult is None
            or self._runner.is_busy()
            or not self._runner.is_queue_empty()
            or time.monotonic() - consult["at"] < STALE_CONSULT_MIN_AGE_S
        ):
            return None
        self._consult = None
        return consult

    def owns_turn(self, message: Any) -> bool:
        """True when *message* is the active consult's turn.

        Surfaces must keep classic TTS silent for that turn. Line equality
        (not substring containment) matches coalesced queued text. Every text
        the consult has run under matches (original task and each steer).
        """
        consult = self._consult
        if not consult or not isinstance(message, str):
            return False
        return any(_owns_turn_text(task, message) for task in consult["tasks"])

    def on_turn_complete(
        self, message: Any, response: str, *, interrupted: bool = False
    ) -> bool:
        """Report a finished consult turn back to the voice session.
        True → this turn was consumed (the surface must not TTS it).

        ``interrupted`` marks a turn that stopped early (a steer interrupts
        the running turn): its partial output is never the consult result —
        the resubmitted turn owns that — so the consult stays armed.
        """
        with self._lock:
            consult = self._consult
            if not consult or interrupted or not self.owns_turn(message):
                return False
            self._consult = None
        if not self._session.alive:
            return False
        output = (response or "").strip() or "Hermes finished with no text output."
        if len(output) > MAX_CONSULT_OUTPUT_CHARS:
            output = (
                output[:MAX_CONSULT_OUTPUT_CHARS]
                + "\n[truncated — full text is in session history]"
            )
        self._session.send_function_output(consult["call_id"], output)
        return True

    def narrate_tool(self, function_name: str) -> None:
        """Speak a short verbatim progress line while a consult runs."""
        if (
            not function_name
            or self._consult is None
            or not self._narrate
            or not self._session.alive
        ):
            return
        now = time.monotonic()
        if now - self._last_narrate < NARRATE_INTERVAL_S:
            return
        self._last_narrate = now
        try:
            self._session.speak_verbatim(
                f"Running {function_name.replace('_', ' ')}.", interruptible=True
            )
        except Exception:
            pass

    def notify(self, text: str) -> None:
        """Best-effort spoken notice (e.g. 'Hermes needs your approval')."""
        if self._consult is None:
            return
        try:
            self._session.speak_verbatim(text, interruptible=True)
        except Exception:
            pass


__all__ = [
    "MAX_CONSULT_OUTPUT_CHARS",
    "NARRATE_INTERVAL_S",
    "STALE_CONSULT_MIN_AGE_S",
    "TurnRunner",
    "VoiceSupervisorController",
]
