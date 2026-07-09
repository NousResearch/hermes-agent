"""Per-satellite conversation turn state machine.

Pure logic: every method takes time as an argument and performs no I/O,
so the whole machine is unit-testable without sockets or clocks. The
adapter owns the side effects each returned action implies.
"""

from enum import Enum
from typing import Callable, Optional


class TurnPhase(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


class TurnMachine:
    def __init__(
        self,
        detector_factory: Callable,
        *,
        listen_timeout_seconds: float = 30.0,
    ):
        self._detector_factory = detector_factory
        self.listen_timeout_seconds = listen_timeout_seconds
        self.phase = TurnPhase.IDLE
        self._detector = None
        self._listen_started: float = 0.0
        # Monotonic turn generation. Bumped on every turn start AND every
        # reset, so an STT callback carrying an old turn_id can always be
        # recognized as stale — even if the machine has since re-entered
        # TRANSCRIBING for a newer turn.
        self.turn_id: int = 0

    def on_pipeline_start(self, now: float) -> bool:
        """Satellite reported wake + pipeline start. True if a turn began."""
        if self.phase is not TurnPhase.IDLE:
            return False
        self.turn_id += 1
        self.phase = TurnPhase.LISTENING
        self._detector = self._detector_factory()
        self._listen_started = now
        return True

    def on_audio(
        self, pcm: bytes, seconds: float, rate: int, now: float
    ) -> Optional[tuple]:
        if self.phase is not TurnPhase.LISTENING:
            return None
        # Timeout is checked before feeding: the chunk that crosses the
        # deadline is dropped rather than endpoint-checked (intentional —
        # a turn that stalled this long should abort, not dispatch).
        if now - self._listen_started > self.listen_timeout_seconds:
            self.to_idle()
            return ("abort",)
        assert self._detector is not None  # set in on_pipeline_start, which alone enters LISTENING
        utterance = self._detector.feed(pcm, seconds)
        if utterance is None:
            return None
        self.phase = TurnPhase.TRANSCRIBING
        return ("transcribe", utterance, rate, self.turn_id)

    def on_transcript_ready(self, text: str, turn_id: int) -> tuple:
        if turn_id != self.turn_id or self.phase is not TurnPhase.TRANSCRIBING:
            # STT callback from an earlier turn (aborted, disconnected, or
            # superseded by a new wake) or a duplicate within this turn:
            # ignore WITHOUT touching state, so a stale result can neither
            # dispatch as the current turn nor tear a live turn down.
            return ("stale",)
        if not text:
            self.to_idle()
            return ("abort",)
        self.phase = TurnPhase.THINKING
        return ("dispatch", text)

    def on_reply_started(self) -> None:
        # Called by play_tts from THINKING (turn reply) or IDLE (announce
        # path — marking the satellite busy while an announcement plays).
        # No-op when already SPEAKING so duplicate playback events can't
        # re-enter.
        if self.phase is TurnPhase.SPEAKING:
            return
        self.phase = TurnPhase.SPEAKING

    def on_playback_done(self) -> None:
        self.to_idle()

    def to_idle(self) -> None:
        self.turn_id += 1  # invalidate any in-flight callbacks for this turn
        self.phase = TurnPhase.IDLE
        self._detector = None
