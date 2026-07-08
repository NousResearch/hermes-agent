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

    def on_pipeline_start(self, now: float) -> bool:
        """Satellite reported wake + pipeline start. True if a turn began."""
        if self.phase is not TurnPhase.IDLE:
            return False
        self.phase = TurnPhase.LISTENING
        self._detector = self._detector_factory()
        self._listen_started = now
        return True

    def on_audio(
        self, pcm: bytes, seconds: float, rate: int, now: float
    ) -> Optional[tuple]:
        if self.phase is not TurnPhase.LISTENING:
            return None
        if now - self._listen_started > self.listen_timeout_seconds:
            self.to_idle()
            return ("abort",)
        utterance = self._detector.feed(pcm, seconds)
        if utterance is None:
            return None
        self.phase = TurnPhase.TRANSCRIBING
        return ("transcribe", utterance, rate)

    def on_transcript_ready(self, text: str) -> tuple:
        if not text:
            self.to_idle()
            return ("abort",)
        self.phase = TurnPhase.THINKING
        return ("dispatch", text)

    def on_reply_started(self) -> None:
        self.phase = TurnPhase.SPEAKING

    def on_playback_done(self) -> None:
        self.to_idle()

    def to_idle(self) -> None:
        self.phase = TurnPhase.IDLE
        self._detector = None
