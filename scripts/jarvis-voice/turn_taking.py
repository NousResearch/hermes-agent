"""Small turn-taking helpers for JARVIS voice."""

from __future__ import annotations

import os


def read_float_env(name: str, default: float, minimum: float | None = None) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def read_int_env(name: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def read_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class TurnTakingGate:
    """Track playback tail and owner interruption without touching audio APIs."""

    def __init__(
        self,
        playback_tail_sec: float = 0.7,
        interrupt_rms: float = 2400.0,
        interrupt_frames: int = 3,
        voice_interrupt_enabled: bool = False,
    ) -> None:
        self.playback_tail_sec = max(0.0, playback_tail_sec)
        self.interrupt_rms = max(0.0, interrupt_rms)
        self.interrupt_frames = max(1, interrupt_frames)
        self.voice_interrupt_enabled = voice_interrupt_enabled
        self._last_play = 0.0
        self._loud_playback_frames = 0

    def note_playback(self, now: float) -> None:
        self._last_play = now

    def in_playback_tail(self, now: float) -> bool:
        return 0.0 < now - self._last_play < self.playback_tail_sec

    def should_interrupt_playback(self, rms: float) -> bool:
        if not self.voice_interrupt_enabled:
            self._loud_playback_frames = 0
            return False

        if rms >= self.interrupt_rms:
            self._loud_playback_frames += 1
        else:
            self._loud_playback_frames = 0

        if self._loud_playback_frames >= self.interrupt_frames:
            self._loud_playback_frames = 0
            return True
        return False

    def should_clear_for_server_interrupt(self, now: float, playback_active: bool) -> bool:
        if playback_active or self.in_playback_tail(now):
            return False
        return True

    def reset(self) -> None:
        self._last_play = 0.0
        self._loud_playback_frames = 0


class LocalSpeechTurnDetector:
    """Detect local speech boundaries before sending audio to Gemini Live."""

    def __init__(
        self,
        speech_rms: float = 500.0,
        silence_ms: int = 900,
        min_speech_frames: int = 1,
    ) -> None:
        self.speech_rms = max(0.0, speech_rms)
        self.silence_ms = max(100, silence_ms)
        self.min_speech_frames = max(1, min_speech_frames)
        self.active = False
        self._speech_frames = 0
        self._silent_ms = 0.0

    def observe(self, rms: float, frame_ms: float) -> str | None:
        frame_ms = max(0.0, frame_ms)
        if rms > self.speech_rms:
            self._speech_frames += 1
            self._silent_ms = 0.0
            if not self.active and self._speech_frames >= self.min_speech_frames:
                self.active = True
                return "start"
            return "speech" if self.active else None

        self._speech_frames = 0
        if self.active:
            self._silent_ms += frame_ms
            if self._silent_ms >= self.silence_ms:
                self.active = False
                self._silent_ms = 0.0
                return "end"
        return None

    def reset(self) -> None:
        self.active = False
        self._speech_frames = 0
        self._silent_ms = 0.0
