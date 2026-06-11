"""Provider-neutral realtime voice session interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Union


@dataclass(frozen=True)
class RealtimeAudioDelta:
    """PCM16 audio emitted by a realtime provider for playback."""

    pcm16: bytes
    sample_rate: int


@dataclass(frozen=True)
class RealtimeTranscriptDelta:
    """User or assistant transcript text emitted during a realtime session."""

    role: Literal["user", "assistant"]
    text: str
    final: bool = False


@dataclass(frozen=True)
class RealtimeToolCall:
    """Tool request emitted by a realtime provider."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


RealtimeVoiceEvent = Union[
    RealtimeAudioDelta,
    RealtimeTranscriptDelta,
    RealtimeToolCall,
]


class RealtimeVoiceSession(ABC):
    """Contract implemented by concrete realtime voice provider sessions."""

    @abstractmethod
    async def start(self) -> None:
        """Open the provider session and prepare to send/receive audio."""

    @abstractmethod
    async def stop(self) -> None:
        """Close the provider session and release associated resources."""

    @abstractmethod
    async def send_audio_pcm16(self, data: bytes, sample_rate: int) -> None:
        """Stream PCM16 microphone/input audio into the provider."""

    @abstractmethod
    async def interrupt(self) -> None:
        """Cancel current provider output after user barge-in."""

    async def submit_tool_result(self, call_id: str, output: str) -> None:
        """Return a custom tool result to the provider conversation if supported."""
        raise NotImplementedError("Realtime provider does not support tool results")

    @abstractmethod
    async def update_instructions(self, instructions: str) -> None:
        """Update provider session instructions without reconnecting."""
