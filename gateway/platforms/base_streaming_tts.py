"""Streaming-TTS format descriptor, handle and turn-scoped suppression helpers (#60671).

Re-exported from ``gateway.platforms.base``; no imports from it (base imports this module).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
# --------------------------------------------------------------------------- Streaming TTS format
# descriptor and handle (#60671) ---------------------------------------------------------------------------
class AudioFormat:
    """Declared PCM format for a streaming-TTS session: every ``write_streaming_tts``
    chunk must be raw little-endian PCM at this rate / channels / sample width."""
    sample_rate: int = 24000
    channels: int = 1
    sample_width: int = 2  # bytes per sample (int16 = 2)


@dataclass
class StreamingTTSHandle:
    """Opaque handle returned by ``begin_streaming_tts``; adapters may extend it with
    platform state. The base fields are consumer bookkeeping / cancellation."""
    chat_id: str = ""
    audio_format: AudioFormat = field(default_factory=AudioFormat)
    # True once the first PCM chunk is written: a later failure then ends cleanly instead of
    # falling back to whole-file TTS (don't replay already-audible output).
    audible: bool = False
    aborted: bool = False  # set by abort_streaming_tts; late chunks are dropped


def streaming_tts_turn_key(session_key: str | None, turn_marker: Any = None, *, event: Any = None) -> str | None:
    """Per-turn streaming-TTS suppression key — turn-scoped (not chat-scoped) so
    overlapping turns in one chat can't suppress each other's fallback paths.
    ``turn_marker`` is normally the run generation, else the event's message/update id."""
    if not session_key:
        return None
    if turn_marker is None and event is not None:
        turn_marker = getattr(event, "message_id", None) or getattr(event, "platform_update_id", None)
    return None if turn_marker is None else f"{session_key}:{turn_marker}"


def streaming_tts_should_skip_whole_file(completed_turns: set[str], session_key: str | None,
                                         turn_marker: Any = None, *, event: Any = None) -> bool:
    """Pure, turn-scoped auto-TTS suppression decision (testable without the adapter stack)."""
    turn_key = streaming_tts_turn_key(session_key, turn_marker, event=event)
    return bool(turn_key and turn_key in completed_turns)
