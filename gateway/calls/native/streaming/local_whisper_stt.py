"""Local faster-whisper SpeechToTextPort (segment STT, transcribe-on-finalize).

``LocalWhisperSTT`` buffers pushed 16 kHz mono PCM frames and, on
``finalize()`` (called by the session at the turn endpoint), transcribes the
whole buffered turn with local faster-whisper and returns a single ``FINAL``
``TranscriptEvent``. No partials, no cloud, no API key.

faster-whisper is heavy (ctranslate2 + onnxruntime) and lives in the opt-in
``simplex-streaming-local-stt`` extra. There is NO top-level
``import faster_whisper`` here — the factory imports it lazily so the streaming
package stays importable without the extra. numpy is imported lazily inside
``finalize()`` for the same reason.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from importlib.util import find_spec
from typing import TYPE_CHECKING

from .types import (
    AudioFrame,
    MediaFormat,
    StreamingCallContext,
    TranscriptEvent,
    TranscriptKind,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = ["LocalWhisperSTT", "build_local_whisper_stt"]

_PROVIDER = "faster-whisper"


class LocalWhisperSTT:
    """Satisfies ``SpeechToTextPort`` over local faster-whisper.

    The transcriber is injected (``transcribe: np.ndarray -> str``) so unit
    tests can mock it; the factory wraps the real ``WhisperModel.transcribe``.
    """

    def __init__(
        self,
        *,
        media: MediaFormat,
        transcribe: "Callable[[np.ndarray], str]",
        call_id: str = "",
    ) -> None:
        self._media = media
        self._transcribe: "Callable[[np.ndarray], str] | None" = transcribe
        self._call_id = call_id
        self._buffer = bytearray()
        self._first_ms: int | None = None
        self._last_ms: int | None = None

    async def start(self, ctx: StreamingCallContext) -> None:
        if not self._call_id:
            self._call_id = ctx.call_id

    async def push(self, frame: AudioFrame) -> None:
        if frame.media.sample_rate != 16000 or frame.media.channels != 1:
            raise ValueError(
                "LocalWhisperSTT requires 16kHz mono audio; got "
                f"{frame.media.sample_rate}Hz / {frame.media.channels}ch"
            )
        self._buffer.extend(frame.pcm16)
        if self._first_ms is None:
            self._first_ms = frame.timestamp_ms
        self._last_ms = frame.timestamp_ms

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        # No partials this slice: a real, empty async generator.
        if False:  # pragma: no cover - never yields
            yield

    async def finalize(self) -> TranscriptEvent | None:
        if not self._buffer:
            return None
        assert self._transcribe is not None
        import numpy as np

        audio = np.frombuffer(bytes(self._buffer), np.int16).astype(np.float32) / 32768.0
        # faster-whisper transcription is synchronous and CPU-bound (ctranslate2).
        # finalize() is awaited inline on the session's call task at the turn
        # endpoint, so running it directly would stall inbound-frame handling and
        # barge-in reflexes for the whole transcription. Offload to a worker thread.
        text = await asyncio.to_thread(self._transcribe, audio)
        event = TranscriptEvent(
            call_id=self._call_id,
            kind=TranscriptKind.FINAL,
            text=text.strip(),
            start_ms=self._first_ms or 0,
            end_ms=self._last_ms or 0,
            provider=_PROVIDER,
        )
        self._clear()
        return event

    async def cancel(self) -> None:
        self._clear()

    async def close(self) -> None:
        self._clear()
        self._transcribe = None

    def _clear(self) -> None:
        self._buffer = bytearray()
        self._first_ms = None
        self._last_ms = None


def build_local_whisper_stt(
    media: MediaFormat,
    *,
    call_id: str = "",
    model: str = "distil-small.en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> LocalWhisperSTT:
    """Construct a ``LocalWhisperSTT`` backed by a real ``WhisperModel``.

    Raises ``RuntimeError`` naming the extra when faster-whisper is absent.
    """
    if find_spec("faster_whisper") is None:
        raise RuntimeError(
            "LocalWhisperSTT requires the optional faster-whisper dependency. "
            "Install: pip install 'hermes-agent[simplex-streaming-local-stt]'"
        )
    from faster_whisper import WhisperModel

    whisper_model = WhisperModel(model, device=device, compute_type=compute_type)

    def _transcribe(audio: "np.ndarray") -> str:
        segments, _info = whisper_model.transcribe(audio, language="en")
        return " ".join(segment.text for segment in segments)

    return LocalWhisperSTT(media=media, transcribe=_transcribe, call_id=call_id)
