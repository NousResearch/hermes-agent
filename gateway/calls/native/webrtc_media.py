from __future__ import annotations

import math
import struct
import time
from typing import Any, Protocol


class VoiceTurnPipeline(Protocol):
    async def process_pcm16(
        self,
        *,
        call_id: str,
        pcm16: bytes,
        sample_rate: int,
    ) -> Any:
        raise NotImplementedError


class QueuedOutputTrack(Protocol):
    async def queue_file(self, audio_path: str) -> None:
        raise NotImplementedError


def pcm16_rms(pcm16: bytes) -> float:
    if len(pcm16) < 2:
        return 0.0
    usable_len = len(pcm16) - (len(pcm16) % 2)
    sample_count = usable_len // 2
    if sample_count <= 0:
        return 0.0
    total = 0
    for (sample,) in struct.iter_unpack("<h", pcm16[:usable_len]):
        total += sample * sample
    return math.sqrt(total / sample_count)


class AudioUtteranceAccumulator:
    def __init__(
        self,
        *,
        call_id: str,
        pipeline: VoiceTurnPipeline,
        output_track: QueuedOutputTrack,
        sample_rate: int,
        voice_rms_threshold: float = 500.0,
        silence_seconds: float = 0.8,
    ) -> None:
        self.call_id = call_id
        self.pipeline = pipeline
        self.output_track = output_track
        self.sample_rate = sample_rate
        self.voice_rms_threshold = voice_rms_threshold
        self.silence_seconds = silence_seconds
        self._audio_buffer: list[bytes] = []
        self._last_voice_time: float | None = None
        self._processing = False

    async def accept_pcm16(self, pcm16: bytes, *, now: float | None = None) -> None:
        if self._processing:
            return
        current_time = time.monotonic() if now is None else now
        energy = pcm16_rms(pcm16)
        if energy > self.voice_rms_threshold:
            self._audio_buffer.append(pcm16)
            self._last_voice_time = current_time
            return
        if not self._audio_buffer or self._last_voice_time is None:
            return
        if current_time - self._last_voice_time >= self.silence_seconds:
            await self.flush()

    async def flush(self) -> None:
        if not self._audio_buffer or self._processing:
            return
        pcm16 = b"".join(self._audio_buffer)
        self._audio_buffer = []
        self._last_voice_time = None
        self._processing = True
        try:
            result = await self.pipeline.process_pcm16(
                call_id=self.call_id,
                pcm16=pcm16,
                sample_rate=self.sample_rate,
            )
            if bool(getattr(result, "ok", False)) and getattr(result, "audio_path", None):
                await self.output_track.queue_file(str(result.audio_path))
        finally:
            self._processing = False
