"""The gateway's streaming-TTS contract, spoken into a MatrixRTC call.

``gateway/platforms/base.py`` declares five hooks a voice-capable adapter overrides so the
gateway can hand it PCM *while the model is still generating*
(``supports_/begin_/write_/finish_/abort_streaming_tts``); the defaults decline and the
gateway falls back to synthesising the whole reply to a file. This mixin implements both
halves for Matrix: the streaming path over ``MatrixRTCPublisher``, and
``play_in_voice_channel`` for the whole-file path when streaming declined or failed.

It is a mixin rather than more methods on ``adapter.py`` because that file is already 3 kLOC
and AGENTS.md is explicit that new behaviour goes in a topical sibling. All state is
``getattr``-guarded: adapter instances built with ``object.__new__`` (the house test
pattern) never run ``__init__``.

Nothing here joins a call. ``start_rtc_audio`` is the seam ``/voice join`` calls once it has
a connected room, which is Phase 4's wiring.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, Optional

from gateway.platforms.base import AudioFormat, SendResult, StreamingTTSHandle, _lazy_attr

from .publisher import LIVEKIT_SAMPLE_RATE, SAMPLE_WIDTH, MatrixRTCPublisher

logger = logging.getLogger(__name__)

# A reply the model rambles through still has to end; ffmpeg is decoding, not transcoding.
DECODE_TIMEOUT = 60


def decode_to_livekit_pcm(path: str, channels: int = 1) -> bytes:
    """Decode any audio file to raw s16le PCM at LiveKit's rate. Empty when ffmpeg is absent.

    Blocking subprocess work — call via ``asyncio.to_thread``. Decoding straight to 48 kHz
    (rather than to the contract's 24 kHz) keeps the whole-file path off the streaming
    resampler, whose state belongs to whichever turn is mid-sentence.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not os.path.isfile(path):
        return b""
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "s16le", "-acodec", "pcm_s16le",
         "-ar", str(LIVEKIT_SAMPLE_RATE), "-ac", str(channels), "pipe:1"],
        capture_output=True, timeout=DECODE_TIMEOUT, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        logger.warning("MatrixRTC: ffmpeg could not decode %s", path)
        return b""
    return result.stdout


class MatrixRTCOutboundMixin:
    """Speaking half of a MatrixRTC call. Mixed into ``MatrixAdapter``."""

    # --- publisher registry (the /voice join seam) ---

    @property
    def rtc_publishers(self) -> Dict[str, MatrixRTCPublisher]:
        """room id -> the publisher currently speaking into that room's call."""
        return _lazy_attr(self, "_rtc_publishers", dict)

    @property
    def _rtc_stream_owners(self) -> Dict[str, StreamingTTSHandle]:
        """room id -> the handle allowed to write right now. See ``begin_streaming_tts``."""
        return _lazy_attr(self, "_rtc_stream_owner_map", dict)

    async def start_rtc_audio(self, room_id: str, room: Any, *,
                              sample_rate: int = AudioFormat.sample_rate,
                              channels: int = AudioFormat.channels) -> MatrixRTCPublisher:
        """Publish an outbound track on *room*, the connected LiveKit room for *room_id*.

        Idempotent per room: joining a call twice reuses the publication rather than
        stacking a second microphone on the participant.
        """
        if (existing := self.rtc_publishers.get(room_id)) is not None:
            return existing
        publisher = MatrixRTCPublisher(room, sample_rate=sample_rate, channels=channels)
        await publisher.start()
        self.rtc_publishers[room_id] = publisher
        return publisher

    async def stop_rtc_audio(self, room_id: str) -> None:
        """Tear the publication down on leave. Later writes are dropped, not queued."""
        self._rtc_stream_owners.pop(room_id, None)
        if (publisher := self.rtc_publishers.pop(room_id, None)) is not None:
            await publisher.close()

    def is_in_voice_channel(self, room_id: str) -> bool:
        """True while this room has a live call we can speak into."""
        publisher = self.rtc_publishers.get(room_id)
        return publisher is not None and publisher.live

    def is_speaking_in(self, room_id: str) -> bool:
        """True while our own voice is still playing in *room_id* — the echo window.

        The receiver's gate, not a report about the humans on the call: Discord's
        ``get_voice_channel_info`` puts an ``is_speaking`` flag on each *member*, which is a
        different question and deliberately not what this answers.
        """
        publisher = self.rtc_publishers.get(room_id)
        return publisher is not None and publisher.speaking

    # --- streaming TTS contract ---

    def supports_streaming_tts(self, chat_id: str, audio_format: AudioFormat) -> bool:
        """Streamable when the room has a live call and the format matches the publication.

        Width is checked because ``AudioFrame`` is s16-only, and channels because the
        published track was created with a fixed count. The *rate* is not: it is data the
        resampler is constructed from, and ``begin_streaming_tts`` is where a mismatch with
        the already-built resampler is caught.
        """
        if not self.is_in_voice_channel(chat_id):
            return False
        return (audio_format.sample_width == SAMPLE_WIDTH
                and audio_format.channels == self.rtc_publishers[chat_id].channels)

    async def begin_streaming_tts(
        self, chat_id: str, audio_format: AudioFormat,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[StreamingTTSHandle]:
        """Take the room's floor for one turn, or decline so the gateway falls back.

        A room has one outbound track, so two overlapping turns cannot both write into it —
        their clauses would interleave, and they share the resampler's state. The newest
        turn wins and the superseded handle's later writes are dropped, which is the
        contract's own "late chunks are silently dropped" rule one level up.
        """
        publisher = self.rtc_publishers.get(chat_id)
        if publisher is None or not publisher.live:
            return None
        if publisher.sample_rate != audio_format.sample_rate:
            logger.debug("MatrixRTC: %s publishes at %d Hz, TTS offered %d Hz",
                         chat_id, publisher.sample_rate, audio_format.sample_rate)
            return None
        handle = StreamingTTSHandle(chat_id=chat_id, audio_format=audio_format)
        self._rtc_stream_owners[chat_id] = handle
        return handle

    async def write_streaming_tts(self, handle: StreamingTTSHandle, chunk: bytes) -> None:
        """Write one contract-rate PCM chunk to the room's track."""
        if (publisher := self._owned_publisher(handle)) is not None:
            await publisher.write(chunk)

    async def finish_streaming_tts(self, handle: StreamingTTSHandle, *,
                                   interrupted: bool = False) -> None:
        """End of the reply: play the tail out, unless the turn was cut short."""
        publisher = self._owned_publisher(handle)
        self._rtc_stream_owners.pop(handle.chat_id, None)
        if publisher is None:
            return
        if interrupted:
            publisher.clear()
            return
        await publisher.flush()

    async def abort_streaming_tts(self, handle: StreamingTTSHandle,
                                  error: Optional[str] = None) -> None:
        """Drop what has not been heard yet and release the floor. Idempotent."""
        publisher = self._owned_publisher(handle)
        self._rtc_stream_owners.pop(handle.chat_id, None)
        handle.aborted = True
        if publisher is not None:
            logger.info("MatrixRTC: aborting speech in %s: %s",
                        handle.chat_id, error or "cancelled")
            publisher.clear()

    def _owned_publisher(self, handle: StreamingTTSHandle) -> Optional[MatrixRTCPublisher]:
        """The publisher *handle* still holds the floor on, else None."""
        if handle.aborted or self._rtc_stream_owners.get(handle.chat_id) is not handle:
            return None
        return self.rtc_publishers.get(handle.chat_id)

    # --- whole-file fallback ---

    async def play_in_voice_channel(self, room_id: str, audio_path: str) -> bool:
        """Play a finished audio file into the room's call. False when there is no call.

        The non-streaming path: the gateway synthesised the whole reply to a file because
        streaming declined, failed before anything was audible, or was never configured.
        """
        publisher = self.rtc_publishers.get(room_id)
        if publisher is None or not publisher.live:
            return False
        try:
            pcm = await asyncio.to_thread(decode_to_livekit_pcm, audio_path, publisher.channels)
            if not pcm:
                return False
            await publisher.write_native(pcm)
            await publisher.drain()
        except Exception as exc:
            logger.warning("MatrixRTC: playback of %s failed: %s", audio_path, exc)
            return False
        return True

    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        """Speak into the room's call when one is live, else send an MSC3245 voice message."""
        if self.is_in_voice_channel(chat_id):
            return SendResult(success=await self.play_in_voice_channel(chat_id, audio_path))
        return await super().play_tts(chat_id=chat_id, audio_path=audio_path, **kwargs)
