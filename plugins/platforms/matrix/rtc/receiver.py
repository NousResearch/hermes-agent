"""LiveKit room lifecycle for MatrixRTC: join, subscribe, hear the user.

Inbound half of the duplex. Publishing TTS, mapping utterances onto a gateway session,
and writing ``m.rtc.member`` so the bot shows up in a client's call UI are each their own
concern and are not wired here — but *whether we are currently talking* is passed in, because
a room hears what the bot says and the only sane place to drop that is before it is buffered.

Two traps this module exists to encapsulate, both established against a live SFU:

* **Ask the SDK for 16 kHz mono.** The wire carries 48 kHz; ``AudioStream`` takes a
  ``sample_rate`` that reaches the Rust FFI, so the native resampler hands us exactly
  what Whisper wants. Resampling in Python — or shelling out to ffmpeg as the Discord
  receiver must — buys nothing here.
* **Sleep 0.5 s after ``disconnect()``.** The FFI's tokio worker is still draining when
  ``disconnect()`` returns; letting the event loop close over it aborts the process with
  a non-unwinding panic *after a completely successful run*, so the exit code lies.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from .segmenter import (
    CHANNELS, SAMPLE_RATE, SPEECH_RMS, TurnSegmenter, _positive_float, _rtc_config,
    pcm_duration, pcm_rms, transcribe_pcm)

logger = logging.getLogger(__name__)

LAZY_FEATURE = "platform.matrix_rtc"

# How often we ask the segmenter whether anyone has stopped talking. Matches the
# Discord voice loop; well under SILENCE_THRESHOLD, so the poll never sets the latency.
POLL_INTERVAL = 0.2

# The FFI worker outlives disconnect() by a hair. See the module docstring.
FFI_DRAIN_DELAY = 0.5

# Barge-in: how much unbroken inbound speech, heard while we are the one talking, counts as
# the user cutting us off rather than our own voice coming back. Short enough to feel like an
# interruption, long enough that a door or a keyboard does not stop a reply mid-word.
BARGE_IN_DURATION = 0.3
# RMS floor a frame must clear to count towards that. It is the segmenter's own speech floor:
# one definition of "somebody is talking" for both halves of the duplex, so a room quiet
# enough to end a turn cannot simultaneously be loud enough to interrupt a reply.
BARGE_IN_RMS = SPEECH_RMS


def livekit_available() -> bool:
    """True when the LiveKit SDK can be imported (or its lazy feature is satisfied)."""
    try:
        from tools.lazy_deps import is_available
        return is_available(LAZY_FEATURE)
    except Exception:
        try:
            import livekit.rtc  # noqa: F401
            return True
        except ImportError:
            return False


class MatrixRTCReceiver:
    """Joins a LiveKit room and calls *on_transcript* once per completed utterance.

    ``on_transcript(identity, transcript)`` is awaited on the receiver's own loop;
    ``identity`` is the LiveKit participant identity, which the Matrix JWT service
    derives as ``{matrix_user_id}:{device_id}``.

    *is_authorized(identity)* is consulted once per utterance *before* transcription, so
    audio from a participant the operator never allowed is never sent to Whisper at all.
    Omitting it transcribes every speaker and leaves the allowlist entirely to the caller.

    *is_speaking()* is the echo gate: while it is true the bot's own voice is in the room, so
    inbound audio is discarded instead of buffered — otherwise the reply is transcribed back
    as if the user had said it, and the bot answers itself. Speech that keeps coming through
    that gate is the user talking over the reply, and calls *on_barge_in(identity)* once.
    Without either callable the receiver behaves exactly as it did before: everything heard
    is transcribed.
    """

    def __init__(self, on_transcript: Callable[[str, str], Awaitable[None]],
                 segmenter: Optional[TurnSegmenter] = None,
                 sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS,
                 is_authorized: Optional[Callable[[str], bool]] = None,
                 is_speaking: Optional[Callable[[], bool]] = None,
                 on_barge_in: Optional[Callable[[str], Awaitable[None]]] = None):
        cfg = _rtc_config()
        self._on_transcript = on_transcript
        self._is_authorized = is_authorized
        self._is_speaking = is_speaking
        self._on_barge_in = on_barge_in
        self.sample_rate = sample_rate
        self.channels = channels
        self.barge_in_duration = _positive_float(
            cfg.get("barge_in_duration"), BARGE_IN_DURATION)
        self.barge_in_rms = _positive_float(cfg.get("barge_in_rms"), BARGE_IN_RMS)
        self.segmenter = segmenter or TurnSegmenter(sample_rate, channels)
        self._interrupting = 0.0  # unbroken seconds of speech heard while we talk
        self._room: Any = None
        self._tasks: set[asyncio.Task] = set()
        self._poll_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def room(self) -> Any:
        """The connected LiveKit room, or None before ``connect`` / after ``close``.

        The outbound publisher adds its track to *this* connection: a call is one room
        membership with a microphone, not two.
        """
        return self._room

    # --- lifecycle ---

    async def connect(self, sfu_url: str, jwt: str) -> None:
        """Join the SFU and start listening. *jwt* comes from ``focus.fetch_livekit_credentials``."""
        from tools.lazy_deps import ensure
        await asyncio.to_thread(ensure, LAZY_FEATURE, prompt=False)
        from livekit import rtc

        room = rtc.Room()

        @room.on("track_subscribed")
        def _on_track(track, publication, participant):  # noqa: ARG001 - SDK signature
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            logger.info("MatrixRTC: subscribed to audio from %s", participant.identity)
            self._spawn(self._drain_track(rtc, track, participant.identity))

        await room.connect(sfu_url, jwt, options=rtc.RoomOptions(auto_subscribe=True))
        self._room = room
        self._running = True
        logger.info("MatrixRTC: joined as %s", room.local_participant.identity)
        self._poll_task = asyncio.create_task(self._poll_silence())

    async def close(self) -> None:
        """Leave the room, emit whatever was still buffered, and let the FFI drain."""
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            await asyncio.gather(self._poll_task, return_exceptions=True)
            self._poll_task = None
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
        await self._emit(self.segmenter.flush_pending())
        if self._room is not None:
            await self._room.disconnect()
            self._room = None
            # Not cosmetic: without this the process aborts on a successful run.
            await asyncio.sleep(FFI_DRAIN_DELAY)

    # --- internals ---

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _drain_track(self, rtc, track, identity: str) -> None:
        """Feed one remote track's PCM into the segmenter until it ends."""
        stream = rtc.AudioStream(
            track, sample_rate=self.sample_rate, num_channels=self.channels)
        try:
            async for event in stream:
                pcm = bytes(event.frame.data)
                if self._is_speaking is not None and self._is_speaking():
                    await self._hear_through_our_own_voice(identity, pcm)
                    continue
                self._interrupting = 0.0
                self.segmenter.feed(identity, pcm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("MatrixRTC: audio stream for %s ended: %s", identity, exc)
        finally:
            await stream.aclose()

    async def _hear_through_our_own_voice(self, identity: str, pcm: bytes) -> None:
        """Handle one frame that arrived while we were speaking. The frame is never buffered.

        Dropping it is the echo fix — whether it reached us off the user's speakers or straight
        back off the SFU, it is our own reply, and transcribing it makes the bot answer itself.
        Loud audio that keeps arriving anyway is the user interrupting, which is worth exactly
        one callback: the reply it aborts is what closes this gate again.
        """
        if self._on_barge_in is None:
            return
        if pcm_rms(pcm) < self.barge_in_rms:
            self._interrupting = 0.0  # a gap: whatever came before was not an interruption
            return
        was_below = self._interrupting < self.barge_in_duration
        self._interrupting += pcm_duration(pcm, self.sample_rate, self.channels)
        if not was_below or self._interrupting < self.barge_in_duration:
            return
        logger.info("MatrixRTC: %s spoke over us, interrupting", identity)
        try:
            await self._on_barge_in(identity)
        except Exception:
            logger.error("MatrixRTC: barge-in callback failed", exc_info=True)

    async def _poll_silence(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(POLL_INTERVAL)
                await self._emit(self.segmenter.check_silence())
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("MatrixRTC: silence poll loop failed", exc_info=True)

    async def _emit(self, utterances) -> None:
        for identity, pcm in utterances:
            if self._is_authorized is not None and not self._is_authorized(identity):
                logger.info("MatrixRTC: discarding audio from %s before transcription", identity)
                continue
            try:
                transcript = await asyncio.to_thread(
                    transcribe_pcm, pcm, self.sample_rate, self.channels)
            except Exception as exc:
                logger.warning("MatrixRTC: transcription failed for %s: %s", identity, exc)
                continue
            if not transcript:
                continue
            logger.info("MatrixRTC voice input from %s: %s", identity, transcript[:100])
            try:
                await self._on_transcript(identity, transcript)
            except Exception:
                logger.error("MatrixRTC: transcript callback failed", exc_info=True)
