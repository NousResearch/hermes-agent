"""Outbound audio for MatrixRTC: publish one microphone track, feed it PCM.

The mirror image of ``receiver.py`` over the same connected ``rtc.Room``, and just as
deliberately dumb — it owns no policy about *when* to speak. Deciding that, and the
gateway's streaming-TTS contract, are ``outbound.py``'s job.

Three things this module exists to get right:

* **The track outlives the turn.** ``start()`` publishes once per call; every reply writes
  into the same ``AudioSource``. Publishing per turn would flash a join/leave in every
  client's call UI for each sentence the bot says.
* **Resample with the SDK's own sox binding, not by hand.** The gateway's streaming-TTS
  contract is 24 kHz (``AudioFormat``); LiveKit carries 48 kHz. ``rtc.AudioResampler`` is in
  the wheel we already depend on, is stateful across chunks (so clause boundaries do not
  click), and keeps the conversion off the Python side of the FFI.
* **Sleep after the source is gone.** Same tokio trap as ``receiver.close()``: the FFI
  worker is still draining when the handle is disposed, and letting the loop close over it
  aborts the process *after a completely successful run*.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .receiver import FFI_DRAIN_DELAY

logger = logging.getLogger(__name__)

# What LiveKit carries. The gateway's AudioFormat default (24 kHz) is the other end.
LIVEKIT_SAMPLE_RATE = 48000
SAMPLE_WIDTH = 2  # s16; the only width AudioFrame accepts

# Frames handed to capture_frame(). capture_frame paces itself against the source's queue,
# so a whole file arrives over its own duration rather than in one unbounded push.
FRAME_DURATION = 0.02
TRACK_NAME = "hermes"


class MatrixRTCPublisher:
    """One published microphone track on an already-connected LiveKit room.

    *room* is the live ``rtc.Room`` — the same object ``MatrixRTCReceiver`` joined with, so
    the bot speaks and listens as one participant rather than joining the call twice.
    *sample_rate* is the rate the **caller** writes at (the gateway contract's, not the
    wire's): ``write`` resamples to 48 kHz, ``write_native`` assumes 48 kHz already.
    """

    def __init__(self, room: Any, *, sample_rate: int = 24000, channels: int = 1):
        self._room = room
        self.sample_rate = sample_rate
        self.channels = channels
        self._rtc: Any = None  # the livekit.rtc module, kept for AudioFrame construction
        self._source: Any = None
        self._track: Any = None
        self._publication: Any = None
        self._resampler: Any = None
        self._speaking = False

    live = property(lambda self: self._source is not None)
    # True from the first captured frame until the queue has played out or been dropped.
    # Inbound audio in that window is our own voice coming back, so the receiver stops
    # transcribing it; sustained speech through it is the user talking over us.
    speaking = property(lambda self: self._source is not None and self._speaking)

    # --- lifecycle ---

    async def start(self) -> None:
        """Publish the track. Idempotent: a second call on a live publisher does nothing."""
        if self._source is not None:
            return
        from livekit import rtc

        self._rtc = rtc
        self._source = rtc.AudioSource(LIVEKIT_SAMPLE_RATE, self.channels)
        if self.sample_rate != LIVEKIT_SAMPLE_RATE:
            self._resampler = rtc.AudioResampler(
                self.sample_rate, LIVEKIT_SAMPLE_RATE, num_channels=self.channels)
        self._track = rtc.LocalAudioTrack.create_audio_track(TRACK_NAME, self._source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        self._publication = await self._room.local_participant.publish_track(
            self._track, options)
        logger.info("MatrixRTC: publishing %d Hz audio as %r", LIVEKIT_SAMPLE_RATE, TRACK_NAME)

    async def close(self) -> None:
        """Unpublish, drop the source, and let the FFI drain. Safe to call twice."""
        source, self._source = self._source, None
        self._resampler = self._track = None
        self._speaking = False
        if self._publication is not None and (sid := getattr(self._publication, "sid", None)):
            try:
                await self._room.local_participant.unpublish_track(sid)
            except Exception as exc:
                logger.debug("MatrixRTC: unpublish failed: %s", exc)
        self._publication = None
        if source is not None:
            await source.aclose()
            # Not cosmetic: without this the process aborts on a successful run.
            await asyncio.sleep(FFI_DRAIN_DELAY)

    # --- audio ---

    async def write(self, pcm: bytes) -> None:
        """Write PCM at the contract rate, resampling to what LiveKit carries."""
        if self._source is None or not pcm:
            return
        if self._resampler is None:
            await self.write_native(pcm)
            return
        # bytearray, not bytes: push() only treats a bytearray as raw PCM — anything else it
        # assumes is an AudioFrame and dereferences as one.
        for frame in self._resampler.push(bytearray(pcm)):
            await self._capture(frame)

    async def write_native(self, pcm: bytes) -> None:
        """Write PCM that is already at ``LIVEKIT_SAMPLE_RATE`` — the whole-file path."""
        if self._source is None or not pcm:
            return
        stride = int(LIVEKIT_SAMPLE_RATE * FRAME_DURATION) * self.channels * SAMPLE_WIDTH
        for start in range(0, len(pcm), stride):
            await self._capture(self._frame(pcm[start:start + stride]))

    async def flush(self) -> None:
        """End of a streamed utterance: push the resampler's tail, then play the queue out."""
        if self._source is None:
            return
        if self._resampler is not None:
            for frame in self._resampler.flush():
                await self._capture(frame)
        await self.drain()

    async def drain(self) -> None:
        """Wait for queued audio to finish playing, leaving the resampler alone.

        The whole-file path wants this and not ``flush``: the resampler's state belongs to
        whichever streamed turn is mid-sentence, and draining its tail here would splice a
        fragment of that turn onto the end of the file.
        """
        if self._source is not None:
            await self._source.wait_for_playout()
            self._speaking = False

    def clear(self) -> None:
        """Drop audio queued but not yet heard (abort / barge-in)."""
        self._speaking = False
        if self._source is not None:
            self._source.clear_queue()

    # --- internals ---

    def _frame(self, pcm: bytes) -> Any:
        samples = len(pcm) // (self.channels * SAMPLE_WIDTH)
        return self._rtc.AudioFrame(pcm, LIVEKIT_SAMPLE_RATE, self.channels, samples)

    async def _capture(self, frame: Any) -> None:
        if (source := self._source) is not None:
            self._speaking = True
            await source.capture_frame(frame)
