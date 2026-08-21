from __future__ import annotations

"""
Continuous PCM audio mixer for Discord voice channels.

discord.py (Rapptz) ships no audio mixer: ``VoiceClient.play()`` accepts a
single :class:`discord.AudioSource` and raises ``ClientException`` if called
while already playing.  One opus stream per connection, one source feeding it.

This module adds software mixing *upstream* of that single stream.  A
:class:`VoiceMixer` is itself a ``discord.AudioSource`` that discord.py polls
every 20 ms via :meth:`read`.  Internally it sums the 20 ms PCM frames of any
number of child sources, clamps to int16, and returns one blended frame.
discord.py never knows several streams were combined underneath — it just
encodes and sends the single mixed frame.

This gives us, for one voice connection at once:

  * an always-on low-volume **ambient/idle loop** (the "thinking" sound),
  * a **speech** channel (TTS replies, verbal acknowledgements) that plays
    *over* the ambient bed, automatically **ducking** the ambient gain down
    while speech is active and restoring it when speech ends — the smooth
    Grok-voice-mode feel, instead of stop-and-swap.

Design notes
------------
* The mixer is installed **once** per guild on join (``vc.play(mixer)``) and
  runs continuously until the bot leaves.  Children come and go; the mixer
  itself never stops, so there is no ``is_playing()`` race between an
  acknowledgement and the final reply.
* Frame format is Discord-native: 48 kHz, 2 channels, signed 16-bit LE,
  20 ms per frame == ``discord.opus.Encoder.FRAME_SIZE`` bytes
  (3840 = 960 samples * 2 channels * 2 bytes).
* Mixing is a single vectorised int32 add + clip per 20 ms frame (numpy,
  already a core dependency).  CPU cost is negligible.
* :meth:`read` is called from discord.py's audio sender **thread**, while
  children are added/removed from the asyncio event loop thread, so all
  shared state is guarded by a plain ``threading.Lock``.

The mixer NEVER touches the inbound receive path: it only produces the bot's
*outgoing* stream.  The :class:`VoiceReceiver` decodes incoming SSRCs only, so
the mixer's output cannot echo back into transcription.
"""

import logging
import threading
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import discord

try:
    from .ffmpeg_utils import resolve_ffmpeg_executable
except ImportError:
    from ffmpeg_utils import resolve_ffmpeg_executable

if TYPE_CHECKING:  # numpy is an optional ("voice" extra) dep — never import at runtime top-level
    import numpy as np

logger = logging.getLogger(__name__)


def _require_numpy():
    """Import numpy lazily.

    numpy ships in the optional ``voice`` extra, not the base install, so this
    module must import cleanly without it (the Discord adapter imports this
    file unconditionally).  Callers that actually mix audio call this; if the
    voice extra isn't installed they get a clear error instead of a top-level
    ImportError that would break the whole adapter import.
    """
    import numpy as np  # noqa: PLC0415 — intentional lazy import
    return np

# Discord-native frame geometry (matches discord.opus.Encoder).
SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2                       # bytes per sample (s16)
FRAME_LENGTH_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_LENGTH_MS // 1000   # 960
FRAME_SIZE = SAMPLES_PER_FRAME * CHANNELS * SAMPLE_WIDTH    # 3840 bytes
BYTES_PER_MS = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH // 1000  # 192
SILENCE_FRAME = b"\x00" * FRAME_SIZE


class MixerChild:
    """A single audio stream feeding into :class:`VoiceMixer`.

    Wraps raw 48 kHz / stereo / s16le PCM bytes.  ``read_frame`` hands back one
    20 ms frame at a time, optionally looping, with a per-child gain applied.
    """

    __slots__ = (
        "name", "_pcm", "_pos", "loop", "gain",
        "is_speech", "fade_frames", "_fade_done", "_finished",
    )

    def __init__(
        self,
        name: str,
        pcm: bytes,
        *,
        loop: bool = False,
        gain: float = 1.0,
        is_speech: bool = False,
        fade_in_ms: int = 0,
    ):
        # Pad to a whole number of frames so looping is seamless and the final
        # partial frame doesn't click.
        remainder = len(pcm) % FRAME_SIZE
        if remainder:
            pcm = pcm + b"\x00" * (FRAME_SIZE - remainder)
        self.name = name
        self._pcm = pcm
        self._pos = 0
        self.loop = loop
        self.gain = float(gain)
        self.is_speech = is_speech
        # Linear fade-in over N frames avoids a click when a loud child starts.
        self.fade_frames = max(0, fade_in_ms // FRAME_LENGTH_MS)
        self._fade_done = 0
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    def read_frame(self) -> "Optional[np.ndarray]":
        """Return the next 20 ms frame as an int16 ndarray, or None if done."""
        if self._finished:
            return None
        if self._pos >= len(self._pcm):
            if self.loop and self._pcm:
                self._pos = 0
            else:
                self._finished = True
                return None

        np = _require_numpy()
        chunk = self._pcm[self._pos:self._pos + FRAME_SIZE]
        self._pos += FRAME_SIZE
        if len(chunk) < FRAME_SIZE:
            chunk = chunk + b"\x00" * (FRAME_SIZE - len(chunk))

        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)

        gain = self.gain
        if self.fade_frames and self._fade_done < self.fade_frames:
            self._fade_done += 1
            gain *= self._fade_done / self.fade_frames

        if gain != 1.0:
            samples = samples * gain
        return samples


class StreamingMixerChild:
    """Incremental 24 kHz mono s16le speech source for :class:`VoiceMixer`.

    Provider chunks may split int16 samples arbitrarily.  Complete samples are
    converted immediately to Discord-native 48 kHz stereo s16le (an exact 2x
    rate expansion) and retained in a bounded buffer until the audio thread
    drains 20 ms frames.  An empty-but-open stream returns silence rather than
    EOF so discord.py keeps polling while the provider is between chunks.

    Lifecycle: ``write`` while open, then ``finish`` (producer done; buffered
    audio still drains on the sender thread) or ``abort`` (drop everything).
    The one-shot ``on_drained`` callback fires exactly once, when the child
    reaches its terminal state — either after the buffer drains naturally or
    immediately on abort — so owners can release resources (e.g. the voice
    receiver echo guard) only when no more audio can be emitted.
    """

    def __init__(
        self,
        name: str,
        *,
        gain: float = 1.0,
        max_buffer_bytes: int = 16 * 1024 * 1024,
        on_drained: "Optional[Callable[['StreamingMixerChild'], None]]" = None,
        on_audible: "Optional[Callable[['StreamingMixerChild'], None]]" = None,
    ) -> None:
        self.name = name
        self.gain = float(gain)
        self.is_speech = True
        self._max_buffer_bytes = max(FRAME_SIZE, int(max_buffer_bytes))
        self._buffer = bytearray()
        self._input_carry = bytearray()
        self._lock = threading.Lock()
        self._input_finished = False
        self._aborted = False
        self._finished = False
        self._drained_notified = False
        self._audible_notified = False
        self._audible_pending = False
        self._on_drained = on_drained
        self._on_audible = on_audible

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    @property
    def aborted(self) -> bool:
        """True when playback ended by abort rather than a natural drain."""
        with self._lock:
            return self._aborted

    @property
    def audible_pending(self) -> bool:
        """True after a non-silent frame was pulled but not yet send-acked."""
        with self._lock:
            return self._audible_pending

    @property
    def drained(self) -> bool:
        """True when no more frames will ever be emitted and the one-shot
        ``on_drained`` callback has not yet fired (owner may act on it)."""
        with self._lock:
            return self._finished and not self._drained_notified

    def write(self, pcm_24k_mono: bytes) -> bool:
        """Append provider PCM, returning False when this child is terminal."""
        if not pcm_24k_mono:
            return True
        np = _require_numpy()
        with self._lock:
            if self._aborted:
                return False
            if self._input_finished:
                raise RuntimeError("streaming speech is already finished")
            data = bytes(self._input_carry) + bytes(pcm_24k_mono)
            even = len(data) & ~1
            self._input_carry[:] = data[even:]
            if not even:
                return True
            mono = np.frombuffer(data[:even], dtype=np.int16)
            # 24k mono -> 48k stereo: duplicate each sample once in time and
            # duplicate both rate-expanded samples across the two channels.
            converted = np.repeat(np.repeat(mono, 2), 2).astype(np.int16).tobytes()
            if len(self._buffer) + len(converted) > self._max_buffer_bytes:
                raise BufferError("streaming speech buffer limit exceeded")
            self._buffer.extend(converted)
            return True

    def finish(self) -> None:
        with self._lock:
            self._input_carry.clear()  # incomplete int16 sample is not audio
            self._input_finished = True

    def abort(self) -> None:
        self._abort_silent()
        self._notify_drained()

    def _abort_silent(self) -> None:
        """Abort without firing ``on_drained``.

        Used by the mixer itself (``stop_speech`` / ``cleanup``), which
        already owns the aggregate state and must not re-enter it through the
        callback.
        """
        with self._lock:
            self._aborted = True
            self._input_finished = True
            self._input_carry.clear()
            self._buffer.clear()
            self._finished = True

    def _notify_drained(self) -> None:
        with self._lock:
            if self._drained_notified:
                return
            self._drained_notified = True
            cb = self._on_drained
        if cb is not None:
            cb(self)

    def fire_drained_cb(self) -> None:
        """Fire the one-shot drained callback if the child is terminal.

        Called by the mixer AFTER releasing its own lock, so the callback can
        safely re-enter the mixer or run sender-thread side effects.
        """
        with self._lock:
            if not self._finished or self._drained_notified:
                return
            self._drained_notified = True
            cb = self._on_drained
        if cb is not None:
            cb(self)

    def fire_audible_cb(self) -> None:
        """Fire sender-confirmed audibility after the mixer lock is released."""
        with self._lock:
            if not self._audible_pending:
                return
            self._audible_pending = False
            cb = self._on_audible
        if cb is not None:
            cb(self)

    def read_frame(self) -> "Optional[np.ndarray]":
        np = _require_numpy()
        with self._lock:
            if self._finished or self._aborted:
                return None
            if len(self._buffer) >= FRAME_SIZE:
                chunk = bytes(self._buffer[:FRAME_SIZE])
                del self._buffer[:FRAME_SIZE]
            elif self._input_finished:
                if not self._buffer:
                    # Natural drain: terminal state.  The callback fires via
                    # fire_drained_cb() (called by the mixer after releasing
                    # its lock) — never inline, to avoid lock re-entrancy.
                    self._finished = True
                    return None
                chunk = bytes(self._buffer)
                self._buffer.clear()
                chunk += b"\x00" * (FRAME_SIZE - len(chunk))
            else:
                chunk = SILENCE_FRAME
            if not self._audible_notified and any(chunk):
                self._audible_notified = True
                self._audible_pending = True
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        if self.gain != 1.0:
            samples *= self.gain
        return samples


class VoiceMixer(discord.AudioSource):
    """A continuous ``discord.AudioSource`` that mixes N child streams.

    Use :meth:`set_ambient` to install/replace the looping idle bed and
    :meth:`play_speech` to layer a one-shot clip over it (ducking the ambient
    while it plays).  Both are safe to call from the asyncio loop thread while
    discord.py drains :meth:`read` from its sender thread.
    """

    # discord.AudioSource subclasses set is_opus()==False to receive PCM.
    def is_opus(self) -> bool:  # pragma: no cover - trivial
        return False

    def __init__(
        self,
        *,
        ambient_gain: float = 0.18,
        duck_gain: float = 0.06,
        speech_gain: float = 1.0,
        duck_release_ms: int = 400,
    ):
        self._lock = threading.Lock()
        self._ambient: Optional[MixerChild] = None
        self._speech: List[Union[MixerChild, StreamingMixerChild]] = []
        self._ambient_gain = float(ambient_gain)
        self._duck_gain = float(duck_gain)
        self._speech_gain = float(speech_gain)
        # When speech ends, ramp the ambient back up over this many frames
        # instead of jumping, so the bed swells back smoothly.
        self._duck_release_frames = max(1, duck_release_ms // FRAME_LENGTH_MS)
        self._duck_release_left = 0
        self._closed = False
        # Tracks whether speech is currently active, for external callers that
        # want to avoid double-ducking or know when a reply is mid-flight.
        self._speech_active = False

    # ------------------------------------------------------------------
    # Ambient (idle / "thinking") bed
    # ------------------------------------------------------------------

    def set_ambient(self, pcm: Optional[bytes], *, gain: Optional[float] = None) -> None:
        """Install (or clear, with ``pcm=None``) the looping ambient bed."""
        with self._lock:
            if gain is not None:
                self._ambient_gain = float(gain)
            if not pcm:
                self._ambient = None
                return
            self._ambient = MixerChild(
                "ambient", pcm, loop=True,
                gain=self._effective_ambient_gain(), fade_in_ms=200,
            )

    def _effective_ambient_gain(self) -> float:
        return self._duck_gain if self._speech_active else self._ambient_gain

    # ------------------------------------------------------------------
    # Speech (TTS replies, verbal acks) layered over the ambient bed
    # ------------------------------------------------------------------

    def play_speech(self, pcm: bytes, *, gain: Optional[float] = None,
                    fade_in_ms: int = 40) -> Optional[MixerChild]:
        """Layer a one-shot clip and return its independently trackable child."""
        if not pcm:
            return None
        with self._lock:
            child = MixerChild(
                "speech", pcm, loop=False,
                gain=self._speech_gain if gain is None else float(gain),
                is_speech=True, fade_in_ms=fade_in_ms,
            )
            self._speech.append(child)
            self._speech_active = True
            self._duck_release_left = 0
            if self._ambient is not None:
                self._ambient.gain = self._duck_gain
            return child

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def begin_streaming_speech(
        self,
        *,
        gain: Optional[float] = None,
        max_buffer_bytes: int = 16 * 1024 * 1024,
        on_drained: "Optional[Callable[[StreamingMixerChild], None]]" = None,
        on_audible: "Optional[Callable[[StreamingMixerChild], None]]" = None,
    ) -> StreamingMixerChild:
        """Attach and return an open incremental speech child.

        ``on_drained`` fires once when the child reaches its terminal state
        (natural drain or abort), after the mixer has removed the child and
        released the duck if it was the last speech source.
        """
        def _child_drained(child: StreamingMixerChild) -> None:
            self._streaming_child_drained(child)
            if on_drained is not None:
                on_drained(child)

        child = StreamingMixerChild(
            "streaming-speech",
            gain=self._speech_gain if gain is None else float(gain),
            max_buffer_bytes=max_buffer_bytes,
            on_drained=_child_drained,
            on_audible=on_audible,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("voice mixer is closed")
            self._speech.append(child)
            self._speech_active = True
            self._duck_release_left = 0
            if self._ambient is not None:
                self._ambient.gain = self._duck_gain
        return child

    def _streaming_child_drained(self, child: StreamingMixerChild) -> None:
        """Remove *child* and release the duck when no speech children remain.

        Runs from the sender thread (natural drain) or the producer thread
        (abort); always under the mixer lock, never re-entrant.
        """
        with self._lock:
            try:
                self._speech.remove(child)
            except ValueError:
                pass
            if not self._speech and self._speech_active:
                self._begin_duck_release_locked()

    @property
    def speech_active(self) -> bool:
        with self._lock:
            return self._speech_active

    def stop_speech_child(
        self, child: Union[MixerChild, StreamingMixerChild]
    ) -> None:
        """Stop only *child*, preserving unrelated speech in the same guild."""
        drained: Optional[StreamingMixerChild] = None
        with self._lock:
            try:
                self._speech.remove(child)
            except ValueError:
                return
            if isinstance(child, StreamingMixerChild):
                child._abort_silent()
                drained = child
            else:
                child._finished = True
            if not self._speech and self._speech_active:
                self._begin_duck_release_locked()
        if drained is not None:
            drained.fire_drained_cb()

    def stop_speech(self) -> None:
        """Drop every in-flight speech source and notify streaming owners."""
        drained: List[StreamingMixerChild] = []
        with self._lock:
            for child in self._speech:
                if isinstance(child, StreamingMixerChild):
                    child._abort_silent()
                    drained.append(child)
                else:
                    child._finished = True
            self._speech.clear()
            self._begin_duck_release_locked()
        for child in drained:
            child.fire_drained_cb()

    def _begin_duck_release_locked(self) -> None:
        self._speech_active = False
        self._duck_release_left = self._duck_release_frames

    # ------------------------------------------------------------------
    # AudioSource interface — called from discord.py's sender thread
    # ------------------------------------------------------------------

    def read(self) -> bytes:
        """Return one 20 ms mixed PCM frame (always FRAME_SIZE bytes).

        Returning a non-empty frame keeps discord.py's player alive; we never
        return b"" because that would stop the single underlying stream and we
        want the mixer to run continuously for the lifetime of the connection.
        """
        with self._lock:
            if self._closed:
                return SILENCE_FRAME

            np = _require_numpy()
            acc: "Optional[np.ndarray]" = None
            drained: List[StreamingMixerChild] = []
            # A second source.read() proves discord.py successfully returned
            # from send_audio_packet() for the preceding frame. A pending child
            # is acknowledged now, before this call produces the next frame.
            audible: List[StreamingMixerChild] = [
                child
                for child in self._speech
                if (
                    isinstance(child, StreamingMixerChild)
                    and child.audible_pending
                )
            ]

            # Speech children (drop exhausted ones; release duck when last ends)
            if self._speech:
                still_live: List[Union[MixerChild, StreamingMixerChild]] = []
                for child in self._speech:
                    frame = child.read_frame()
                    if frame is None:
                        # Terminal streaming children fire their drained
                        # callback AFTER this lock is released.
                        if isinstance(child, StreamingMixerChild) and child.drained:
                            drained.append(child)
                        continue
                    acc = frame if acc is None else acc + frame
                    still_live.append(child)
                self._speech = still_live
                if not self._speech and self._speech_active:
                    self._begin_duck_release_locked()

            # Ambient bed — ramp gain back up during duck-release.
            if self._ambient is not None:
                if self._duck_release_left > 0 and not self._speech_active:
                    self._duck_release_left -= 1
                    frac = 1.0 - (self._duck_release_left / self._duck_release_frames)
                    self._ambient.gain = (
                        self._duck_gain
                        + (self._ambient_gain - self._duck_gain) * frac
                    )
                elif not self._speech_active and self._duck_release_left == 0:
                    self._ambient.gain = self._ambient_gain
                amb = self._ambient.read_frame()
                if amb is not None:
                    acc = amb if acc is None else acc + amb

            if acc is None:
                frame_out = SILENCE_FRAME
            else:
                np.clip(acc, -32768, 32767, out=acc)
                frame_out = acc.astype(np.int16).tobytes()

        # Post-send audible acknowledgement and drained callbacks run outside
        # lock because adapter callbacks may re-enter mixer lifecycle methods.
        for child in audible:
            child.fire_audible_cb()
        for child in drained:
            child.fire_drained_cb()
        return frame_out

    def cleanup(self) -> None:  # called by discord.py when playback stops
        drained: List[StreamingMixerChild] = []
        with self._lock:
            self._closed = True
            self._ambient = None
            for child in self._speech:
                if isinstance(child, StreamingMixerChild):
                    child._abort_silent()
                    drained.append(child)
            self._speech.clear()
            self._speech_active = False
        # Notify owners outside the mixer lock so adapter callbacks may safely
        # schedule cleanup/fallback work.
        for child in drained:
            child.fire_drained_cb()


# ----------------------------------------------------------------------
# PCM helpers
# ----------------------------------------------------------------------

def decode_to_pcm(path: str, *, timeout: float = 30.0) -> Optional[bytes]:
    """Decode any audio file to 48 kHz / stereo / s16le PCM via ffmpeg.

    Returns the raw PCM bytes, or None on failure.  ffmpeg is already a hard
    requirement of the voice path (see ``VoiceReceiver.pcm_to_wav``).
    """
    import subprocess

    try:
        proc = subprocess.run(
            [
                resolve_ffmpeg_executable(), "-y", "-loglevel", "error",
                "-i", path,
                "-f", "s16le",
                "-ar", str(SAMPLE_RATE),
                "-ac", str(CHANNELS),
                "pipe:1",
            ],
            capture_output=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("decode_to_pcm failed for %s: %s", path, e)
        return None
    if proc.returncode != 0:
        logger.warning(
            "ffmpeg decode failed for %s (rc=%d): %s",
            path, proc.returncode, (proc.stderr or b"").decode("utf-8", "replace")[:200],
        )
        return None
    return proc.stdout or None


def synth_ambient_pcm(seconds: float = 4.0) -> bytes:
    """Synthesise a subtle looping ambient bed (no asset file required).

    A soft, slowly-pulsing low pad: two detuned sine partials with a gentle
    tremolo, plus a touch of filtered noise.  Designed to loop seamlessly
    (whole number of cycles, zero-crossing endpoints) and sit quietly under
    speech.  Mono content duplicated to stereo.
    """
    np = _require_numpy()
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE

    # Choose base frequencies that complete whole cycles over the loop so the
    # wrap point is click-free.
    def _whole_cycle_freq(target: float) -> float:
        cycles = max(1, round(target * seconds))
        return cycles / seconds

    f1 = _whole_cycle_freq(110.0)
    f2 = _whole_cycle_freq(110.5)
    trem = _whole_cycle_freq(0.5)   # ~0.5 Hz tremolo

    pad = (
        0.55 * np.sin(2 * np.pi * f1 * t)
        + 0.45 * np.sin(2 * np.pi * f2 * t)
    )
    tremolo = 0.6 + 0.4 * (0.5 * (1 + np.sin(2 * np.pi * trem * t)))
    signal = pad * tremolo

    # Smooth filtered noise for air, kept very low.
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(n)
    kernel = np.ones(64) / 64.0
    noise = np.convolve(noise, kernel, mode="same")
    signal = signal + 0.08 * noise

    # Normalise to a modest peak (mixer applies the real ambient gain on top).
    peak = float(np.max(np.abs(signal))) or 1.0
    signal = (signal / peak) * 0.5

    mono16 = (signal * 32767.0).astype(np.int16)
    stereo16 = np.repeat(mono16[:, None], CHANNELS, axis=1).reshape(-1)
    return stereo16.tobytes()
