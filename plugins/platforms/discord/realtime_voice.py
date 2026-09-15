"""Audio bridges between Discord voice channels and the xAI realtime session.

The realtime backend (:mod:`tools.voice_realtime`) is transport-agnostic: it
takes a ``mic_factory`` (frames in) and a ``playout_sink_factory`` (speech
out). This module supplies the Discord implementations:

* :class:`DiscordMicBridge` — fed continuously by the adapter's voice-receive
  drain (48 kHz stereo, Discord-native), sums simultaneous speakers into one
  signal and downsamples to the session's 16 kHz mono input format.
* :class:`MixerPlayoutSink` — receives the supervisor's 24 kHz mono speech,
  upsamples to 48 kHz stereo, and streams it through the guild's continuous
  :class:`~voice_mixer.VoiceMixer` so it ducks the ambient bed and mixes over
  it like any other speech.

Sample-rate notes: 48 kHz / 16 kHz is an exact 3:1 decimation (mix L/R
to 48 kHz mono, then average each 3 samples to 16 kHz). Playback is a
linear 24 kHz → 48 kHz upsample (insert the midpoint between samples)
then dual-mono stereo. Both stay integer-ratio numpy resamples — no
scipy/ffmpeg per frame.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

try:
    from .voice_mixer import StreamSpeechChild, _require_numpy
except ImportError:
    from voice_mixer import StreamSpeechChild, _require_numpy

# 3 stereo int16 frames (L,R × 3) collapse into one mono output sample.
_DOWNSAMPLE_GROUP_BYTES = 12


def mix_pcm16(chunks: Sequence[bytes]) -> bytes:
    """Sum int16 PCM chunks sample-wise into one signal (saturating).

    The chunks are one per speaker for the *same* drain window, so they are
    aligned on their end (all were drained at the same instant); a shorter
    chunk is a speaker who started or stopped mid-window. Summing — rather
    than concatenating — keeps the stream on the real-time clock, which is
    what a shared microphone in the room would capture.
    """
    usable = [c[: len(c) - (len(c) % 2)] for c in chunks if c]
    usable = [c for c in usable if c]
    if not usable:
        return b""
    if len(usable) == 1:
        return bytes(usable[0])
    np = _require_numpy()
    longest = max(len(c) for c in usable) // 2
    acc = np.zeros(longest, dtype=np.int32)
    for chunk in usable:
        samples = np.frombuffer(chunk, dtype=np.int16)
        acc[longest - samples.size:] += samples
    return np.clip(acc, -32768, 32767).astype(np.int16).tobytes()


def downsample_48k_stereo_to_16k_mono(pcm: bytes) -> Tuple[bytes, bytes]:
    """Return ``(mono_16k_pcm, remainder)`` for Discord-native input PCM.

    ``remainder`` is the trailing sub-group slice (< 12 bytes) the caller
    must prepend to the next chunk so decimation stays sample-aligned
    across drains.
    """
    usable = len(pcm) - (len(pcm) % _DOWNSAMPLE_GROUP_BYTES)
    if usable <= 0:
        return b"", pcm
    remainder = pcm[usable:]
    np = _require_numpy()
    stereo = np.frombuffer(pcm[:usable], dtype=np.int16).astype(np.float32)
    # Mix L/R to 48 kHz mono, then average each 3 samples to 16 kHz.
    mono_48k = stereo.reshape(-1, 2).mean(axis=1)
    out = mono_48k.reshape(-1, 3).mean(axis=1)
    return out.astype(np.int16).tobytes(), remainder


def upsample_24k_mono_to_48k_stereo(pcm: bytes) -> bytes:
    """Convert realtime speech PCM to Discord-native playback PCM.

    Linear-interpolate 24 kHz mono to 48 kHz (insert the midpoint between
    consecutive samples; hold the last), then copy each sample to L and R.
    Byte ratio stays 1:4 (one int16 in → four int16 out).
    """
    usable = len(pcm) - (len(pcm) % 2)
    if usable <= 0:
        return b""
    np = _require_numpy()
    mono_24k = np.frombuffer(pcm[:usable], dtype=np.int16).astype(np.float32)
    n = int(mono_24k.size)
    if n == 0:
        return b""
    mono_48k = np.empty(n * 2, dtype=np.float32)
    mono_48k[0::2] = mono_24k
    if n == 1:
        mono_48k[1] = mono_24k[0]
    else:
        mono_48k[1:-1:2] = (mono_24k[:-1] + mono_24k[1:]) * 0.5
        mono_48k[-1] = mono_24k[-1]
    return np.repeat(mono_48k, 2).astype(np.int16).tobytes()


class DiscordMicBridge:
    """The realtime session's "microphone", fed by the VC receive drain.

    The adapter calls :meth:`feed` once per drain with every speaker's
    freshly drained 48 kHz stereo PCM (already allowlist-filtered). The
    chunks are summed into one signal (:func:`mix_pcm16`) so the session
    hears simultaneous speakers the way a shared microphone would, and a
    carry keeps the 3:1 decimation aligned between drains.
    """

    def __init__(self, on_frame: Callable[[bytes], None]):
        self._on_frame = on_frame
        self._carry = b""
        self._closed = False

    def feed(self, chunks: Sequence[bytes]) -> None:
        if self._closed:
            return
        mixed = mix_pcm16(chunks)
        if not mixed:
            return
        frame, self._carry = downsample_48k_stereo_to_16k_mono(self._carry + mixed)
        if frame:
            self._on_frame(frame)

    def close(self) -> None:
        self._closed = True
        self._carry = b""


class MixerPlayoutSink:
    """Realtime playout sink that streams speech through the guild mixer.

    Implements the optional sink extensions the session honors:
    ``clear()`` drops buffered audio on barge-in and ``pending()`` reports
    whether previously written audio is still audibly draining (so the
    session's ``speaking`` state covers the mixer's buffer, not just its
    own queue).

    ``mixer_getter`` is resolved on every write — the mixer is installed
    per-connection and may be replaced across VC reconnects.
    """

    def __init__(self, mixer_getter: Callable[[], Optional[object]], *, gain: float = 1.0):
        self._mixer_getter = mixer_getter
        self._gain = float(gain)
        self._child: Optional[StreamSpeechChild] = None

    def write(self, chunk: bytes) -> None:
        mixer = self._mixer_getter()
        if mixer is None:
            return  # not connected — drop rather than buffer unboundedly
        child = self._child
        if child is None or child.finished:
            child = StreamSpeechChild("realtime-speech", gain=self._gain)
            self._child = child
        child.feed(upsample_24k_mono_to_48k_stereo(chunk))
        # Idempotent; also re-attaches after a stop_speech() detached it.
        mixer.attach_speech_stream(child)

    def clear(self) -> None:
        child = self._child
        if child is not None:
            child.clear()

    def pending(self) -> bool:
        child = self._child
        return bool(child is not None and child.buffered_bytes > 0)

    def set_active(self, active: bool) -> None:
        # Ducking is audibility-driven inside the mixer; nothing to mark.
        pass

    def close(self) -> None:
        child, self._child = self._child, None
        if child is not None:
            child.end()


__all__ = [
    "DiscordMicBridge",
    "MixerPlayoutSink",
    "downsample_48k_stereo_to_16k_mono",
    "mix_pcm16",
    "upsample_24k_mono_to_48k_stereo",
]
