"""Telephony audio primitives: G.711 µ-law codec, PCM16 resampling, framing.

Vendored pure-Python implementations — stdlib ``audioop`` was removed in
Python 3.13 and Hermes supports 3.13, so we carry the ~60 lines ourselves.
Carrier media streams speak µ-law at 8 kHz in 160-byte (20 ms) frames;
realtime voice models speak PCM16 at 16/24 kHz.
"""

from array import array

ULAW_SILENCE_BYTE = 0xFF  # µ-law encoding of 0
FRAME_BYTES = 160          # 20 ms of µ-law @ 8 kHz
FRAME_SECONDS = 0.02

_BIAS = 0x84
_CLIP = 32635


def _decode_sample(byte: int) -> int:
    byte = ~byte & 0xFF
    sign = byte & 0x80
    exponent = (byte >> 4) & 0x07
    sample = ((((byte & 0x0F) << 3) + _BIAS) << exponent) - _BIAS
    return -sample if sign else sample


_DECODE_TABLE = array("h", (_decode_sample(b) for b in range(256)))


def _encode_sample(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    if sample < 0:
        sample = -sample
    if sample > _CLIP:
        sample = _CLIP
    sample += _BIAS
    exponent = sample.bit_length() - 8  # sample >= 0x84 → bit_length >= 8
    if exponent < 0:
        exponent = 0
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


_ENCODE_TABLE = bytes(
    _encode_sample(s - 32768) for s in range(65536)
)  # index by (sample & 0xFFFF) via offset below


def ulaw_to_pcm16(data: bytes) -> bytes:
    """Decode µ-law bytes to little-endian PCM16."""
    out = array("h", bytes(2 * len(data)))
    for i, byte in enumerate(data):
        out[i] = _DECODE_TABLE[byte]
    return out.tobytes()


def pcm16_to_ulaw(data: bytes) -> bytes:
    """Encode little-endian PCM16 to µ-law bytes."""
    samples = array("h")
    samples.frombytes(data[: len(data) - (len(data) % 2)])
    return bytes(_ENCODE_TABLE[(s + 32768) & 0xFFFF] for s in samples)


def resample_pcm16(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resampling of mono PCM16."""
    if src_rate == dst_rate or not data:
        return data
    src = array("h")
    src.frombytes(data[: len(data) - (len(data) % 2)])
    n_src = len(src)
    if n_src == 0:
        return b""
    n_dst = max(1, int(n_src * dst_rate / src_rate))
    out = array("h", bytes(2 * n_dst))
    step = (n_src - 1) / n_dst if n_dst > 1 else 0.0
    pos = 0.0
    for i in range(n_dst):
        idx = int(pos)
        frac = pos - idx
        nxt = src[idx + 1] if idx + 1 < n_src else src[idx]
        out[i] = int(src[idx] * (1.0 - frac) + nxt * frac)
        pos += step
    return out.tobytes()


def chunk_frames(data: bytes, frame_bytes: int = FRAME_BYTES) -> list:
    """Split µ-law bytes into fixed frames, padding the tail with silence."""
    frames = []
    for offset in range(0, len(data), frame_bytes):
        frame = data[offset:offset + frame_bytes]
        if len(frame) < frame_bytes:
            frame = frame + bytes([ULAW_SILENCE_BYTE]) * (frame_bytes - len(frame))
        frames.append(frame)
    return frames


def silence_frame(frame_bytes: int = FRAME_BYTES) -> bytes:
    return bytes([ULAW_SILENCE_BYTE]) * frame_bytes
