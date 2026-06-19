"""PCM16 audio helpers: rate conversion, 20 ms framing, RMS.

The bridge speaks PCM 16 kHz (worker side); the realtime model speaks PCM 24 kHz.
These helpers convert between them and chop a stream into the worker's fixed
20 ms / 640-byte frames. All functions operate on little-endian signed 16-bit
mono bytes — the wire format both peers agree on.

Uses numpy when available (fast, exact endianness via ``<i2``); falls back to a
pure-stdlib linear resampler so the plugin has no hard numpy dependency.
"""

from __future__ import annotations

import array
import math
import sys

try:  # optional fast path
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only without numpy
    _np = None


def resample_pcm16(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-resample LE int16 mono PCM from ``src_rate`` to ``dst_rate``."""
    if src_rate == dst_rate or not data:
        return data
    if len(data) % 2:
        data = data[:-1]
    if not data:
        return b""

    if _np is not None:
        a = _np.frombuffer(data, dtype="<i2").astype(_np.float32)
        n = a.shape[0]
        out_n = max(1, int(round(n * dst_rate / src_rate)))
        if n == 1:
            res = _np.full(out_n, a[0], dtype=_np.float32)
        else:
            x_old = _np.linspace(0.0, 1.0, n, endpoint=True)
            x_new = _np.linspace(0.0, 1.0, out_n, endpoint=True)
            res = _np.interp(x_new, x_old, a)
        return _np.clip(_np.round(res), -32768, 32767).astype("<i2").tobytes()

    # Pure-stdlib fallback.
    samples = array.array("h")
    samples.frombytes(data)
    if sys.byteorder == "big":
        samples.byteswap()
    n = len(samples)
    out_n = max(1, int(round(n * dst_rate / src_rate)))
    out = array.array("h", bytes(2 * out_n))
    if out_n == 1:
        out[0] = samples[0]
    else:
        ratio = (n - 1) / (out_n - 1)
        for i in range(out_n):
            pos = i * ratio
            i0 = int(pos)
            frac = pos - i0
            i1 = i0 + 1 if i0 + 1 < n else i0
            out[i] = int(round(samples[i0] * (1.0 - frac) + samples[i1] * frac))
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes()


def frame_pcm16(data: bytes, frame_bytes: int = 640) -> tuple[list[bytes], bytes]:
    """Split ``data`` into fixed-size frames; return ``(frames, residual)``.

    The trailing ``residual`` (< ``frame_bytes``) is meant to be prepended to the
    next chunk so frame boundaries stay aligned across streamed deltas.
    """
    count = len(data) // frame_bytes
    frames = [data[i * frame_bytes : (i + 1) * frame_bytes] for i in range(count)]
    residual = data[count * frame_bytes :]
    return frames, residual


def pcm16_rms(data: bytes) -> float:
    """Root-mean-square amplitude of LE int16 PCM, normalized to 0.0-1.0."""
    if len(data) < 2:
        return 0.0
    if len(data) % 2:
        data = data[:-1]
    if _np is not None:
        a = _np.frombuffer(data, dtype="<i2").astype(_np.float32) / 32768.0
        return float(_np.sqrt(_np.mean(a * a)))
    samples = array.array("h")
    samples.frombytes(data)
    if sys.byteorder == "big":
        samples.byteswap()
    acc = sum((s / 32768.0) ** 2 for s in samples)
    return math.sqrt(acc / len(samples))
