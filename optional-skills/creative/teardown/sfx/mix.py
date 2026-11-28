"""Mixer: place sounds in time, master to a loudness target, write a WAV."""

import math
import os
import random
import struct
import wave

from .loudness import integrated_lufs, true_peak
from .synth import SR, key

DRIVES = (1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0)


class Mix:
    def __init__(self, duration, seed=11):
        random.seed(seed)
        self.duration = duration
        self.n = int(SR * duration)
        self.left = [0.0] * self.n
        self.right = [0.0] * self.n
        self.count = 0
        self._out = None

    def put(self, t, samples, gain=1.0, pan=0.0):
        """Place mono samples at t seconds. pan: -1 left, 0 centre, +1 right (equal power)."""
        ang = (pan + 1) * math.pi / 4
        gl, gr = gain * math.cos(ang), gain * math.sin(ang)
        i0 = int(t * SR)
        for k, s in enumerate(samples):
            i = i0 + k
            if 0 <= i < self.n:
                self.left[i] += s * gl
                self.right[i] += s * gr
        self.count += 1

    def put_stereo(self, t, left, right, gain=1.0):
        """Place a stereo pair (e.g. from synth.pad) at t seconds."""
        i0 = int(t * SR)
        for k, (a, b) in enumerate(zip(left, right)):
            i = i0 + k
            if 0 <= i < self.n:
                self.left[i] += a * gain
                self.right[i] += b * gain
        self.count += 1

    def typing(self, start, end, chars, gain=(0.16, 0.26), jitter=0.006, spread=0.3):
        """One keystroke per character, evenly spaced from start to end, humanized."""
        for k in range(chars):
            tk = start + (end - start) * k / chars
            self.put(tk + random.uniform(-jitter, jitter), key(), random.uniform(*gain), random.uniform(-spread, spread))

    def master(self, target_lufs=-14.0, ceiling_db=-2.0, fade_out=0.25):
        """Soft-clip and set gain so the mix hits target_lufs with true peak <= ceiling_db.

        Drive only rises when the ceiling would otherwise be crossed. The ceiling
        leaves room for AAC overshoot, so the muxed MP4 stays under -1 dBTP. If no
        drive reaches the target within the ceiling, the mix comes out quieter
        rather than crushed.
        """
        peak = max(max(map(abs, self.left)), max(map(abs, self.right))) or 1.0
        fade_n = int(fade_out * SR)
        fade = [1.0 if i < self.n - fade_n else (self.n - i) / fade_n for i in range(self.n)]
        ceiling = 10 ** (ceiling_db / 20)

        best = None
        for drive in DRIVES:
            norm = math.tanh(drive)
            left = [math.tanh(drive * x / peak) / norm * f for x, f in zip(self.left, fade)]
            right = [math.tanh(drive * x / peak) / norm * f for x, f in zip(self.right, fade)]
            gain = 10 ** ((target_lufs - integrated_lufs(left, right)) / 20)
            if gain > ceiling:
                continue
            tp = max(true_peak(left), true_peak(right))
            fit = min(gain, ceiling / tp)
            if best is None or fit / gain > best[0]:
                best = (fit / gain, drive, left, right, fit, tp)
            if gain * tp <= ceiling:
                break
        if best is None:
            drive = DRIVES[0]
            norm = math.tanh(drive)
            left = [math.tanh(drive * x / peak) / norm * f for x, f in zip(self.left, fade)]
            right = [math.tanh(drive * x / peak) / norm * f for x, f in zip(self.right, fade)]
            tp = max(true_peak(left), true_peak(right))
            best = (0, drive, left, right, ceiling / tp, tp)
        _, drive, left, right, gain, tp = best

        self._out = ([x * gain for x in left], [x * gain for x in right])
        lufs = integrated_lufs(*self._out)
        if lufs < target_lufs - 0.5:
            print(
                f"warning: {lufs:.1f} LUFS, short of {target_lufs}: the mix is too sparse to get louder "
                "without clipping. Add sustained sound (a pad, longer tails) or accept it quieter."
            )
        return {"lufs": lufs, "peak_db": 20 * math.log10(gain * tp), "drive": drive}

    def write(self, path):
        """Write 16-bit stereo WAV; masters with defaults first if master() wasn't called."""
        if self._out is None:
            self.master()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        left, right = self._out
        with wave.open(path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            frames = bytearray()
            for a, b in zip(left, right):
                frames += struct.pack("<hh", int(a * 32767), int(b * 32767))
            wf.writeframes(bytes(frames))
