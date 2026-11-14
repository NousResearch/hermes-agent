"""DSP building blocks (oscillators, filters, delays, reverb), pure Python stdlib at 48 kHz.

Used by sound scripts that need more than the one-shot generators in synth.py,
e.g. sims/engine/sound.py.

Signals are plain lists of floats. Speed comes from two tricks:

- A Python per-sample loop costs about 0.3 µs/sample and a list comprehension
  about 0.05 µs. So constant-pitch oscillators are one comprehension (the naive
  ramp) plus a sparse PolyBLEP fix-up at each wrap. Feedback delays and the
  reverb combs/allpasses run one block at a time, where the block is no longer
  than the delay, so every block depends only on earlier blocks and is a single
  comprehension.
- Filters whose state really is sequential (SVF, biquad, ladder) are tight
  loops, run only on short one-shots or on the few full-length buses that need
  them.
"""

import math

from .synth import SR

TAU = 2.0 * math.pi
_tanh = math.tanh


# ---------------------------------------------------------------- buffers


def zeros(n):
    return [0.0] * n


def add(dst, src, at=0, gain=1.0):
    """dst[at:] += gain * src, in place, clipped to dst."""
    if at < 0:
        src = src[-at:]
        at = 0
    end = min(len(dst), at + len(src))
    if end <= at:
        return
    seg = dst[at:end]
    if gain == 1.0:
        dst[at:end] = [a + b for a, b in zip(seg, src)]
    else:
        dst[at:end] = [a + b * gain for a, b in zip(seg, src)]


def scaled(x, g):
    return [v * g for v in x]


def mul(x, y):
    return [a * b for a, b in zip(x, y)]


def summed(a, b):
    return [p + q for p, q in zip(a, b)]


def rms(x):
    return math.sqrt(sum(v * v for v in x) / len(x)) if x else 0.0


def peak(x):
    return max(max(x), -min(x)) if x else 0.0


def pan_gains(pan):
    """Equal-power pan, -1 left .. +1 right."""
    ang = (pan + 1.0) * math.pi / 4.0
    return math.cos(ang), math.sin(ang)


def fade(x, fin=0, fout=0):
    """Linear fade in/out over fin/fout samples, in place."""
    n = len(x)
    fin, fout = min(fin, n), min(fout, n)
    for i in range(fin):
        x[i] *= i / fin
    for i in range(fout):
        x[n - 1 - i] *= i / fout
    return x


# ------------------------------------------------------------ oscillators


def _blep(out, phase, dt, n, h=1.0):
    """Subtract the PolyBLEP residual at every wrap of the ramp phase + i*dt (downward jump 2h)."""
    k = 1
    while True:
        j = math.ceil((k - phase) / dt)
        # float guard: make j the first sample whose wrapped phase is small
        if j < n and (phase + j * dt) % 1.0 > 0.5:
            j += 1
        if j >= 1 and (phase + (j - 1) * dt) % 1.0 < 0.5:
            j -= 1
        if j - 1 >= n:
            break
        if 0 <= j < n:
            u = ((phase + j * dt) % 1.0) / dt
            if u < 1.0:
                out[j] -= h * (u + u - u * u - 1.0)
        if 1 <= j <= n:
            u = (((phase + (j - 1) * dt) % 1.0) - 1.0) / dt
            if u > -1.0:
                out[j - 1] -= h * (u * u + u + u + 1.0)
        k += 1


def saw(freq, n, phase=0.0):
    """Band-limited sawtooth (PolyBLEP) at constant freq, -1..1."""
    dt = freq / SR
    phase %= 1.0
    out = [2.0 * ((phase + i * dt) % 1.0) - 1.0 for i in range(n)]
    if 0.0 < dt < 0.5:
        _blep(out, phase, dt, n)
    return out


def pulse(freq, n, width=0.5, phase=0.0):
    """Band-limited pulse as the difference of two PolyBLEP saws; zero mean."""
    a = saw(freq, n, phase)
    b = saw(freq, n, phase + width)
    return [p - q for p, q in zip(a, b)]


def sine(freq, n, phase=0.0):
    w = TAU * freq / SR
    ph = TAU * phase
    sin = math.sin
    return [sin(ph + w * i) for i in range(n)]


def noise(rng, n):
    r = rng.random
    return [2.0 * r() - 1.0 for _ in range(n)]


# ---------------------------------------------------------------- filters


def _norm(b0, b1, b2, a0, a1, a2):
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def lp(fc, q=0.7071):
    w = TAU * min(fc, 0.45 * SR) / SR
    c, al = math.cos(w), math.sin(w) / (2 * q)
    return _norm((1 - c) / 2, 1 - c, (1 - c) / 2, 1 + al, -2 * c, 1 - al)


def hp(fc, q=0.7071):
    w = TAU * min(fc, 0.45 * SR) / SR
    c, al = math.cos(w), math.sin(w) / (2 * q)
    return _norm((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al)


def bp(fc, q=1.0):
    w = TAU * min(fc, 0.45 * SR) / SR
    c, al = math.cos(w), math.sin(w) / (2 * q)
    return _norm(al, 0.0, -al, 1 + al, -2 * c, 1 - al)


def peaking(fc, q, db):
    a = 10 ** (db / 40)
    w = TAU * min(fc, 0.45 * SR) / SR
    c, al = math.cos(w), math.sin(w) / (2 * q)
    return _norm(1 + al * a, -2 * c, 1 - al * a, 1 + al / a, -2 * c, 1 - al / a)


def shelf(fc, db, high=True):
    a = 10 ** (db / 40)
    w = TAU * min(fc, 0.45 * SR) / SR
    c, s = math.cos(w), math.sin(w)
    al = s / 2 * math.sqrt(2)
    sa = 2 * math.sqrt(a) * al
    if high:
        return _norm(
            a * ((a + 1) + (a - 1) * c + sa), -2 * a * ((a - 1) + (a + 1) * c), a * ((a + 1) + (a - 1) * c - sa),
            (a + 1) - (a - 1) * c + sa, 2 * ((a - 1) - (a + 1) * c), (a + 1) - (a - 1) * c - sa,
        )
    return _norm(
        a * ((a + 1) - (a - 1) * c + sa), 2 * a * ((a - 1) - (a + 1) * c), a * ((a + 1) - (a - 1) * c - sa),
        (a + 1) + (a - 1) * c + sa, -2 * ((a - 1) + (a + 1) * c), (a + 1) + (a - 1) * c - sa,
    )


def biquad(x, c):
    """Static biquad (transposed direct form II); c from lp/hp/bp/peaking/shelf."""
    b0, b1, b2, a1, a2 = c
    z1 = z2 = 0.0
    out = [0.0] * len(x)
    i = 0
    for v in x:
        y = b0 * v + z1
        z1 = b1 * v - a1 * y + z2
        z2 = b2 * v - a2 * y
        out[i] = y
        i += 1
    return out


def svf(x, fc, q=0.7071, mode="lp", ctrl=32):
    """Zero-delay-feedback state-variable filter (Cytomic/Simper), stable under fast modulation.

    fc: Hz, or a function of the sample index evaluated every `ctrl` samples.
    mode: 'lp', 'bp' or 'hp'.
    """
    n = len(x)
    out = [0.0] * n
    k = 1.0 / q
    ic1 = ic2 = 0.0
    const = not callable(fc)
    if const:
        ctrl = max(n, 1)
    tan = math.tan
    pis = math.pi / SR
    for s in range(0, n, ctrl):
        f = fc if const else fc(s)
        f = 10.0 if f < 10.0 else min(f, 20000.0)
        g = tan(pis * f)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        e = min(n, s + ctrl)
        if mode == "lp":
            for i in range(s, e):
                v3 = x[i] - ic2
                v1 = a1 * ic1 + a2 * v3
                v2 = ic2 + a2 * ic1 + a3 * v3
                ic1 = v1 + v1 - ic1
                ic2 = v2 + v2 - ic2
                out[i] = v2
        elif mode == "bp":
            for i in range(s, e):
                v3 = x[i] - ic2
                v1 = a1 * ic1 + a2 * v3
                v2 = ic2 + a2 * ic1 + a3 * v3
                ic1 = v1 + v1 - ic1
                ic2 = v2 + v2 - ic2
                out[i] = v1
        else:
            for i in range(s, e):
                xi = x[i]
                v3 = xi - ic2
                v1 = a1 * ic1 + a2 * v3
                v2 = ic2 + a2 * ic1 + a3 * v3
                ic1 = v1 + v1 - ic1
                ic2 = v2 + v2 - ic2
                out[i] = xi - k * v1 - v2
    return out


def onepole_hp(x, fc):
    """Gentle 6 dB/oct high-pass; used as a DC blocker."""
    a = math.exp(-TAU * fc / SR)
    out = [0.0] * len(x)
    px = py = 0.0
    i = 0
    for v in x:
        py = a * (py + v - px)
        px = v
        out[i] = py
        i += 1
    return out


def sweep(x, start, end, f0, f1, mode="lp", q=0.9, wet_in=0.0, wet_out=0.0):
    """Filter x[start:end] in place, cutoff gliding exponentially f0 -> f1.

    wet_in / wet_out (fractions of the region) crossfade from / back to the
    dry signal, so the region's edges are seamless.
    """
    start, end = max(0, start), min(len(x), end)
    n = end - start
    if n <= 0:
        return
    seg = x[start:end]
    r = f1 / f0
    y = svf(seg, lambda i: f0 * r ** (i / n), q, mode, 32)
    a = max(1, int(wet_in * n))
    b = max(1, int(wet_out * n))
    out = []
    for i in range(n):
        w = 1.0
        if wet_in and i < a:
            w = i / a
        if wet_out and i > n - b:
            w = min(w, (n - i) / b)
        out.append(seg[i] + (y[i] - seg[i]) * w)
    x[start:end] = out


# ------------------------------------------------------- delay and reverb

_COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
_ALLPASSES = (556, 441, 341, 225)


def _comb(x, m, fb, damp):
    """Freeverb comb, block-vectorized; the in-loop damping is a 2-tap low-pass."""
    n = len(x)
    a = fb * (1.0 - damp)
    b = fb * damp
    w = [0.0] * (n + m + 1)  # w[i + m + 1] = W(i)
    for s in range(0, n, m):
        e = min(s + m, n)
        w[s + m + 1:e + m + 1] = [xi + a * p + b * q for xi, p, q in zip(x[s:e], w[s + 1:e + 1], w[s:e])]
    return w[1:n + 1]


def _allpass(x, m, g=0.5):
    n = len(x)
    v = [0.0] * (n + m)  # v[i + m] = V(i)
    for s in range(0, n, m):
        e = min(s + m, n)
        v[s + m:e + m] = [xi + g * p for xi, p in zip(x[s:e], v[s:e])]
    return [p - xi for xi, p in zip(x, v)]


def reverb(x, size=0.7, damp=0.45, scale=1.0, spread=23):
    """Freeverb (8 combs + 4 allpasses per side): mono in, (left, right) wet out.

    size 0..1 maps to comb feedback 0.7..0.98; scale stretches every delay
    (0.3 = small room, 1.5 = hall). Output is roughly level-matched to input.
    """
    fb = 0.7 + 0.28 * size
    r = SR / 44100 * scale
    outs = []
    for off in (0, spread):
        acc = None
        for m in _COMBS:
            c = _comb(x, max(8, int((m + off) * r)), fb, damp)
            acc = c if acc is None else [p + q for p, q in zip(acc, c)]
        for m in _ALLPASSES:
            acc = _allpass(acc, max(8, int((m + off) * r)))
        outs.append(acc)
    g = math.sqrt((1 - fb * fb) / 8.0) * 1.4
    return [v * g for v in outs[0]], [v * g for v in outs[1]]


def pingpong(x, d, fb=0.5, damp=0.35):
    """Ping-pong delay, mono in -> (left, right) wet; first echo left at d samples, then alternating."""
    n = len(x)
    xp = [0.0] * d + x
    a = [0.0] * (n + d + 1)  # a[i + d + 1] = A(i)
    b = [0.0] * (n + d + 1)
    c0 = fb * (1.0 - damp)
    c1 = fb * damp
    for s in range(0, n, d):
        e = min(s + d, n)
        av = [xi + c0 * p + c1 * q for xi, p, q in zip(xp[s:e], b[s + 1:e + 1], b[s:e])]
        bv = [c0 * p + c1 * q for p, q in zip(a[s + 1:e + 1], a[s:e])]
        a[s + d + 1:e + d + 1] = av
        b[s + d + 1:e + d + 1] = bv
    return a[d + 1:], b[d + 1:]


# --------------------------------------------------------------- dynamics


def softclip(x, drive=1.5):
    norm = 1.0 / _tanh(drive)
    return [_tanh(drive * v) * norm for v in x]


def compress(left, right, thresh_db=-12.0, ratio=2.0, attack=0.01, release=0.15, knee_db=6.0, block=64):
    """Stereo-linked RMS bus compressor at control rate (one gain per block, interpolated)."""
    n = len(left)
    pw = [a * a + b * b for a, b in zip(left, right)]
    ca = math.exp(-block / (attack * SR))
    cr = math.exp(-block / (release * SR))
    slope = 1.0 / ratio - 1.0
    half = knee_db / 2
    gr = 0.0
    gains = []
    # warm the detector on the first second so the start isn't uncompressed
    for warm in (True, False):
        for s in range(0, n if not warm else min(n, SR), block):
            p = sum(pw[s:s + block]) / (2 * block)
            db = 10 * math.log10(p + 1e-12)
            over = db - thresh_db
            if over <= -half:
                want = 0.0
            elif over < half:
                want = slope * (over + half) ** 2 / (2 * knee_db)
            else:
                want = slope * over
            c = ca if want < gr else cr
            gr = c * gr + (1 - c) * want
            if not warm:
                gains.append(10 ** (gr / 20))
    ramp = [k / block for k in range(block)]
    g = []
    prev = gains[0]
    for cur in gains:
        d = cur - prev
        g.extend([prev + d * r for r in ramp])
        prev = cur
    return [a * b for a, b in zip(left, g)], [a * b for a, b in zip(right, g)]


def mono_below(left, right, fc=120.0):
    """Collapse the side signal below fc (24 dB/oct), so the low end is mono."""
    mid = [(a + b) * 0.5 for a, b in zip(left, right)]
    side = [(a - b) * 0.5 for a, b in zip(left, right)]
    c = hp(fc)
    side = biquad(biquad(side, c), c)
    return [m + s for m, s in zip(mid, side)], [m - s for m, s in zip(mid, side)]


# --------------------------------------------------------- glitch effects


def crush(x, bits=8, hold=2):
    """Bit and sample-rate reduction (intentionally aliasing)."""
    q = float(2 ** (bits - 1))
    return [round(x[i - i % hold] * q) / q for i in range(len(x))]


def stutter(x, start, slice_n, repeats, decay=0.9, edge=48):
    """Beat-repeat in place: x[start:start+slice_n] repeated `repeats` times."""
    sl = x[start:start + slice_n]
    if len(sl) < slice_n:
        return
    sl = fade(sl[:], edge, edge)
    for r in range(repeats):
        a = start + r * slice_n
        if a + slice_n > len(x):
            break
        x[a:a + slice_n] = [v * decay ** r for v in sl]


def tapestop(x, start, n, curve=1.6):
    """Tape stop in place over x[start:start+n]: playback slows to zero."""
    seg = x[start:start + n]
    n = len(seg)
    if n < 2:
        return
    out = []
    pos = 0.0
    for i in range(n):
        j = int(pos)
        if j + 1 >= n:
            out.append(0.0)
        else:
            f = pos - j
            out.append(seg[j] + (seg[j + 1] - seg[j]) * f)
        pos += (1 - i / n) ** curve
    fade(out, 0, min(n, 480))
    x[start:start + n] = out
