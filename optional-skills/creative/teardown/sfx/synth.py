"""Sound generators.

Every generator returns mono samples (a list of floats, roughly -1..1) at SR.
The first line of each docstring says what it's for; `python run_sfx.py --list`
prints them. To add a sound, write a function here in the same shape.
Use the module-level `random` for noise so renders stay reproducible.
"""

import array
import bisect
import math
import os
import random
import subprocess

SR = 48000
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def sample(name, start=0.0, dur=None):
    """Recorded sound from sfx/samples (or any path): keyboards, room tone. start/dur trim it in seconds."""
    path = name if os.path.isabs(name) else os.path.join(SAMPLES, name)
    cmd = [FFMPEG, "-v", "error", "-ss", str(start), "-i", path]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=300).stdout
    pcm = array.array("h")
    pcm.frombytes(raw[: len(raw) // 2 * 2])
    return [x / 32768 for x in pcm]


def bandnoise(n, fc_at, q=0.8):
    """Building block: white noise through a band-pass, fc_at(k) is the cutoff in Hz at sample k."""
    low = band = 0.0
    out = []
    for k in range(n):
        f = 2 * math.sin(math.pi * min(fc_at(k), SR / 6) / SR)
        x = random.uniform(-1, 1)
        low += f * band
        high = x - low - q * band
        band += f * high
        out.append(band)
    return out


def click(freq=1800, dur=0.05):
    """UI click: a control snapping to a step, a toggle, a tick. Higher freq sounds smaller."""
    n = int(dur * SR)
    out = []
    for k in range(n):
        tt = k / SR
        s = math.sin(2 * math.pi * freq * tt) * math.exp(-tt / 0.012)
        s += random.uniform(-1, 1) * math.exp(-tt / 0.0015) * 0.6
        out.append(s)
    return out


def key():
    """One keystroke; every call is slightly different. For typing use Mix.typing()."""
    n = int(0.018 * SR)
    fc = random.uniform(2400, 4200)
    noise = bandnoise(n, lambda k: fc, q=0.5)
    thock = random.uniform(160, 220)
    return [
        noise[k] * math.exp(-k / (0.0025 * SR)) * 2.2
        + math.sin(2 * math.pi * thock * k / SR) * math.exp(-k / (0.006 * SR)) * 0.5
        for k in range(n)
    ]


def blip(freq, dur=0.35, tau=0.12):
    """Soft tonal blip: a word or element landing, one note of a sequence. Keep freq in key."""
    n = int(dur * SR)
    out = []
    ph = 0.0
    for k in range(n):
        tt = k / SR
        ph += 2 * math.pi * freq / SR
        a = min(1, tt / 0.003) * math.exp(-tt / tau)
        out.append(a * (math.sin(ph) + 0.25 * math.sin(2 * ph) + 0.08 * math.sin(3 * ph)))
    return out


def pop(f0, f1, dur=0.16):
    """Pitched pop: a badge, chip or callout popping in. Sweeps f0 to f1; raise f0 across a sequence."""
    n = int(dur * SR)
    out = []
    ph = 0.0
    for k in range(n):
        tt = k / SR
        f = f0 * (f1 / f0) ** min(1, tt / 0.06)
        ph += 2 * math.pi * f / SR
        a = min(1, tt / 0.002) * math.exp(-tt / 0.045)
        out.append(a * math.sin(ph))
    return out


def chime(freqs, dur=1.8, tau=0.7):
    """Bell chord: success, arrival, a payoff. Pass freqs from the video's key."""
    n = int(dur * SR)
    out = [0.0] * n
    for j, f in enumerate(freqs):
        ph1 = ph2 = 0.0
        g = 1.0 / (1 + j * 0.35)
        for k in range(n):
            tt = k / SR
            ph1 += 2 * math.pi * f / SR
            ph2 += 2 * math.pi * f * 2.76 / SR
            a = min(1, tt / 0.004) * math.exp(-tt / tau)
            out[k] += g * a * (math.sin(ph1) + 0.18 * math.sin(ph2) * math.exp(-tt / 0.15))
    return out


def whoosh(dur, f0, f1, peak=0.6, q=0.9):
    """Air whoosh: movement, reveals, transitions. f0 to f1 sets direction, peak (0..1) where it's loudest."""
    n = int(dur * SR)
    p = math.log(0.5) / math.log(peak)
    noise = bandnoise(n, lambda k: f0 * (f1 / f0) ** (k / n), q=q)
    return [noise[k] * math.sin(math.pi * (k / n) ** p) ** 2 * 1.6 for k in range(n)]


def riser(dur, f0=260, f1=5200):
    """Build-up into a payoff. End it exactly on the hit; one big riser per video."""
    n = int(dur * SR)
    noise = bandnoise(n, lambda k: f0 * (f1 / f0) ** (k / n), q=0.6)
    out = []
    ph = 0.0
    for k in range(n):
        x = k / n
        ph += 2 * math.pi * (180 * (4.0 ** x)) / SR
        tail = min(1, (n - k) / (0.02 * SR))
        out.append((noise[k] * 1.3 + 0.35 * math.sin(ph)) * x ** 2.2 * tail)
    return out


def impact(dur=1.2):
    """Low impact for the hero moment. One per video."""
    n = int(dur * SR)
    out = []
    ph = 0.0
    lp = 0.0
    for k in range(n):
        tt = k / SR
        f = 38 + 60 * math.exp(-tt / 0.07)
        ph += 2 * math.pi * f / SR
        lp += 0.08 * (random.uniform(-1, 1) - lp)
        s = math.sin(ph) * math.exp(-tt / 0.38)
        s += lp * math.exp(-tt / 0.05) * 3.0
        out.append(math.tanh(1.6 * s))
    return out


def scratch(dur):
    """Pencil on paper: a sketch or line drawing itself on."""
    n = int(dur * SR)
    noise = bandnoise(n, lambda k: 3800 + 1400 * math.sin(k / SR * 23), q=0.35)
    out = []
    for k in range(n):
        tt = k / SR
        stroke = max(0.0, math.sin(tt * 2 * math.pi * 7.5)) ** 3
        env = math.sin(math.pi * k / n)
        out.append(noise[k] * stroke * env * 1.4)
    return out


def envelope(points):
    """Helper: piecewise-linear gain curve from [(t, value), ...], for pad volume and ducking."""
    pts = sorted(points)
    ts = [p[0] for p in pts]

    def at(t):
        i = bisect.bisect_right(ts, t)
        if i == 0:
            return pts[0][1]
        if i == len(pts):
            return pts[-1][1]
        (t0, v0), (t1, v1) = pts[i - 1], pts[i]
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0)

    return at


def pad(duration, chords, volume=None, detune=0.6, tremolo=0.23, xfade=0.35):
    """Music bed under everything; returns (left, right) for Mix.put_stereo. chords = [(start_s, [freqs]), ...]."""
    n = int(duration * SR)
    left = [0.0] * n
    right = [0.0] * n
    for ci, (t0, freqs) in enumerate(chords):
        t_next = chords[ci + 1][0] if ci + 1 < len(chords) else None
        i0 = int(t0 * SR)
        i1 = n if t_next is None else min(n, int((t_next + xfade) * SR))
        phl = [0.0] * len(freqs)
        phr = [0.0] * len(freqs)
        for i in range(i0, i1):
            t = i / SR
            w = 1.0 if ci == 0 else min(1.0, (t - t0) / xfade)
            if t_next is not None and t > t_next:
                w *= max(0.0, 1 - (t - t_next) / xfade)
            v = w * (volume(t) if volume else 1.0) * (0.85 + 0.15 * math.sin(2 * math.pi * tremolo * t))
            sl = sr = 0.0
            for j, f in enumerate(freqs):
                phl[j] += 2 * math.pi * (f - detune) / SR
                phr[j] += 2 * math.pi * (f + detune) / SR
                g = 1 / (1 + j * 0.5)
                sl += g * math.sin(phl[j])
                sr += g * math.sin(phr[j])
            left[i] += sl * v
            right[i] += sr * v
    return left, right


# ---------------------------------------------------------------- a small band, for music beds
# Added 2026-10-18 for the coffee-machine film. Soft, warm, low in the mix: the bed sits under the picture,
# the accents (clicks, hiss) sit on top of it.


def epiano(freq, dur=2.6, vel=1.0, decay=1.1):
    """Electric-piano note (tine and warm body), for lo-fi chords. Keep freq in key; vel 0..1."""
    n = int(dur * SR)
    out = [0.0] * n
    w1 = 2 * math.pi * freq / SR
    w2 = 2 * w1
    wt = 2 * math.pi * freq * 6.98 / SR
    wtrem = 2 * math.pi * 4.6 / SR
    sin = math.sin
    exp = math.exp
    for k in range(n):
        t = k / SR
        a = (t / 0.006 if t < 0.006 else 1.0) * exp(-t / decay)
        body = sin(w1 * k) + 0.28 * sin(w2 * k) * exp(-t / 0.5)
        tine = 0.16 * sin(wt * k) * exp(-t / 0.09)
        out[k] = vel * a * (body + tine) * (1 + 0.1 * sin(wtrem * k))
    tail = min(n, int(0.08 * SR))
    for i in range(tail):
        out[n - 1 - i] *= i / tail
    return out


def bass(freq, dur=0.9, vel=1.0):
    """Round bass note: sine plus a little second harmonic, gently saturated. Roots under the chords."""
    n = int(dur * SR)
    out = [0.0] * n
    w = 2 * math.pi * freq / SR
    sin = math.sin
    exp = math.exp
    for k in range(n):
        t = k / SR
        a = (t / 0.012 if t < 0.012 else 1.0) * exp(-t / 0.45)
        out[k] = math.tanh(1.4 * vel * a * (sin(w * k) + 0.22 * sin(2 * w * k)))
    tail = min(n, int(0.05 * SR))
    for i in range(tail):
        out[n - 1 - i] *= i / tail
    return out


def kick(vel=1.0):
    """Soft kick: a sine dropping from 110 to 46 Hz. Lo-fi, not club."""
    n = int(0.32 * SR)
    out = [0.0] * n
    ph = 0.0
    for k in range(n):
        t = k / SR
        f = 46 + 64 * math.exp(-t / 0.035)
        ph += 2 * math.pi * f / SR
        out[k] = vel * math.sin(ph) * math.exp(-t / 0.11) * min(1.0, t / 0.002)
    return out


def snare(vel=1.0):
    """Brushed snare: band-passed noise with a short body tone, for beats 2 and 4."""
    n = int(0.22 * SR)
    noise = bandnoise(n, lambda k: 2100, q=0.55)
    return [vel * (noise[k] * 1.6 * math.exp(-k / (0.06 * SR)) + 0.25 * math.sin(2 * math.pi * 190 * k / SR) * math.exp(-k / (0.025 * SR))) for k in range(n)]


def hat(vel=1.0):
    """Closed hat tick: high band noise, very short. Swing the off-beats."""
    n = int(0.05 * SR)
    noise = bandnoise(n, lambda k: 7600, q=0.4)
    return [vel * noise[k] * 1.8 * math.exp(-k / (0.011 * SR)) for k in range(n)]


def crackle(dur, density=7.0, hiss=0.012):
    """Vinyl crackle bed: sparse clicks over a faint hiss. density is clicks per second."""
    n = int(dur * SR)
    out = [random.uniform(-1, 1) * hiss for _ in range(n)]
    for _ in range(int(dur * density)):
        at = random.randrange(n)
        g = random.uniform(0.05, 0.35) * random.choice((-1, 1))
        for i in range(min(40, n - at)):
            out[at + i] += g * math.exp(-i / 6.0)
    return out
