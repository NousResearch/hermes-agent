"""Shared pieces for the figure scores: a parameterised lo-fi bed and a handful of accents.

A figure's score.py imports this (it sits next to the sfx engine on sys.path when run through
run_sfx.py) and adds its own accents on the tour's seconds.
"""

import math
import os
import random

from sfx.dsp import add, reverb, svf, zeros
from sfx.synth import SR, bandnoise, bass, blip, epiano, hat, kick, snare


def env_fade(x, fin, fout):
    n = len(x)
    a = max(1, int(fin * SR))
    b = max(1, int(fout * SR))
    for i in range(min(a, n)):
        x[i] *= i / a
    for i in range(min(b, n)):
        x[n - 1 - i] *= i / b
    return x


def bed(dur, bpm, chords, motif=(), motif_voice="epiano", bass_pattern=((0, 1.15, 0.9), (2.5, 0.85, 0.6)),
        kicks=(0, 2.75), snares=(1, 3), hats=8, swing=0.12, drums_from=1, keys_lp=4200, crackle_level=0.0, keys_vel=0.9):
    """A looping bed: e-piano chords (one per bar), a bass line, soft drums and an optional motif every other bar.

    chords: [([note Hz, ...], root Hz), ...]. bass_pattern: (beat, length in beats, velocity). Returns stereo keys
    plus mono bass and drum stems, all faded in and out.
    """
    if os.environ.get("FIG_VARIANT"):
        # the second-upload variant: three semitones up, a touch faster, the other motif voice
        up = 2 ** (3 / 12)
        chords = [([f * up for f in notes], root * up) for notes, root in chords]
        motif = [(bt, f * up) for bt, f in motif]
        motif_voice = "blip" if motif_voice == "epiano" else "epiano"
        bpm *= 1.06
    beat = 60 / bpm
    bar = 4 * beat
    n = int(dur * SR)
    keys, low, drums = zeros(n), zeros(n), zeros(n)
    bars = int(dur / bar) + 1
    for b in range(bars):
        t = 0.12 + b * bar
        notes, root = chords[b % len(chords)]
        for i, f in enumerate(notes):
            add(keys, epiano(f, 2.9 * 84 / bpm + 0.4, keys_vel), int((t + 0.012 * i) * SR))
            if i >= 2:
                add(keys, epiano(f, 1.3, keys_vel * 0.5), int((t + 1.5 * beat + 0.01 * i) * SR))
        if b % 2 == 1:
            for bt, f in motif:
                v = epiano(f, 1.1, 0.34, 0.6) if motif_voice == "epiano" else blip(f, 0.32, 0.11)
                add(keys, v, int((t + bt * beat) * SR), 1.0 if motif_voice == "epiano" else 0.5)
        for bt, length, vel in bass_pattern:
            add(low, bass(root, length * beat, vel), int((t + bt * beat) * SR))
        if drums_from <= b < bars - 2:
            for bt in kicks:
                add(drums, kick(1.0 if bt == 0 else 0.75), int((t + bt * beat) * SR))
            for bt in snares:
                add(drums, snare(0.55), int((t + bt * beat) * SR))
            for e in range(hats):
                sw = swing * beat if (hats == 8 and e % 2) else 0.0
                add(drums, hat(0.5 if e % 2 == 0 else 0.32), int((t + e * 4 * beat / hats + sw) * SR))
    keys = svf(keys, keys_lp, 0.6, "lp")
    wl, wr = reverb(keys, size=0.6, damp=0.5, scale=0.9)
    left = [k * 0.8 + w * 0.36 for k, w in zip(keys, wl)]
    right = [k * 0.8 + w * 0.36 for k, w in zip(keys, wr)]
    low = svf(low, 900, 0.7, "lp")
    drums = svf(drums, 7000, 0.6, "lp")
    for buf in (left, right, low, drums):
        env_fade(buf, 1.0, 2.2)
    return left, right, low, drums


def thud(f=95, decay=0.12, dur=0.4):
    """A soft body hitting something: a plush toy landing, a knock."""
    n = int(dur * SR)
    return [math.sin(2 * math.pi * f * k / SR) * math.exp(-k / (decay * SR)) * min(1, k / 40) for k in range(n)]


def motor(dur, f0=220, grit_hz=1400, wobble=17):
    """A small electric motor: a buzzy tone plus brush noise, faded at both ends."""
    n = int(dur * SR)
    grit = bandnoise(n, lambda k: grit_hz, q=0.8)
    out = []
    for k in range(n):
        t = k / SR
        m = 1 + 0.15 * math.sin(2 * math.pi * wobble * t)
        s = sum(math.sin(2 * math.pi * f0 * h * t) / h for h in (1, 2, 3, 4))
        out.append(m * (0.35 * s + 0.5 * grit[k]))
    return env_fade(out, 0.06, 0.1)


def airy(dur, f0, f1, level=1.0):
    """Moving air: band noise sweeping from f0 to f1 with a soft swell."""
    n = int(dur * SR)
    x = bandnoise(n, lambda k: f0 * (f1 / f0) ** (k / n), q=0.7)
    return [x[k] * level * 1.4 * math.sin(math.pi * k / n) for k in range(n)]


def bang(level=1.0, seed=0):
    """A small explosion: a low punch and a burst of dark noise."""
    rnd = random.Random(seed)
    n = int(0.42 * SR)
    noise = bandnoise(n, lambda k: 700 + 1600 * math.exp(-k / (0.03 * SR)), q=0.5)
    out = []
    ph = 0.0
    for k in range(n):
        t = k / SR
        f = 48 + 90 * math.exp(-t / 0.03)
        ph += 2 * math.pi * f / SR
        s = math.sin(ph) * math.exp(-t / 0.12) * 1.1 + noise[k] * 1.8 * math.exp(-t / 0.05)
        out.append(level * math.tanh(1.5 * s) * (0.92 + 0.08 * rnd.random()))
    return out


def puff(level=1.0):
    """A breath of gas out of a pipe."""
    n = int(0.3 * SR)
    x = bandnoise(n, lambda k: 900, q=0.6)
    return [level * x[k] * 1.6 * math.sin(math.pi * min(1, k / (0.06 * SR)) / 2) * math.exp(-k / (0.09 * SR)) for k in range(n)]


def rumble(dur, f_at, level=1.0):
    """An engine's low hum whose pitch follows f_at(t)."""
    n = int(dur * SR)
    out = []
    ph = 0.0
    for k in range(n):
        f = f_at(k / SR)
        ph += 2 * math.pi * f / SR
        out.append(level * (math.sin(ph) + 0.4 * math.sin(2 * ph) + 0.2 * math.sin(3 * ph)))
    return env_fade(out, 0.3, 0.4)


_SEMI = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}


def hz(name):
    """Note name to frequency: "A4" = 440, "F#3", "Bb2"."""
    base, rest = name[0], name[1:]
    shift = 0
    while rest and rest[0] in "#b":
        shift += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    return 440.0 * 2 ** ((_SEMI[base] + shift + 12 * (int(rest) - 4)) / 12)


def chords(*specs):
    """chords("Fmaj7:F3 A3 C4 E4/F2", ...) -> bed() chord list; the part after / is the bass root."""
    out = []
    for spec in specs:
        notes, root = spec.split(":", 1)[1].split("/")
        out.append(([hz(n) for n in notes.split()], hz(root.strip())))
    return out


def beep(freq, dur=0.12, level=1.0):
    """A clean electronic beep: a keypad, a timer, a piezo horn."""
    n = int(dur * SR)
    a = max(1, int(0.004 * SR))
    return [level * math.sin(2 * math.pi * freq * k / SR) * min(1, k / a, (n - k) / a) for k in range(n)]


def rattle(dur, rate=18.0, f=3200, seed=0, level=1.0):
    """Many small hard things settling: coins in a box, shavings, capsules."""
    rnd = random.Random(seed)
    n = int(dur * SR)
    out = zeros(n)
    t = 0.0
    while t < dur:
        k = int(t * SR)
        tick = [math.sin(2 * math.pi * f * (0.7 + 0.6 * rnd.random()) * j / SR) * math.exp(-j / (0.004 * SR)) for j in range(int(0.03 * SR))]
        add(out, tick, k, level * (0.4 + 0.6 * rnd.random()) * math.sin(math.pi * min(1.0, t / dur)))
        t += rnd.expovariate(rate)
    return out


def hiss(dur, fc=5200, level=1.0):
    """Steam or air escaping: bright band noise with a soft swell."""
    n = int(dur * SR)
    x = bandnoise(n, lambda k: fc, q=0.5)
    return [x[k] * level * 1.3 * min(1, k / (0.15 * SR), (n - k) / (0.25 * SR)) for k in range(n)]


def spans(tl, field, pred):
    """Intervals (t0, t1) in seconds where pred(sampled value of field) holds, from a timeline made with --sample."""
    i = tl["fields"].index(field)
    rate, out, start = tl["rate"], [], None
    for k, row in enumerate(tl["samples"]):
        on = bool(pred(row[i]))
        if on and start is None:
            start = k / rate
        elif not on and start is not None:
            out.append((start, k / rate))
            start = None
    if start is not None:
        out.append((start, len(tl["samples"]) / rate))
    return out


def rising(tl, field, pred):
    """Moments where pred(sampled value) turns true."""
    return [a for a, _ in spans(tl, field, pred)]
