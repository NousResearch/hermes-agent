"""Score for the V8 four-stroke explorer: a cool bed (Am9 chords, 92 bpm) at one level the whole way, a soft low thump
on every power stroke (one every 90 degrees of crank, so the beat tightens when the operator opens the throttle), a
quiet click on every button press and a rising air under the throttle drag.

    python run_sfx.py sims/v8-shared/score.py

Times and the crank model mirror v8-shared/film.js (SCRIPT, rpmAt, 1/20 playback).
"""

import os

from sfx.score_kit import airy, bed, chords, hz, thud
from sfx.synth import click

TIMELINE = "../v8-film/timeline.json"
OUT = os.path.join(os.environ.get("TEARDOWN_OUT", "teardown-out"), "v8-score.wav")
TARGET_LUFS = -14.0
SEED = 13

CHORDS = chords("Am9:A2 E3 G3 B3 C4/A1", "Fmaj7:F2 C3 E3 A3/F1", "Cmaj7:C3 E3 G3 B3/C2", "G6:G2 D3 E3 B3/G1")
MOTIF = [(0.5, hz("E5")), (1.5, hz("C5")), (2.5, hz("B4"))]
CLICKS = (3.5, 6.9, 9.7, 12.1, 14.7, 17.3, 19.9, 22.5)
RPM = (12.1, 700.0, 2400.0, 1.4)


def smoother(x):
    u = min(1.0, max(0.0, x))
    return u * u * u * (u * (u * 6 - 15) + 10)


def rpm_at(t):
    t0, a, b, d = RPM
    return a + (b - a) * smoother((t - t0) / d)


def firings(dur, dt=1 / 480):
    out, th, t, nxt = [], 12.0, 0.0, 90.0
    while t < dur:
        th += rpm_at(t) / 60 * 360 * dt / 20
        if th >= nxt:
            out.append(t)
            nxt += 90
        t += dt
    return out


def score(mix, tl):
    dur = tl["duration"]
    left, right, low, drums = bed(dur, 92, CHORDS, MOTIF, motif_voice="epiano", kicks=(0, 2.5), snares=(), hats=8, swing=0.0, keys_vel=0.7, keys_lp=3600)
    mix.put_stereo(0, left, right, 0.12)
    mix.put(0, low, 0.12)
    mix.put(0, drums, 0.04)
    for t in firings(dur - 0.3):
        mix.put(t, thud(62, 0.05, 0.3), 0.045)
    for t in CLICKS:
        mix.put(t, click(2400, 0.025), 0.055)
    mix.put(RPM[0], airy(RPM[3] + 0.6, 400, 1600, 0.6), 0.03)
    # every camera move gets a soft air
    for t in (3.5, 9.7, 14.7, 17.3, 19.9):
        mix.put(t, airy(1.2, 700, 1500, 0.6), 0.018)
