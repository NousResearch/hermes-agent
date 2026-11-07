"""Score for the Cessna 337 main landing gear film: a light bed (Dmaj9 chords, 96 bpm) at one level the whole way, a
hydraulic whine under every gear transit, a soft air at each lift-off, a tyre thud at each touchdown and a quiet
click on every button the operator presses.

    python run_sfx.py sims/cessna-shared/score.py

Times from cessna-shared/film.js (SCRIPT) and flight.js (the sequences' own timings).
"""

import os

from sfx.score_kit import airy, bed, chords, hz, motor, thud
from sfx.synth import click

TIMELINE = "../cessna-film/timeline.json"
OUT = os.path.join(os.environ.get("TEARDOWN_OUT", "teardown-out"), "cessna-score.wav")
TARGET_LUFS = -14.0
SEED = 17

CHORDS = chords("Dmaj9:D2 A2 C#3 E3 F#3/D1", "Bm9:B1 F#2 A2 C#3 D3/B0", "Gmaj7:G2 D3 F#3 B3/G1", "A6sus:A2 D3 E3 F#3 B3/A1")
MOTIF = [(0.5, hz("F#5")), (1.5, hz("A5")), (2.5, hz("E5"))]
CLICKS = (2.4, 5.4, 5.9, 6.5, 8.7, 10.9, 11.5, 12.7, 13.0, 13.4, 16.2, 17.8, 18.3, 18.7, 19.1)
TAKEOFFS = (0.0, 13.0)
LANDINGS = (5.4, 19.1)


def score(mix, tl):
    dur = tl["duration"]
    left, right, low, drums = bed(dur, 96, CHORDS, MOTIF, motif_voice="epiano", kicks=(0, 2.5), snares=(), hats=8, swing=0.0, keys_vel=0.7, keys_lp=3600)
    mix.put_stereo(0, left, right, 0.12)
    mix.put(0, low, 0.12)
    mix.put(0, drums, 0.045)
    for t0 in TAKEOFFS:
        mix.put(t0 + 1.8, airy(1.6, 500, 1600, 0.6), 0.03)
        mix.put(t0 + 2.3, motor(2.2, 190, 1300, 14), 0.02)
    for t0 in LANDINGS:
        mix.put(t0 + 0.6, motor(2.4, 175, 1200, 14), 0.02)
        if t0 + 5.2 < dur - 0.3:
            mix.put(t0 + 5.2, thud(80, 0.05, 0.4), 0.09)
    for t in CLICKS:
        mix.put(t, click(2400, 0.025), 0.055)
