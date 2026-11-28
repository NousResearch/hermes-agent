#!/usr/bin/env python
"""Beat map for a music track, so a sim can cut and hit on the music.

    python sims/beats.py music/track.mp3 [--duration 30] [--start auto|SECONDS] [-o out.json]

Decodes with ffmpeg, then in pure Python:
- tempo from the autocorrelation of the onset envelope (prefers 100-160 BPM),
- the beat grid, aligned to the onsets,
- downbeats (the beat phase with the most low-band attack),
- kicks (low-band onset peaks, with strength 0..1),
- energy per bar, and the drop (the biggest sustained jump in energy).

With --start auto the 30 s window is placed so the drop lands at ~40 % of the
clip, snapped to a bar. All times in the output are relative to the window start.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from array import array

FFMPEG = os.environ.get('FFMPEG', 'ffmpeg')
SR = 11025
HOP = 256
FPS = SR / HOP  # envelope frames per second (~43)


def decode(path, filt=None):
    cmd = [FFMPEG, '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR)]
    if filt:
        cmd += ['-af', filt]
    cmd += ['-f', 's16le', '-']
    raw = subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=300).stdout
    a = array('h')
    a.frombytes(raw[: len(raw) // 2 * 2])
    return a


def frame_energy(samples):
    n = len(samples) // HOP
    out = array('d', bytes(8 * n))
    for i in range(n):
        s = 0
        for x in samples[i * HOP:(i + 1) * HOP]:
            s += x * x
        out[i] = s / HOP
    return out


def onset(env):
    """Positive change of log energy, lightly smoothed."""
    lg = [math.log(1e-3 + e) for e in env]
    d = [0.0] + [max(0.0, lg[i] - lg[i - 1]) for i in range(1, len(lg))]
    return [0.5 * d[i] + 0.25 * (d[i - 1] if i else 0) + 0.25 * (d[i + 1] if i + 1 < len(d) else 0)
            for i in range(len(d))]


def tempo(nov):
    """Best period in frames from the autocorrelation, weighted toward ~126 BPM."""
    n = len(nov)
    mean = sum(nov) / n
    x = [v - mean for v in nov]
    best, best_lag = -1e18, None
    scores = {}
    lo = int(FPS * 60 / 190)
    hi = int(FPS * 60 / 70) + 1
    for lag in range(lo, hi):
        s = 0.0
        for i in range(n - lag):
            s += x[i] * x[i + lag]
        bpm = 60 * FPS / lag
        w = math.exp(-0.5 * (math.log2(bpm / 126) / 0.6) ** 2)
        scores[lag] = s
        if s * w > best:
            best, best_lag = s * w, lag
    # parabolic refinement
    l = best_lag
    if l - 1 in scores and l + 1 in scores:
        a, b, c = scores[l - 1], scores[l], scores[l + 1]
        den = a - 2 * b + c
        if den:
            l = l + 0.5 * (a - c) / den
    return l


def beat_grid(nov, period):
    """Phase of the grid that sits on the most onset energy, then local snapping."""
    n = len(nov)
    best, best_off = -1, 0
    for off in range(int(period) + 1):
        s, k = 0.0, 0
        while True:
            i = int(round(off + k * period))  # noqa: RUF046
            if i >= n:
                break
            s += nov[i]
            k += 1
        if s > best:
            best, best_off = s, off
    beats = []
    while best_off - period >= 0:  # grid back to the very start
        best_off -= period
    k = 0
    while True:
        c = best_off + k * period
        if c >= n:
            break
        i = int(round(c))  # noqa: RUF046
        lo, hi = max(0, i - 2), min(n - 1, i + 2)
        j = max(range(lo, hi + 1), key=lambda q: nov[q])
        # only snap if the peak is clearly there, else keep the grid
        beats.append(j if nov[j] > 1.5 * nov[i] + 1e-9 else i)
        k += 1
    return beats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('track')
    ap.add_argument('--duration', type=float, default=30)
    ap.add_argument('--start', default='auto')
    ap.add_argument('-o', '--out')
    args = ap.parse_args()

    full = decode(args.track)
    low = decode(args.track, 'lowpass=f=140,lowpass=f=140')
    total = len(full) / SR
    env = frame_energy(full)
    envl = frame_energy(low)
    nov = onset(env)
    novl = onset(envl)
    comb = [a + 1.5 * b for a, b in zip(nov, novl)]

    period = tempo(comb)
    bpm = 60 * FPS / period
    beats = beat_grid(comb, period)
    # downbeat phase: most low-band attack
    phase = max(range(4), key=lambda p: sum(novl[b] for b in beats[p::4]))
    downs = beats[phase::4]

    # kicks: low-band onset peaks above an adaptive threshold
    srt = sorted(novl)
    thr = srt[int(len(srt) * 0.9)]
    peak = max(novl) or 1
    kicks = []
    for i in range(1, len(novl) - 1):
        if novl[i] > thr and novl[i] >= novl[i - 1] and novl[i] > novl[i + 1]:
            if not kicks or i - kicks[-1][0] > FPS * 0.18:
                kicks.append((i, novl[i] / peak))

    # energy per bar
    bar_e = []
    for b in range(len(downs) - 1):
        seg = env[downs[b]:downs[b + 1]]
        bar_e.append(math.sqrt(sum(seg) / max(1, len(seg))))
    mx = max(bar_e) or 1
    bar_e = [e / mx for e in bar_e]

    # drop: biggest jump vs the previous 4 bars that holds for the next 2
    drop_bar, best = None, 0
    for b in range(4, len(bar_e) - 2):
        before = sum(bar_e[b - 4:b]) / 4
        after = min(bar_e[b:b + 2])
        j = after - before
        if j > best:
            best, drop_bar = j, b
    drop_t = downs[drop_bar] / FPS if drop_bar is not None else None

    D = args.duration
    if args.start == 'auto':
        if drop_t is not None and best > 0.12:
            want = drop_t - 0.4 * D
        else:
            # loudest window
            nb = max(1, int(D / (240 / bpm)))
            want_bar = max(range(max(1, len(bar_e) - nb)), key=lambda b: sum(bar_e[b:b + nb]))
            want = downs[want_bar] / FPS
        want = min(max(0.0, want), max(0.0, total - D))
        start = min((d / FPS for d in downs), key=lambda d: abs(d - want))
        if start + D > total:
            start = max(0.0, total - D)
    else:
        start = float(args.start)
    end = start + D

    rel = lambda f: round(f / FPS - start, 3)
    inwin = lambda f: start - 1e-6 <= f / FPS < end
    bar_starts = [d for d in downs if inwin(d)]
    out = {
        'track': os.path.abspath(args.track),
        'start': round(start, 3),
        'duration': D,
        'bpm': round(bpm, 2),
        'beats': [rel(b) for b in beats if inwin(b)],
        'downbeats': [rel(d) for d in bar_starts],
        'kicks': [[rel(i), round(s, 3)] for i, s in kicks if inwin(i)],
        'barEnergy': [round(bar_e[downs.index(d)], 3) for d in bar_starts if downs.index(d) < len(bar_e)],
        'drop': round(drop_t - start, 3) if drop_t is not None and start <= drop_t < end else None,
    }
    path = args.out or os.path.splitext(args.track)[0] + f'.beats-{int(round(start))}.json'  # noqa: RUF046
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f'{os.path.basename(args.track)}: {out["bpm"]} bpm, window {start:.2f}-{end:.2f} s, '
          f'drop {"%.2f" % out["drop"] if out["drop"] is not None else "none"} (clip time), '  # noqa: UP031
          f'{len(out["beats"])} beats, {len(out["kicks"])} kicks -> {path}')


if __name__ == '__main__':
    sys.exit(main())
