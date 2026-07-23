#!/usr/bin/env python3
"""Push audio/visual sets to a running strudel-hydra server.

Standard library only. A "set" is JSON: an `audio` string (strudel pattern
code), a `visual` string (hydra code), or both, plus an optional `label`. The
server relays it over SSE and every open browser hot-swaps to it.
"""
import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request


def post(base, path, data, timeout=5):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(base, path, timeout=5):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read())


def push_set(base, audio=None, visual=None, label="live"):
    """Convenience used by sh_examples.py."""
    payload = {"label": label}
    if audio is not None:
        payload["audio"] = audio
    if visual is not None:
        payload["visual"] = visual
    return post(base, "/push", payload)


def observe(base):
    """Latest measured features: {"data": {...} | None, "age": seconds}."""
    return get(base, "/telemetry")


# Scalar audio feature keys usable as fitness targets (the `bands` array is
# skipped — target overall energy via `level`, spectral tilt via `centroid`).
SCORABLE = ("level", "centroid")


def score(features, target):
    """Fitness in (0, 1]: 1 / (1 + euclidean distance) over shared scalar keys.
    Higher is closer to the target. Returns None if nothing comparable."""
    if not features or not target:
        return None
    sq, used = 0.0, 0
    for k in SCORABLE:
        if k in target and isinstance(features.get(k), (int, float)):
            sq += (float(features[k]) - float(target[k])) ** 2
            used += 1
    if not used:
        return None
    return round(1.0 / (1.0 + math.sqrt(sq)), 4)


def main():
    ap = argparse.ArgumentParser(description="push a set to the strudel-hydra server")
    ap.add_argument("--base", default="http://127.0.0.1:8765", help="server base URL")
    ap.add_argument("--audio", help="strudel pattern code")
    ap.add_argument("--visual", help="hydra code")
    ap.add_argument("--audio-file", help="read strudel code from a file")
    ap.add_argument("--visual-file", help="read hydra code from a file")
    ap.add_argument("--label", default="live")
    ap.add_argument("--hush", action="store_true", help="panic: stop audio + clear visuals")
    ap.add_argument("--status", action="store_true", help="print connected browser count")
    ap.add_argument("--observe", action="store_true", help="print latest measured features")
    ap.add_argument("--target", help="JSON target vector; with --observe, adds a fitness score")
    ap.add_argument("--wait", type=float, default=0.0, help="seconds to sleep before observing")
    args = ap.parse_args()

    try:
        if args.status:
            print(json.dumps(get(args.base, "/status")))
            return
        if args.observe:
            if args.wait:
                time.sleep(args.wait)
            resp = observe(args.base)
            if args.target:
                resp["score"] = score(resp.get("data"), json.loads(args.target))
            print(json.dumps(resp))
            return
        if args.hush:
            print(json.dumps(post(args.base, "/push", {"cmd": "hush"})))
            return

        audio = args.audio
        if args.audio_file:
            audio = open(args.audio_file, encoding="utf-8").read()
        visual = args.visual
        if args.visual_file:
            visual = open(args.visual_file, encoding="utf-8").read()
        if audio is None and visual is None:
            ap.error("nothing to push: give --audio/--visual (or --hush)")

        print(json.dumps(push_set(args.base, audio, visual, args.label)))
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach {args.base}: {e.reason}  (is sh_server.py running?)")


if __name__ == "__main__":
    main()
