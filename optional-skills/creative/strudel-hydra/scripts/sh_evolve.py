#!/usr/bin/env python3
"""Evaluate one generation of candidate sets against a fitness target.

The agent is the mutation operator: it writes candidate sets (variations of the
current best) and calls this to *measure and rank* them. It pushes each
candidate, lets it settle, reads the telemetry the page reports, scores it
against the target vector, and prints the candidates sorted best-first. The
agent then keeps the winner, mutates again, and repeats — a closed
perception→action→selection loop. Run it on an interval with the `/loop` skill
for an autonomous, self-improving VJ.

Candidates JSON (a file, or stdin with `-`) is a list of objects:
    [{"label": "a", "audio": "...", "visual": "..."}, ...]

Example:
    python3 sh_evolve.py --target '{"brightness":0.5,"motion":0.35,"level":0.6}' \
        --wait 1.5 cands.json
"""
import argparse
import json
import sys
import time

from sh_client import observe, push_set, score


def evaluate(base, candidates, target, wait):
    ranked = []
    for c in candidates:
        push_set(base, c.get("audio"), c.get("visual"), c.get("label", "cand"))
        time.sleep(wait)  # let the set play + the page report a fresh measurement
        resp = observe(base)
        features = resp.get("data")
        ranked.append({
            "label": c.get("label", "cand"),
            "score": score(features, target),
            "features": features,
            "age": resp.get("age"),
        })
    # Unscored candidates (no telemetry yet) sort last.
    ranked.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0)))
    return ranked


def main():
    ap = argparse.ArgumentParser(description="score + rank one generation of candidate sets")
    ap.add_argument("candidates", help="JSON file of candidate sets, or - for stdin")
    ap.add_argument("--target", required=True, help="JSON fitness target, e.g. '{\"brightness\":0.5}'")
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--wait", type=float, default=1.5, help="settle seconds between push and observe")
    args = ap.parse_args()

    raw = sys.stdin.read() if args.candidates == "-" else open(args.candidates, encoding="utf-8").read()
    candidates = json.loads(raw)
    if not isinstance(candidates, list) or not candidates:
        sys.exit("candidates must be a non-empty JSON list")
    target = json.loads(args.target)

    ranked = evaluate(args.base, candidates, target, args.wait)
    print(json.dumps({"target": target, "ranked": ranked}, indent=2))
    best = ranked[0]
    print(f"\nbest: {best['label']}  score={best['score']}", file=sys.stderr)


if __name__ == "__main__":
    main()
