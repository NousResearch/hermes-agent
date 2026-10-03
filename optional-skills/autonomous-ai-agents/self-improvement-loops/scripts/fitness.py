#!/usr/bin/env python3
"""Weighted fitness score for a goal, with an optional append-only history.

    python fitness.py SPEC.json '{"speed": 0.8, "accuracy": 0.6}' [--log HISTORY.jsonl]

SPEC.json: {"name": "...", "target": "...", "dimensions": [{"name": "speed", "weight": 0.5}, ...]}
Weights must be positive and sum to 1; every dimension needs exactly one score in [0, 1].
Prints JSON: score, per-dimension contribution, and the delta against the last logged run of
the same spec. The arithmetic lives here so the model never computes a weighted sum in prose.
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


class SpecError(ValueError):
    pass


def validate_spec(spec):
    dims = spec.get("dimensions")
    if not isinstance(spec.get("name"), str) or not spec["name"].strip():
        raise SpecError("spec needs a non-empty 'name'")
    if not isinstance(dims, list) or not dims:
        raise SpecError("spec needs a non-empty 'dimensions' list")
    names = [d.get("name") for d in dims]
    if len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        raise SpecError("dimension names must be unique non-empty strings")
    weights = [d.get("weight") for d in dims]
    if not all(isinstance(w, (int, float)) and not isinstance(w, bool) and w > 0 for w in weights):
        raise SpecError("every dimension weight must be a positive number")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise SpecError(f"dimension weights must sum to 1 (got {sum(weights):.6g})")


def score(spec, scores):
    validate_spec(spec)
    expected = {d["name"] for d in spec["dimensions"]}
    if set(scores) != expected:
        missing, extra = sorted(expected - set(scores)), sorted(set(scores) - expected)
        raise SpecError(f"scores must cover exactly the spec's dimensions (missing={missing}, extra={extra})")
    for name, value in scores.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise SpecError(f"score for {name!r} must be a number in [0, 1]")
    breakdown = {d["name"]: d["weight"] * scores[d["name"]] for d in spec["dimensions"]}
    return {"name": spec["name"], "score": round(sum(breakdown.values()), 6),
            "breakdown": {k: round(v, 6) for k, v in breakdown.items()}}


def last_logged(log_path, name):
    if not log_path.exists():
        return None
    previous = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line) if line.strip() else None
        if entry and entry.get("name") == name:
            previous = entry
    return previous


def main(argv=None):
    parser = argparse.ArgumentParser(description="Weighted fitness score for a goal.")
    parser.add_argument("spec", type=Path, help="fitness spec JSON file")
    parser.add_argument("scores", help="JSON object mapping dimension name to a score in [0, 1]")
    parser.add_argument("--log", type=Path, help="append the result to this JSONL history file")
    args = parser.parse_args(argv)
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        scores = json.loads(args.scores)
        if not isinstance(scores, dict):
            raise SpecError("scores must be a JSON object")
        result = score(spec, scores)
    except (OSError, json.JSONDecodeError, SpecError) as exc:
        print(f"fitness: {exc}", file=sys.stderr)
        return 2
    previous = last_logged(args.log, result["name"]) if args.log else None
    result["previous"] = previous["score"] if previous else None
    result["delta"] = round(result["score"] - previous["score"], 6) if previous else None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        entry = {**result, "scores": scores, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with args.log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
