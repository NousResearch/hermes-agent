"""
show_snapshot.py — Dump the population from a darwinian-evolver snapshot pickle.

Usage:
    python show_snapshot.py PATH/TO/iteration_N.pkl [--field prompt_template]

The script is intentionally Organism-agnostic: it walks `org.__dict__` and prints
all str fields. By default it shows `prompt_template` if present; pass --field to
target a different attribute (e.g. `regex_pattern`, `sql_query`, `code_block`).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", type=Path)
    ap.add_argument(
        "--field",
        default=None,
        help="Organism attribute to display. Defaults to the first str field found.",
    )
    ap.add_argument("--top", type=int, default=None, help="Show only top N by score.")
    args = ap.parse_args()

    if not args.snapshot.exists():
        sys.exit(f"snapshot not found: {args.snapshot}")

    # The outer pickle wraps a dict; the inner pickle contains the actual organism
    # objects, which must be importable under their original dotted path. If you
    # ran a custom driver, make sure its module is on sys.path before calling this.
    outer = pickle.loads(args.snapshot.read_bytes())
    if not isinstance(outer, dict) or "population_snapshot" not in outer:
        sys.exit("not a darwinian-evolver snapshot (no population_snapshot key)")
    inner = pickle.loads(outer["population_snapshot"])
    pairs = inner["organisms"]  # list of (Organism, EvaluationResult)

    print(f"# organisms: {len(pairs)}\n")
    ranked = sorted(pairs, key=lambda p: getattr(p[1], "score", 0) or 0, reverse=True)
    if args.top:
        ranked = ranked[: args.top]

    for i, (org, res) in enumerate(ranked):
        score = getattr(res, "score", float("nan"))
        print(f"=== rank {i} score={score:.3f} ===")
        # pick field
        field = args.field
        if field is None:
            for k, v in vars(org).items():
                if isinstance(v, str) and not k.startswith("_") and k not in ("id",):
                    field = k
                    break
        val = getattr(org, field, None) if field else None
        if val is None:
            print(f"  (no string field; org fields: {list(vars(org).keys())})")
        else:
            print(f"  {field} ({len(val)} chars):")
            for ln in val.splitlines()[:30]:
                print(f"    {ln}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
