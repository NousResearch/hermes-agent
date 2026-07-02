#!/usr/bin/env python3
"""rejudge.py — A/B realized-change INSTRUMENTS on stored response pairs (no re-derivation).

Input: a validate_evsi.py run made with --keep-responses (rows carry baseline_resp/new_resp).
For each pair it re-judges realized change with the GRADED (mid-scale-anchored) judge and compares
against the ORIGINAL judge's stored score on identical texts:

  * saturation — frac of scores exactly at 0/1 (the original sits ~71% at the endpoints; a better
    instrument spreads the middle WITHOUT reshuffling the order)
  * agreement  — Spearman ρ between the two instruments (high = same ordering, safe to adopt)
  * prediction — pooled ρ of q_value vs each instrument (does de-saturation sharpen or blur the
    projected→realized link?)

Adoption rule (do-no-harm): adopt graded only if endpoint mass drops substantially AND agreement
with the original stays high AND the q_value link does not degrade.

  python3 evals/rejudge.py /path/tier2.json --judge-model fast --out /path/rejudged.json
"""

import argparse
import json
import os
import statistics as st
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

import pipeline  # noqa: E402
import validate_evsi  # noqa: E402
from analyze_evsi import spearman  # noqa: E402


def saturation(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"n": len(vals),
            "frac_0": round(sum(1 for v in vals if v == 0.0) / len(vals), 3),
            "frac_1": round(sum(1 for v in vals if v == 1.0) / len(vals), 3),
            "frac_endpoints": round(sum(1 for v in vals if v in (0.0, 1.0)) / len(vals), 3),
            "mean": round(st.mean(vals), 3), "sd": round(st.pstdev(vals), 3),
            "distinct": len({round(v, 2) for v in vals})}


def rejudge_rows(rows, judge_model, timeout, progress=True):
    """One graded re-judge per stored pair. Returns slim comparison rows."""
    out = []
    for i, r in enumerate(rows):
        graded = validate_evsi.change_judge_graded(
            r.get("prompt_text") or r["prompt"], r["baseline_resp"], r["new_resp"],
            judge_model, timeout)
        out.append({"prompt": r["prompt"], "question": r["question"], "answer": r.get("answer"),
                    "q_value": r.get("q_value"), "orig": r["realized_change"],
                    "graded": None if graded is None else round(graded, 3)})
        if progress:
            print(f"  {i + 1}/{len(rows)} orig={r['realized_change']} graded={graded}",
                  file=sys.stderr, flush=True)
    return out


def compare(pairs):
    """Instrument comparison over rejudged pairs (orig vs graded on identical texts)."""
    ok = [p for p in pairs if p["orig"] is not None and p["graded"] is not None]
    orig, graded = [p["orig"] for p in ok], [p["graded"] for p in ok]
    qv = [p["q_value"] for p in ok if p.get("q_value") is not None]
    stats = {
        "n": len(ok),
        "orig": saturation(orig), "graded": saturation(graded),
        "agreement_rho": spearman(orig, graded),
        "qvalue_vs_orig_rho": spearman([p["q_value"] for p in ok], orig) if len(qv) == len(ok) else None,
        "qvalue_vs_graded_rho": spearman([p["q_value"] for p in ok], graded) if len(qv) == len(ok) else None,
    }
    return stats


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("path", help="validate_evsi output made with --keep-responses")
    p.add_argument("--judge-model", default="fast")
    p.add_argument("--max-rows", type=int, default=0, help="cap pairs (0 = all), evenly sampled.")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--out")
    args = p.parse_args(argv)

    with open(args.path) as f:
        data = json.load(f)
    rows = [r for r in data.get("rows", [])
            if r.get("baseline_resp") and r.get("new_resp") and r.get("realized_change") is not None
            and r.get("method", "absolute") == "absolute"]  # ab runs: one judged row per pair
    if not rows:
        print("no rows with stored responses — rerun validate_evsi with --keep-responses",
              file=sys.stderr)
        return 2
    if args.max_rows and len(rows) > args.max_rows:
        step = len(rows) / args.max_rows
        rows = [rows[int(i * step)] for i in range(args.max_rows)]

    t0 = time.time()
    pairs = rejudge_rows(rows, pipeline.resolve_alias(args.judge_model), args.timeout)
    stats = compare(pairs)
    out = {"pairs": pairs, "stats": stats, "judge_model": args.judge_model,
           "source": args.path, "elapsed_s": round(time.time() - t0, 1)}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
