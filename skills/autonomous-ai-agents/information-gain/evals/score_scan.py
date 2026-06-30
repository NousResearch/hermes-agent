#!/usr/bin/env python3
"""score_scan.py — cheap value-structure scan across the test bank (no realized_change).

Runs info-gain once per prompt and records the distribution of the scoring components over
ALL scored candidates (pre-gate/threshold): U, EVSI, value, stakes, Δ, derivable_prob, bucket
size, and how many candidates fall below the discard threshold. Aggregates by category so we can
see whether/how different KINDS of Hermes tasks produce different value structures.

Light (one info-gain per prompt, no per-pair re-derivation), incremental per-prompt writes —
robust to a mid-run kill. Run on the host:
  OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes \
    python3 evals/score_scan.py --out /Users/dadleet/.hermes/score_scan.json
"""

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

import infogain  # noqa: E402
import testbank  # noqa: E402


def _agg(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "mean": round(st.mean(xs), 3),
            "sd": round(st.pstdev(xs), 3), "min": round(min(xs), 3), "max": round(max(xs), 3)}


def scan_prompt(pr, cfg):
    r = infogain.run(pr["problem"], cfg)
    sa = r.get("all_scored", [])
    ans = [a for q in sa for a in (q.get("answers") or [])]
    thr = cfg["discard_threshold"]
    vals = [q.get("value", 0) for q in sa]
    return {
        "id": pr["id"], "cat": pr.get("cat", "?"),
        "candidates": len(sa), "bucket": len(r["bucket"]),
        "frac_below_thr": round(sum(1 for v in vals if v < thr) / len(vals), 3) if vals else None,
        "U": [q.get("u") for q in sa], "evsi": [q.get("evsi") for q in sa],
        "value": vals, "derivable": [q.get("derivable_prob") for q in sa],
        "stakes": [a.get("stakes") for a in ans], "delta": [a.get("delta_plan") for a in ans],
    }


def render(rows, thr):
    keys = ["U", "value", "evsi", "derivable", "stakes", "delta"]
    print(f"\n{'id':<18}{'cat':<15}{'cand':>5}{'buck':>5}{'<thr':>6}  "
          f"{'U_mean(sd)':>12}{'val_mean':>9}{'deriv':>7}")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["cat"]].append(r)
        um, us = _agg(r["U"]) and (_agg(r["U"])["mean"], _agg(r["U"])["sd"]) or (0, 0)
        va = _agg(r["value"])
        de = _agg(r["derivable"])
        print(f"{r['id']:<18}{r['cat']:<15}{r['candidates']:>5}{r['bucket']:>5}"
              f"{(r['frac_below_thr'] or 0):>6.0%}  {um:>7.2f}({us:.2f}){(va['mean'] if va else 0):>9.2f}"
              f"{(de['mean'] if de else 0):>7.2f}")

    print(f"\n=== by category (mean over all candidates; discard_threshold={thr}) ===")
    print(f"{'category':<15}{'prompts':>8}{'buckets':>8}  {'U_sd':>6}{'U_mean':>7}"
          f"{'value':>7}{'evsi':>6}{'deriv':>7}{'stakes':>7}{'<thr':>6}")
    cat_summary = {}
    for cat, rs in sorted(by_cat.items()):
        pool = {k: [v for r in rs for v in r[k]] for k in keys}
        u, va, ev, de, stk = (_agg(pool["U"]), _agg(pool["value"]), _agg(pool["evsi"]),
                              _agg(pool["derivable"]), _agg(pool["stakes"]))
        below = st.mean([r["frac_below_thr"] for r in rs if r["frac_below_thr"] is not None] or [0])
        bsum = sum(r["bucket"] for r in rs)
        cat_summary[cat] = {"prompts": len(rs), "buckets": bsum, "U": u, "value": va,
                            "evsi": ev, "derivable": de, "stakes": stk, "frac_below": round(below, 3)}
        print(f"{cat:<15}{len(rs):>8}{bsum:>8}  {(u['sd'] if u else 0):>6.2f}{(u['mean'] if u else 0):>7.2f}"
              f"{(va['mean'] if va else 0):>7.2f}{(ev['mean'] if ev else 0):>6.2f}"
              f"{(de['mean'] if de else 0):>7.2f}{(stk['mean'] if stk else 0):>7.2f}{below:>6.0%}")
    return cat_summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--ids", nargs="*", help="subset of prompt ids (default: whole bank).")
    p.add_argument("--gen-model", default="fast")
    p.add_argument("--include-life", action="store_true", help="also scan the LIFE control pool.")
    args = p.parse_args(argv)

    cfg = dict(infogain.DEFAULTS)
    for k in ("plan_model", "question_gen_model", "answer_model"):
        cfg[k] = args.gen_model
    cfg["max_rounds"] = 1
    cfg["mode"] = "focus"  # value_judge stays at the shipped deepseek default

    pool = testbank.ALL if args.include_life else testbank.LIFE + testbank.BANK
    prompts = [x for x in pool if not args.ids or x["id"] in args.ids]
    rows, t0 = [], time.time()
    for pr in prompts:
        print(f"… scan {pr['id']} ({pr['cat']})", file=sys.stderr, flush=True)
        try:
            rows.append(scan_prompt(pr, cfg))
        except Exception as e:
            rows.append({"id": pr["id"], "cat": pr.get("cat"), "error": str(e),
                         "candidates": 0, "bucket": 0, "frac_below_thr": None,
                         "U": [], "evsi": [], "value": [], "derivable": [], "stakes": [], "delta": []})
        if args.out:  # incremental — survive a kill
            with open(args.out, "w") as f:
                json.dump({"rows": rows, "n": len(rows), "partial": True,
                           "elapsed_s": round(time.time() - t0, 1)}, f, indent=2, default=str)

    cat = render([r for r in rows if not r.get("error")], cfg["discard_threshold"])
    out = {"rows": rows, "n": len(rows), "partial": False, "by_category": cat,
           "discard_threshold": cfg["discard_threshold"], "elapsed_s": round(time.time() - t0, 1)}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {args.out} ({len(rows)} prompts, {out['elapsed_s']}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
