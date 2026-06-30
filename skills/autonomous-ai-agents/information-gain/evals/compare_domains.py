#!/usr/bin/env python3
"""compare_domains.py — side-by-side of two validate_evsi runs (e.g. life vs agentic).

Tests whether the Phase-1 EVSI conclusions are domain-bound. For each domain prints the
distributions the conclusions hinge on:
  - U spread (was compressed 0.73-0.98 / inert in the life set — does it spread / discriminate?)
  - stakes spread + bimodality (EVSI was ~entirely stakes — is stakes more informative here?)
  - projected_delta spread; realized_change saturation (% at exactly 0/1)
  - per-answer Δ calibration Spearman(projected_delta, realized_change)
  - which formula best predicts realized_change (max-Δ vs EVSI vs mean-Δ vs U-only)
  - U-inertness: within-prompt order-changing pairs between √(U·EVSI) and EVSI-only

Usage:
  python3 evals/compare_domains.py life=/path/evsi_validation.json agentic=/path/evsi_validation_agentic.json
"""

import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from analyze_evsi import load_rows, by_question, spearman  # noqa: E402


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _frac(xs, pred):
    return sum(1 for x in xs if pred(x)) / len(xs) if xs else float("nan")


def reorderings(questions):
    """Within-prompt pairs whose order differs between √(U·EVSI) and EVSI-only. 0 == U inert."""
    by_prompt = defaultdict(list)
    for q in questions:
        by_prompt[q["prompt"]].append(q)
    flips = total = 0
    for qs in by_prompt.values():
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                total += 1
                a, b = qs[i], qs[j]
                sv = (a["q_value"] > b["q_value"]) - (a["q_value"] < b["q_value"])
                se = (a["q_evsi"] > b["q_evsi"]) - (a["q_evsi"] < b["q_evsi"])
                if sv != se:
                    flips += 1
    return flips, total


def formula_vs_realized(questions, target="realized_change"):
    by_prompt = defaultdict(list)
    for q in questions:
        by_prompt[q["prompt"]].append(q)
    out = {}
    for name, key in [("max-Δ", "max_delta"), ("EVSI/value", "q_evsi"),
                      ("mean-Δ", "mean_delta"), ("U-only", "q_u")]:
        rhos = []
        for qs in by_prompt.values():
            if len(qs) < 2:
                continue
            r = spearman([q[key] for q in qs], [q[target] for q in qs])
            if r is not None:
                rhos.append(r)
        out[name] = _mean(rhos) if rhos else float("nan")
    return out


def summarize(label, path):
    rows = load_rows(path)
    qs = by_question(rows)
    # attach P-weighted mean stakes per question
    g = defaultdict(list)
    for r in rows:
        g[(r["prompt"], r["question"])].append(r)
    for item in qs:
        rs = g[(item["prompt"], item["question"])]
        ptot = sum(max(0, x["prob"]) for x in rs) or 1
        item["mean_stakes"] = sum((max(0, x["prob"]) / ptot) * x["stakes"] for x in rs)

    U = [q["q_u"] for q in qs]
    stk = [r["stakes"] for r in rows]
    pd = [r["projected_delta"] for r in rows]
    rc = [r["realized_change"] for r in rows]
    flips, total = reorderings(qs)
    cal = spearman(pd, rc)

    print(f"\n{'='*64}\n{label.upper()}  —  {len(qs)} questions / {len(rows)} answers / "
          f"{len({q['prompt'] for q in qs})} prompts\n{'='*64}")
    print(f"  U (uncertainty)   mean={_mean(U):.3f}  std={_std(U):.3f}  "
          f"range=[{min(U):.3f},{max(U):.3f}]  spread={max(U)-min(U):.3f}")
    print(f"  stakes            mean={_mean(stk):.3f}  std={_std(stk):.3f}  "
          f"hi(>.7)={_frac(stk, lambda x: x>.7):.0%}  lo(<.3)={_frac(stk, lambda x: x<.3):.0%}  "
          f"(bimodal if hi+lo large, mid small)")
    print(f"  projected_delta   mean={_mean(pd):.3f}  std={_std(pd):.3f}")
    print(f"  realized_change   mean={_mean(rc):.3f}  saturation(0 or 1)="
          f"{_frac(rc, lambda x: x<=0.01 or x>=0.99):.0%}")
    print(f"  Δ calibration     per-answer Spearman(projected_delta, realized_change) = "
          f"{cal:+.3f}" if cal is not None else "  Δ calibration n/a")
    print(f"  U inert?          within-prompt order-changing pairs = {flips}/{total}  "
          f"({'INERT' if flips==0 else 'U reorders'})")
    print(f"  best formula vs realized_change (mean per-prompt Spearman):")
    fr = formula_vs_realized(qs)
    for name, v in sorted(fr.items(), key=lambda kv: -kv[1] if kv[1]==kv[1] else 1):
        print(f"      {name:<12} {v:+.3f}")


def main(argv):
    pairs = [a.split("=", 1) for a in argv[1:] if "=" in a]
    if not pairs:
        print("usage: compare_domains.py life=PATH agentic=PATH", file=sys.stderr)
        return 1
    for label, path in pairs:
        try:
            summarize(label, path)
        except Exception as e:
            print(f"\n{label}: ERROR {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
