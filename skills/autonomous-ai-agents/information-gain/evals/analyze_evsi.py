#!/usr/bin/env python3
"""analyze_evsi.py — post-hoc analysis of validate_evsi.py output (P1a + P1c). No model calls.

Reads the rows JSON (one row per question×answer with projected vs realized change) and reports:

  P1a calibration
    - per-answer correlation: does projected `delta_plan` predict `realized_change`?
      (Pearson + Spearman + a binned calibration curve + saturation diagnostics)
    - per-question: does projected EVSI / value track the question's REALIZED value
      (Σ_a P'(a)·realized_change(a), P' = prob renormalized over tested answers)?

  P1c formula ablations (the near-free study)
    - rank questions per prompt under alternative projected formulas:
        value=√(U·EVSI)  |  EVSI-only  |  U-only  |  max-Δ  |  mean-Δ (P-weighted)
      and measure which projected ranking best matches the REALIZED ranking
      (mean Spearman across prompts). The winner is the formula worth shipping.

Usage:  python3 evals/analyze_evsi.py /Users/dadleet/.hermes/evsi_validation.json
"""

import json
import math
import sys
from collections import defaultdict


# ---- pure-python stats (no scipy) -------------------------------------------

def _ranks(xs):
    """Average-rank (handles ties) for Spearman."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    return pearson(_ranks(xs), _ranks(ys))


# ---- load + group ------------------------------------------------------------

def load_rows(path):
    with open(path) as f:
        data = json.load(f)
    return [r for r in data.get("rows", []) if "realized_change" in r and r["realized_change"] is not None]


def by_question(rows):
    """(prompt, question) -> dict with tested answers + projected question scores."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["prompt"], r["question"])].append(r)
    out = []
    for (prompt, q), rs in groups.items():
        ptot = sum(max(0.0, x["prob"]) for x in rs) or 1.0
        for x in rs:
            x["_pn"] = max(0.0, x["prob"]) / ptot  # renormalized over tested answers
        realized_change = sum(x["_pn"] * x["realized_change"] for x in rs)
        realized_evsi = sum(x["_pn"] * x["realized_change"] * x["stakes"] for x in rs)
        # realized_regret is the CLEAN, method-independent realized EVSI term (uses realized_stakes,
        # not the projected stakes that confound realized_evsi). Falls back to realized_change if a
        # row lacks it (older runs).
        realized_regret = sum(x["_pn"] * (x.get("realized_regret")
                                          if x.get("realized_regret") is not None
                                          else x["realized_change"]) for x in rs)
        # realized_stakes is where the WITHIN-TASK signal actually lives (realized_change is
        # within-task-dead — ρ≈0.04, noise). P′-weighted, method-independent; falls back to 0 if absent.
        realized_stakes = sum(x["_pn"] * (x.get("realized_stakes") or 0.0) for x in rs)
        max_delta = max(x["projected_delta"] for x in rs)
        mean_delta = sum(x["_pn"] * x["projected_delta"] for x in rs)
        mean_stakes = sum(x["_pn"] * x.get("stakes", 0.0) for x in rs)  # projected stakes (for ablation)
        out.append({
            "prompt": prompt, "question": q, "n_ans": len(rs),
            "lens": rs[0].get("lens") or "", "family": rs[0].get("family") or "",
            "q_u": rs[0]["q_u"], "q_evsi": rs[0]["q_evsi"], "q_value": rs[0]["q_value"],
            "max_delta": max_delta, "mean_delta": mean_delta, "mean_stakes": mean_stakes,
            "realized_change": realized_change, "realized_evsi": realized_evsi,
            "realized_regret": realized_regret, "realized_stakes": realized_stakes,
        })
    return out


# ---- P1a ---------------------------------------------------------------------

def p1a(rows, questions):
    pd = [r["projected_delta"] for r in rows]
    rc = [r["realized_change"] for r in rows]
    print("=" * 70)
    print("P1a — CALIBRATION: does projection predict realized change?")
    print("=" * 70)
    print(f"\nper-answer (n={len(rows)}):  projected_delta vs realized_change")
    print(f"  Pearson  r = {pearson(pd, rc):+.3f}" if pearson(pd, rc) is not None else "  Pearson  n/a")
    print(f"  Spearman ρ = {spearman(pd, rc):+.3f}" if spearman(pd, rc) is not None else "  Spearman n/a")

    # binned calibration curve
    print("\n  calibration curve (mean realized | projected_delta bin):")
    bins = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]
    for lo, hi in bins:
        sel = [r["realized_change"] for r in rows if lo <= r["projected_delta"] < hi]
        bar = "█" * round((sum(sel) / len(sel)) * 20) if sel else ""
        print(f"    Δ[{lo:.1f},{hi:.1f}): n={len(sel):>2}  mean_realized={sum(sel)/len(sel):.2f} {bar}"
              if sel else f"    Δ[{lo:.1f},{hi:.1f}): n= 0")

    # saturation diagnostics (the realized judge clustering at 0/1)
    at1 = sum(1 for x in rc if x >= 0.99)
    at0 = sum(1 for x in rc if x <= 0.01)
    print(f"\n  realized saturation: {at0}/{len(rc)} at 0.0, {at1}/{len(rc)} at 1.0 "
          f"({100*(at0+at1)/len(rc):.0f}% extreme) — discrimination concern if high")

    # per-question: projected EVSI / value vs realized
    qe = [q["q_evsi"] for q in questions]
    qv = [q["q_value"] for q in questions]
    qr = [q["realized_change"] for q in questions]
    qre = [q["realized_evsi"] for q in questions]
    print(f"\nper-question (n={len(questions)}):")
    print(f"  projected EVSI  vs realized_change : Spearman ρ = {spearman(qe, qr)}")
    print(f"  projected value vs realized_change : Spearman ρ = {spearman(qv, qr)}")
    print(f"  projected EVSI  vs realized_EVSI   : Spearman ρ = {spearman(qe, qre)}")


def per_lens(questions):
    """Per-lens attribution (#25 premortem eval): question-level realized value grouped by the
    generation lens carried on validate_evsi rows. Empty lens = flat generator (or pre-lens run).
    Prints n, mean projected value, and the realized targets so a two-arm off/on comparison can
    see whether a lens's questions earn their bucket slots."""
    by = defaultdict(list)
    for q in questions:
        by[q.get("lens") or "(none)"].append(q)
    if set(by) <= {"(none)"}:
        return  # flat run — nothing to attribute
    print("\n" + "=" * 70)
    print("PER-LENS ATTRIBUTION (question-level, P'-weighted realized targets)")
    print("=" * 70)
    print(f"  {'lens':<12}{'n_q':>4}{'value':>8}{'r_change':>10}{'r_stakes':>10}{'r_regret':>10}")
    for lens, qs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"  {lens:<12}{len(qs):>4}"
              f"{sum(q['q_value'] for q in qs)/len(qs):>8.3f}"
              f"{sum(q['realized_change'] for q in qs)/len(qs):>10.3f}"
              f"{sum(q['realized_stakes'] for q in qs)/len(qs):>10.3f}"
              f"{sum(q['realized_regret'] for q in qs)/len(qs):>10.3f}")


def _pick_abs(qs, floor):
    return [q for q in qs if q["q_value"] >= floor]


def _pick_topk(qs, k):
    return sorted(qs, key=lambda q: -q["q_value"])[:k]


def _pick_rel(qs, frac, abs_floor=0.0):
    top = max(q["q_value"] for q in qs)
    keep = max(abs_floor, frac * top)
    return [q for q in qs if q["q_value"] >= keep]


# Selection policies under comparison (#23 evidence): the calibrated absolute floor (live default),
# top-K by rank (what the investigator wrapper uses — absolute thresholds mis-fire across regimes),
# and the built-but-off rank-relative keep (`rel_keep_frac`, keep >= max(floor, frac·top)).
SELECTION_POLICIES = {
    "abs>=0.30": lambda qs: _pick_abs(qs, 0.30),
    "top3-rank": lambda qs: _pick_topk(qs, 3),
    "top5-rank": lambda qs: _pick_topk(qs, 5),
    "rel>=0.6*top": lambda qs: _pick_rel(qs, 0.6),
    "max(0.30,0.6*top)": lambda qs: _pick_rel(qs, 0.6, abs_floor=0.30),
}


def selection_policies(questions, min_q=4):
    """Which SELECTION rule best captures realized value? For each prompt with enough scored
    questions, apply each policy to the projected ranking and measure the fraction of the prompt's
    total positive realized_regret the kept set captures, and at what size. Post-hoc only — the
    scoring formula is untouched; this is the evidence for/against flipping `rel_keep_frac`."""
    by_prompt = defaultdict(list)
    for q in questions:
        if q.get("realized_regret") is not None:
            by_prompt[q["prompt"]].append(q)
    by_prompt = {p: qs for p, qs in by_prompt.items() if len(qs) >= min_q}
    if not by_prompt:
        return None
    print("\n" + "=" * 70)
    print(f"SELECTION POLICIES — realized_regret capture (n={len(by_prompt)} prompts, "
          f">= {min_q} questions each)")
    print("=" * 70)
    print(f"  {'policy':<20}{'capture':>9}{'kept/prompt':>13}{'kept_regret':>13}{'drop_regret':>13}")
    out = {}
    for name, pick in SELECTION_POLICIES.items():
        caps, sizes, kept_r, drop_r = [], [], [], []
        for p, qs in by_prompt.items():
            kept = pick(qs)
            kept_ids = {id(q) for q in kept}
            total = sum(max(q["realized_regret"], 0.0) for q in qs)
            if total > 0:
                caps.append(sum(max(q["realized_regret"], 0.0) for q in kept) / total)
            sizes.append(len(kept))
            kept_r += [q["realized_regret"] for q in kept]
            drop_r += [q["realized_regret"] for q in qs if id(q) not in kept_ids]
        out[name] = {
            "capture": round(sum(caps) / len(caps), 3) if caps else None,
            "mean_kept": round(sum(sizes) / len(sizes), 2),
            "kept_regret_mean": round(sum(kept_r) / len(kept_r), 3) if kept_r else None,
            "dropped_regret_mean": round(sum(drop_r) / len(drop_r), 3) if drop_r else None,
        }
        o = out[name]
        print(f"  {name:<20}{(o['capture'] if o['capture'] is not None else 0):>9.2f}"
              f"{o['mean_kept']:>13.1f}"
              f"{(o['kept_regret_mean'] if o['kept_regret_mean'] is not None else 0):>13.3f}"
              f"{(o['dropped_regret_mean'] if o['dropped_regret_mean'] is not None else 0):>13.3f}")
    print("  (capture = share of the prompt's positive realized_regret the kept set retains; a good"
          "\n   policy keeps capture high at a small kept/prompt and leaves dropped_regret low)")
    return out


# ---- P1c ---------------------------------------------------------------------

FORMULAS = {
    "value √(U·EVSI)": lambda q: q["q_value"],
    "EVSI-only":       lambda q: q["q_evsi"],
    "U-only":          lambda q: q["q_u"],
    "max-Δ":           lambda q: q["max_delta"],
    "mean-Δ (Pwt)":    lambda q: q["mean_delta"],
    # stakes-only: does the change/EVSI half of √(U·EVSI) earn its keep WITHIN a task, or does projected
    # stakes alone rank as well? (Phase 3 structural diagnosis — the formula stays FROZEN regardless.)
    "stakes-only":     lambda q: q.get("mean_stakes", 0.0),
}


def p1c(questions, target_key="realized_change"):
    print("\n" + "=" * 70)
    print(f"P1c — FORMULA ABLATIONS: which projected formula best matches realized")
    print(f"      (target = {target_key}; mean Spearman of per-prompt rankings)")
    print("=" * 70)
    by_prompt = defaultdict(list)
    for q in questions:
        by_prompt[q["prompt"]].append(q)
    results = {}
    for name, fn in FORMULAS.items():
        rhos = []
        for prompt, qs in by_prompt.items():
            if len(qs) < 2:
                continue
            rho = spearman([fn(q) for q in qs], [q[target_key] for q in qs])
            if rho is not None:
                rhos.append(rho)
        results[name] = sum(rhos) / len(rhos) if rhos else None
    for name, mean_rho in sorted(results.items(), key=lambda kv: (kv[1] is None, -(kv[1] or -9))):
        star = "  <- best" if mean_rho is not None and mean_rho == max(
            (v for v in results.values() if v is not None), default=None) else ""
        print(f"  {name:<18} mean ρ = {mean_rho:+.3f}{star}" if mean_rho is not None
              else f"  {name:<18} mean ρ = n/a")
    return results


# The #24 gate ranks WITHIN-TASK against these targets, in priority order. PRIMARY is `realized_regret`
# — the realized EVSI analog (realized_change × realized_stakes), i.e. exactly what q_value=√(U·EVSI) is
# meant to predict, and the strongest within-task target at n=12. stakes/change are reported alongside.
# (Historical note: at n=6 realized_change looked within-task-dead (ρ≈0.04) and stakes looked primary;
# both were SMALL-SAMPLE NOISE — at n=12 all three carry within-task signal for the absolute judge.)
_GATE_TARGETS = [
    ("realized_regret", "PRIMARY — realized EVSI (change×stakes)"),
    ("realized_stakes", "secondary"),
    ("realized_change", "secondary"),
]


def within_task_rhos(questions, target_key):
    """{prompt: per-prompt Spearman of the shipped formula (value √(U·EVSI)) vs target_key}. Per-prompt
    (not just the mean) so the A/B can show whether a 'win' is broad or carried by 1-2 outliers."""
    by_prompt = defaultdict(list)
    for q in questions:
        by_prompt[q["prompt"]].append(q)
    out = {}
    for prompt, qs in by_prompt.items():
        if len(qs) < 2:
            continue
        rho = spearman([q["q_value"] for q in qs], [q[target_key] for q in qs])
        if rho is not None:
            out[prompt] = rho
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _paired(deltas):
    """mean / sd / se / wins / losses for a paired per-prompt Δρ (pairwise − absolute)."""
    n = len(deltas)
    if n == 0:
        return None
    m = sum(deltas) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in deltas) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    return {"n": n, "mean": m, "sd": sd, "se": se,
            "wins": sum(1 for d in deltas if d > 1e-6), "losses": sum(1 for d in deltas if d < -1e-6)}


def ab_within_task(rows):
    """The #24 GATE. Compares each elicitation method's WITHIN-TASK ranking (per-prompt Spearman of
    value √(U·EVSI) vs the realized target). Ranked on `realized_stakes` (where within-task signal
    lives), NOT `realized_change` (within-task-dead). Adopt pairwise ONLY if it beats absolute by
    Δρ>0.02 on the PRIMARY target AND the per-prompt paired delta is broad, not a 1-2-outlier mean."""
    by_method = defaultdict(list)
    for r in rows:
        by_method[r.get("method") or "absolute"].append(r)
    if len(by_method) < 2:
        return  # not an A/B run
    print("\n" + "=" * 70)
    print("#24 GATE — WITHIN-TASK RANKING by elicitation method (per-prompt mean ρ)")
    print("=" * 70)
    rhos = {m: {t: within_task_rhos(by_question(mr), t) for t, _ in _GATE_TARGETS}
            for m, mr in by_method.items()}
    for target, label in _GATE_TARGETS:
        means = {m: _mean(list(rhos[m][target].values())) for m in rhos}
        best = max((v for v in means.values() if v is not None), default=None)
        print(f"\n  target = {target}  ({label}):")
        for method in sorted(means):
            v = means[method]
            star = "  <- best" if v is not None and v == best else ""
            print(f"    {method:<10} mean ρ = {v:+.3f}{star}" if v is not None
                  else f"    {method:<10} mean ρ = n/a")
    if "absolute" in rhos and "pairwise" in rhos:
        print("\n  VERDICT (keyed on the PRIMARY target realized_regret; per-prompt paired Δρ):")
        for target in ("realized_regret", "realized_stakes"):
            a, p = rhos["absolute"][target], rhos["pairwise"][target]
            common = sorted(set(a) & set(p))
            st = _paired([p[c] - a[c] for c in common])
            if not st:
                continue
            # "distinguishable from zero" guard: broad (majority wins, no worse than 1 loss for a small
            # sample) AND mean beyond ~1 SE. Deliberately conservative — a 2-outlier mean must NOT pass.
            broad = st["wins"] >= st["losses"] * 2 and st["losses"] <= max(1, st["n"] // 4)
            beyond_noise = st["mean"] > st["se"]
            decisive = st["mean"] > 0.02 and broad and beyond_noise
            print(f"    {target}: Δρ mean {st['mean']:+.3f} (sd {st['sd']:.2f}, se {st['se']:.2f}), "
                  f"pairwise wins {st['wins']}/{st['n']} (losses {st['losses']}) → "
                  f"{'ADOPT pairwise' if decisive else 'keep absolute (not a clear, broad win)'}")


def main(argv):
    path = argv[1] if len(argv) > 1 else "/Users/dadleet/.hermes/evsi_validation.json"
    rows = load_rows(path)
    if not rows:
        print(f"no usable rows in {path}", file=sys.stderr)
        return 1
    # If this is an A/B run, report the per-method gate first, then fall through to the standard
    # single-method analysis on the absolute rows (the live default) for the usual diagnostics.
    methods = {r.get("method") for r in rows if r.get("method")}
    if len(methods) >= 2:
        ab_within_task(rows)
        rows = [r for r in rows if (r.get("method") or "absolute") == "absolute"]
    questions = by_question(rows)
    print(f"\nloaded {len(rows)} answer-rows / {len(questions)} questions / "
          f"{len({q['prompt'] for q in questions})} prompts from {path}\n")
    p1a(rows, questions)
    per_lens(questions)
    selection_policies(questions)
    # realized_regret (realized EVSI) is the principled WITHIN-TASK target — what q_value predicts.
    # The ablation shows whether `stakes-only`/`U-only` match `value √(U·EVSI)` within-task (Phase 3
    # structural diagnosis; formula stays FROZEN regardless).
    p1c(questions, "realized_regret")
    p1c(questions, "realized_stakes")
    p1c(questions, "realized_change")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
