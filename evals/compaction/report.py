"""Render a compaction-eval scorecard as a terminal table + markdown.

Usage: python evals/compaction/report.py <results_dir>

Every policy answers the SAME question bank, so the arms are paired and the
per-question ``scores`` array the runner already writes is enough to say whether an
ordering is real. Two policies whose paired interval straddles zero are printed as
tied rather than ranked — at fifteen questions one answer is 6.7 recall points, and
a table that orders them anyway is reporting the question draw.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _stats import bootstrap_ci, fmt_ci, min_detectable_difference, paired_delta_ci, stdev  # noqa: E402

# recall_pct = 100 * sum(scores) / (2 * n), so one point of mean item score is 50 pp.
RECALL_SCALE = 50.0


def main():
    out_dir = Path(sys.argv[1])
    card = json.loads((out_dir / "scorecard.json").read_text(encoding="utf-8"))
    card.sort(key=lambda s: -s["recall_pct"])

    scored = [s for s in card if s.get("scores")]
    n_items = len(scored[0]["scores"]) if scored else 0
    comparable = all(len(s["scores"]) == n_items for s in scored)

    rows = []
    for s in card:
        before = s.get("before_tokens", 0)
        after = s.get("after_tokens", 0)
        kept = f"{100 * after / before:.1f}%" if before else "?"
        scores = s.get("scores") or []
        if scores:
            lo, hi = bootstrap_ci(scores, scale=RECALL_SCALE)
            ci = fmt_ci(lo, hi)
        else:
            ci = "[n/a]"
        rows.append((
            s["policy"], f"{s['recall_pct']}%", ci, f"{before:,}", f"{after:,}", kept,
            str(s.get("compress_seconds", "-")),
        ))

    headers = ("policy", "recall", "95% CI", "tokens before", "tokens after", "kept", "sec")
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))

    # ── which adjacent pairs the question bank can actually order ────────────────
    verdicts: list[str] = []
    if len(scored) >= 2 and comparable:
        print()
        print(f"paired over {n_items} shared questions, ranked order:")
        for hi_s, lo_s in zip(scored, scored[1:]):
            point, lo, hi = paired_delta_ci(hi_s["scores"], lo_s["scores"], scale=RECALL_SCALE)
            sep = lo > 0 or hi < 0
            verdict = "separated" if sep else "TIED — not orderable at this n"
            verdicts.append(f"{hi_s['policy']} vs {lo_s['policy']}: {verdict}")
            print(f"  {hi_s['policy']:<18} vs {lo_s['policy']:<18} "
                  f"{point:+6.1f} pp  {fmt_ci(lo, hi)}  {verdict}")

        pooled = stdev([a - b for a, b in zip(scored[0]["scores"], scored[-1]["scores"])])
        mdd = min_detectable_difference(n_items, pooled) * RECALL_SCALE
        print()
        print(f"smallest difference {n_items} questions could detect "
              f"(80% power, two-sided 0.05): {mdd:.1f} pp")
        print("a gap below that is a question draw, not a result — add questions or "
              "report the tie.")
    elif scored and not comparable:
        print()
        print("NOTE: policies were scored on question banks of different sizes; "
              "the paired comparison is skipped rather than run on mismatched arms.")

    md = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    if verdicts:
        md += ["", f"Paired over {n_items} shared questions:", ""]
        md += [f"- {v}" for v in verdicts]
    (out_dir / "scorecard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nmarkdown -> {out_dir}/scorecard.md")


if __name__ == "__main__":
    main()
