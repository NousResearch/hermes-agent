"""Render an explanation-policy scorecard as a terminal table + markdown.

Usage: python evals/explanation_policy/report.py <results_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HEADERS = ("policy", "modality", "n", "comprehension", "transfer", "calib err", "chars", "sec")


def main():
    out_dir = Path(sys.argv[1])
    card = json.loads((out_dir / "scorecard.json").read_text(encoding="utf-8"))
    # Transfer first: it is the outcome the policy is supposed to move.
    card.sort(key=lambda s: (-s["transfer_pct"], -s["comprehension_pct"]))

    models = {s.get("model") for s in card if s.get("model")}
    if models:
        print(f"model: {', '.join(sorted(models))}   task: {card[0].get('task', '?')}")
        if len(models) > 1:
            print("WARNING: rows come from different models — not comparable.")
        print()

    rows = [
        (
            s["policy"],
            s["modality"],
            str(s["n"]),
            f"{s['comprehension_pct']}%",
            f"{s['transfer_pct']}%",
            f"{s['calibration_error']:.3f}",
            f"{s['explanation_chars']:,}",
            str(s["seconds"]),
        )
        for s in card
    ]

    widths = [max(len(HEADERS[i]), *(len(r[i]) for r in rows)) for i in range(len(HEADERS))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(HEADERS))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(r[i].ljust(widths[i]) for i in range(len(HEADERS))))

    control = next((s for s in card if s["policy"] == "fixed_markdown"), None)
    if control:
        print()
        for s in card:
            if s["policy"] == control["policy"]:
                continue
            d_t = s["transfer_pct"] - control["transfer_pct"]
            d_c = s["comprehension_pct"] - control["comprehension_pct"]
            print(
                f"{s['policy']} vs control: transfer {d_t:+d}pp, "
                f"comprehension {d_c:+d}pp, "
                f"{s['explanation_chars'] / max(control['explanation_chars'], 1):.2f}x length"
            )

    md = ["| " + " | ".join(HEADERS) + " |", "|" + "|".join("---" for _ in HEADERS) + "|"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    (out_dir / "scorecard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nmarkdown -> {out_dir}/scorecard.md")


if __name__ == "__main__":
    main()
