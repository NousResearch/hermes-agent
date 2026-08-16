"""Render a compaction-eval scorecard as a terminal table + markdown.

Usage: python evals/compaction/report.py <results_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    out_dir = Path(sys.argv[1])
    card = json.loads((out_dir / "scorecard.json").read_text(encoding="utf-8"))
    card.sort(key=lambda s: -s["recall_pct"])

    rows = []
    for s in card:
        before = s.get("before_tokens", 0)
        after = s.get("after_tokens", 0)
        kept = f"{100 * after / before:.1f}%" if before else "?"
        rows.append((
            s["policy"], f"{s['recall_pct']}%", f"{before:,}", f"{after:,}", kept,
            str(s.get("compress_seconds", "-")),
        ))

    headers = ("policy", "recall", "tokens before", "tokens after", "kept", "sec")
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))

    md = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    (out_dir / "scorecard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nmarkdown -> {out_dir}/scorecard.md")


if __name__ == "__main__":
    main()
