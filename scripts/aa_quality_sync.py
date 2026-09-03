"""Propose catalog quality updates from Artificial Analysis.

Authoring-time helper — NEVER called at runtime (their terms forbid
client-side keys, the fleet would burn the rate limit, and a
recommendation must not change because a third-party endpoint
hiccuped). Run it when adding a model or refreshing the ordering;
review the printed diff and edit catalog.json yourself. The script
proposes, the commit decides.

The catalog's `quality` stays OUR field: AA-informed where they cover a
model, editorially set where they don't (day-0 releases lag their evals;
some entries never appear). AA's Intelligence Index grades the
full-precision cloud model, not our Q4 build — fine for ordering, never
for display.

Usage:
    export AA_API_KEY=...   # from https://artificialanalysis.ai (free tier)
    python scripts/aa_quality_sync.py

Attribution: scores by Artificial Analysis (https://artificialanalysis.ai).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "hermes_cli" / "local_runtime" / "catalog.json"
AA_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"

# Catalog entry id -> AA slug. Hand-maintained: AA's naming rarely matches
# HF repo names, and a wrong match silently mis-ranks a model. An entry
# absent here (or mapped to None) is editorial-only and never overwritten.
AA_SLUG_BY_ENTRY = {
    "qwen3.8-27b": "qwen3-8-27b",
    "qwen3.8-flash-next": "qwen3-8-flash-next",
    "qwen3.6-35b-a3b": "qwen3-6-35b-a3b",
    "deepseek-v4-flash": "deepseek-v4-flash",
}


def fetch_aa_models(api_key: str) -> dict[str, dict]:
    req = urllib.request.Request(AA_URL, headers={"x-api-key": api_key})
    with urllib.request.urlopen(req, timeout=30) as r:
        doc = json.load(r)
    return {m["slug"]: m for m in doc.get("data", [])}


def main() -> int:
    api_key = os.environ.get("AA_API_KEY", "").strip()
    if not api_key:
        print("AA_API_KEY not set — create a free key at "
              "https://artificialanalysis.ai and export it.", file=sys.stderr)
        return 2

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    aa = fetch_aa_models(api_key)

    print(f"{'entry':24s} {'catalog q':>9s} {'AA index':>9s}  note")
    print("-" * 70)
    for model in catalog["models"]:
        entry_id = model["id"]
        current = model.get("quality", 0)
        slug = AA_SLUG_BY_ENTRY.get(entry_id)
        if not slug:
            print(f"{entry_id:24s} {current:>9d} {'—':>9s}  editorial only (no AA mapping)")
            continue
        hit = aa.get(slug)
        if hit is None:
            print(f"{entry_id:24s} {current:>9d} {'—':>9s}  not in AA data (slug {slug!r})")
            continue
        index = (hit.get("evaluations") or {}).get(
            "artificial_analysis_intelligence_index")
        if index is None:
            print(f"{entry_id:24s} {current:>9d} {'—':>9s}  AA row lacks the index")
            continue
        proposed = round(float(index))
        marker = "" if proposed == current else "  <-- proposes change"
        print(f"{entry_id:24s} {current:>9d} {proposed:>9d}{marker}")

    print("\nReview against the decision table before editing: a quality "
          "change that flips cells in tests/hermes_cli/"
          "test_local_recommendation.py is the actual decision being made.")
    print("Attribution: scores by Artificial Analysis "
          "(https://artificialanalysis.ai).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
