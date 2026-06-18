#!/usr/bin/env python3
"""Extract high-threat items from World Monitor fusion report JSON files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

def _hermes_home() -> Path:
    import os

    raw = (os.environ.get("HERMES_HOME") or "").strip()
    if raw:
        return Path(raw)
    return Path.home() / ".hermes"


def main() -> int:
    reports_dir = _hermes_home() / "worldmonitor-osint" / "reports"
    high_items: list[dict] = []
    cii_high: list[tuple] = []

    for p in sorted(reports_dir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        topic = d.get("topic", "?")
        wm = d.get("worldmonitor") or {}
        sections = wm.get("sections") or {}
        nd = sections.get("news_digest") or {}
        cats = nd.get("categories") or {}
        if isinstance(cats, dict):
            for cat, block in cats.items():
                for it in block.get("items") or []:
                    th = it.get("threat") or {}
                    if th.get("level") == "THREAT_LEVEL_HIGH":
                        high_items.append(
                            {
                                "report": p.name,
                                "topic": topic,
                                "category": cat,
                                "threat_cat": th.get("category"),
                                "title": (it.get("title") or "")[:300],
                                "url": it.get("url") or it.get("link") or "",
                            }
                        )
        rs = sections.get("risk_scores") or {}
        for row in rs.get("ciiScores") or []:
            cs = row.get("combinedScore")
            if cs is not None and cs >= 55:
                cii_high.append(
                    (p.name, topic, row.get("region"), cs, row.get("trend"))
                )

    seen: set[str] = set()
    uniq: list[dict] = []
    for it in high_items:
        t = it["title"]
        if t in seen:
            continue
        seen.add(t)
        uniq.append(it)

    out = {
        "reports_scanned": len(list(reports_dir.glob("*.json"))),
        "unique_high_threat_headlines": uniq,
        "top_cii_regions": sorted(cii_high, key=lambda x: -x[3])[:20],
    }
    out_path = _hermes_home() / "worldmonitor-osint" / "high_threat_digest.json"
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "path": str(out_path),
                "unique_high_threat_count": len(uniq),
                "reports_scanned": out["reports_scanned"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
