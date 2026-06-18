"""Extract high-threat signals from World Monitor snapshot/fusion payloads."""

from __future__ import annotations

from typing import Any


def extract_high_threats(wm: dict[str, Any]) -> dict[str, Any]:
    """Return unique HIGH headlines and elevated CII regions from a WM block."""
    sections = (wm or {}).get("sections") or {}
    high_items: list[dict[str, Any]] = []

    nd = sections.get("news_digest") or {}
    cats = nd.get("categories") or {}
    if isinstance(cats, dict):
        for cat, block in cats.items():
            if not isinstance(block, dict):
                continue
            for it in block.get("items") or []:
                th = it.get("threat") or {}
                if th.get("level") != "THREAT_LEVEL_HIGH":
                    continue
                high_items.append(
                    {
                        "category": cat,
                        "threat_category": th.get("category"),
                        "title": (it.get("title") or "").strip(),
                        "url": it.get("url") or it.get("link") or "",
                    }
                )

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for it in high_items:
        title = it.get("title") or ""
        if not title or title in seen:
            continue
        seen.add(title)
        unique.append(it)

    by_threat_cat: dict[str, list[str]] = {}
    for it in unique:
        key = str(it.get("threat_category") or "general")
        by_threat_cat.setdefault(key, []).append(it["title"])

    rs = sections.get("risk_scores") or {}
    cii_high: list[dict[str, Any]] = []
    for row in rs.get("ciiScores") or []:
        cs = row.get("combinedScore")
        if cs is not None and float(cs) >= 55:
            cii_high.append(
                {
                    "region": row.get("region"),
                    "combinedScore": cs,
                    "trend": row.get("trend"),
                }
            )
    cii_high.sort(key=lambda r: float(r.get("combinedScore") or 0), reverse=True)

    return {
        "unique_high_threat_count": len(unique),
        "high_threat_headlines": unique[:40],
        "high_threat_by_category": by_threat_cat,
        "elevated_cii_regions": cii_high[:12],
    }
