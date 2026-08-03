#!/usr/bin/env python3
"""Build a structured findings.json with evidence chains (stdlib-only).

Aggregates cross_links.csv (entity_resolution output) and an optional
timing.json (timing_analysis output) into a single evidence-chain document.

Output structure:
    {
      "metadata": {...},
      "findings": [
        {
          "id": "F0001",
          "title": "...",
          "severity": "HIGH|MEDIUM|LOW",
          "confidence": "high|medium|low",
          "summary": "...",
          "evidence": [
            {"source": "cross_links.csv", "row": 12, "fields": {...}},
            ...
          ],
          "sources": ["cross_links.csv", "timing.json"]
        }
      ]
    }

Every finding traces to specific source rows. No naked claims.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}
SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _read_cross_links(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_findings(
    cross_links_path: str,
    timing_path: str | None = None,
    out_path: str = "findings.json",
    bundled_threshold: int = 3,
) -> dict:
    findings: list[dict] = []
    next_id = 1

    # 1. Match-based findings, grouped by (left_normalized, right_normalized).
    matches = _read_cross_links(cross_links_path)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for i, row in enumerate(matches):
        row["__row__"] = str(i)
        grouped[(row.get("left_normalized", ""), row.get("right_normalized", ""))].append(row)

    for (left_norm, right_norm), rows in grouped.items():
        if not left_norm or not right_norm:
            continue
        # Use the highest-confidence match for the finding's overall confidence.
        best = min(rows, key=lambda r: CONFIDENCE_ORDER.get(r.get("confidence", "low"), 2))
        finding_id = f"F{next_id:04d}"
        next_id += 1
        evidence = [
            {
                "source": "cross_links.csv",
                "row": int(r["__row__"]),
                "fields": {
                    "match_type": r.get("match_type", ""),
                    "confidence": r.get("confidence", ""),
                    "left_name": r.get("left_name", ""),
                    "right_name": r.get("right_name", ""),
                    "overlap_ratio": r.get("overlap_ratio", ""),
                    "shared_tokens": r.get("shared_tokens", ""),
                },
            }
            for r in rows
        ]
        findings.append(
            {
                "id": finding_id,
                "title": f"Entity match: {best.get('left_name', '')} ↔ {best.get('right_name', '')}",
                "severity": "MEDIUM" if best.get("confidence") == "high" else "LOW",
                "confidence": best.get("confidence", "low"),
                "summary": (
                    f"{len(rows)} cross-link record(s) tie "
                    f"'{best.get('left_name', '')}' to "
                    f"'{best.get('right_name', '')}' "
                    f"(best tier: {best.get('match_type', '')})."
                ),
                "evidence": evidence,
                "sources": ["cross_links.csv"],
            }
        )

    # 2. Bundled-donations findings (if cross_links carries donor↔candidate pattern).
    #    Heuristic: many distinct left names sharing the same right name.
    by_right: dict[str, set[str]] = defaultdict(set)
    by_right_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in matches:
        right = r.get("right_normalized", "")
        left_raw = r.get("left_name", "").strip()
        if right and left_raw:
            by_right[right].add(left_raw)
            by_right_rows[right].append(r)
    for right_norm, lefts in by_right.items():
        if len(lefts) < bundled_threshold:
            continue
        rows = by_right_rows[right_norm]
        right_raw = rows[0].get("right_name", "")
        findings.append(
            {
                "id": f"F{next_id:04d}",
                "title": f"Bundled cross-links: {len(lefts)} distinct left entities ↔ '{right_raw}'",
                "severity": "HIGH",
                "confidence": "medium",
                "summary": (
                    f"{len(lefts)} distinct left-side entities link to "
                    f"'{right_raw}'. Pattern suggests coordinated relationship "
                    f"(e.g. bundled donations, multi-vendor employer)."
                ),
                "evidence": [
                    {
                        "source": "cross_links.csv",
                        "row": int(r.get("__row__", "0")),
                        "fields": {
                            "left_name": r.get("left_name", ""),
                            "match_type": r.get("match_type", ""),
                        },
                    }
                    for r in rows
                ],
                "sources": ["cross_links.csv"],
            }
        )
        next_id += 1

    # 3. Timing-based findings.
    if timing_path and Path(timing_path).exists():
        timing = json.loads(Path(timing_path).read_text())
        for r in timing.get("results", []):
            if not r.get("significant"):
                continue
            findings.append(
                {
                    "id": f"F{next_id:04d}",
                    "title": (
                        f"Donation timing significantly clusters near awards: "
                        f"{r['donor']} ↔ {r['recipient']}"
                    ),
                    "severity": "HIGH" if r["p_value"] < 0.01 else "MEDIUM",
                    "confidence": "medium",
                    "summary": (
                        f"Mean nearest-award distance {r['observed_mean_days']} days "
                        f"(null {r['null_mean_days']} days). p={r['p_value']}, "
                        f"effect size {r['effect_size_sd']} SD. "
                        f"{r['n_donations']} donations, {r['n_award_dates']} awards."
                    ),
                    "evidence": [
                        {
                            "source": "timing.json",
                            "row": None,
                            "fields": r,
                        }
                    ],
                    "sources": ["timing.json"],
                }
            )
            next_id += 1

    # Sort: severity → confidence → id.
    findings.sort(
        key=lambda f: (
            SEVERITY_ORDER.get(f["severity"], 3),
            CONFIDENCE_ORDER.get(f["confidence"], 3),
            f["id"],
        )
    )

    payload = {
        "metadata": {
            "n_findings": len(findings),
            "cross_links_path": cross_links_path,
            "timing_path": timing_path,
            "bundled_threshold": bundled_threshold,
        },
        "findings": findings,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cross-links", required=True)
    p.add_argument("--timing", help="Optional timing.json from timing_analysis.py")
    p.add_argument("--out", default="findings.json")
    p.add_argument(
        "--bundled-threshold",
        type=int,
        default=3,
        help="Minimum distinct left entities to flag as bundled (default 3)",
    )
    a = p.parse_args()

    payload = build_findings(
        cross_links_path=a.cross_links,
        timing_path=a.timing,
        out_path=a.out,
        bundled_threshold=a.bundled_threshold,
    )
    print(f"Wrote {payload['metadata']['n_findings']} findings to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
