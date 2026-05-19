from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCREEN_ONLY_STATUS = "SCREEN-ONLY · NOT EXECUTABLE"


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _priority(abs_vol_pts: float) -> str:
    if abs_vol_pts >= 5.0:
        return "high"
    if abs_vol_pts >= 3.0:
        return "medium"
    return "low"


def _workflow(priority: str) -> str:
    if priority == "high":
        return "draft two-counterparty indicative RFQ review (internal only)"
    if priority == "medium":
        return "add to RFQ review queue"
    return "watch only"


def rank_dislocation_candidates(
    dislocations: list[dict[str, Any]],
    *,
    run_id: str,
    as_of_cst: str,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in dislocations:
        vol_diff = _as_float(row.get("gross_iv_diff_vol_pts"))
        abs_vol = abs(vol_diff)
        priority = _priority(abs_vol)
        enriched.append(
            {
                "run_id": run_id,
                "as_of_cst": as_of_cst,
                "candidate": str(row.get("candidate", "")),
                "gross_iv_diff_vol_pts": round(vol_diff, 2),
                "abs_iv_diff_vol_pts": round(abs_vol, 2),
                "direction": "IBIT rich vs Deribit" if vol_diff >= 0 else "IBIT cheap vs Deribit",
                "priority": priority,
                "source_confidence": str(row.get("confidence", "screen-only")),
                "execution_confidence": "screen-only",
                "evidence_status": SCREEN_ONLY_STATUS,
                "recommended_workflow": _workflow(priority),
                "publishability": "internal-only until quote-verified and counsel-approved",
            }
        )
    enriched.sort(key=lambda item: (-float(item["abs_iv_diff_vol_pts"]), item["candidate"]))
    for idx, row in enumerate(enriched, start=1):
        row["rank"] = idx
    return enriched


def write_candidate_ledger(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return target
