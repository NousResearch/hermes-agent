from __future__ import annotations

from typing import Any


def compute_quality_score(
    *,
    deribit_rows: int,
    ibit_rows: int,
    btc_per_share: float | None,
    cme_available: bool,
    quality_warnings: list[str],
    dislocations: int,
    quote_review_candidates: int,
    freshness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = 100
    notes: list[str] = []

    if deribit_rows <= 0:
        score -= 40
        notes.append("Deribit missing")
    elif deribit_rows < 5:
        score -= 15
        notes.append("Deribit row count low")

    if ibit_rows <= 0:
        score -= 25
        notes.append("IBIT options missing")
    elif ibit_rows < 3:
        score -= 10
        notes.append("IBIT options row count low")

    if btc_per_share is None:
        score -= 30
        notes.append("BTC/share missing")

    material_warnings = [
        warning for warning in quality_warnings
        if "cme source missing" not in warning.lower()
        and "normalized rows:" not in warning.lower()
        and "cme databento normalized rows:" not in warning.lower()
        and "cme databento bbo rows:" not in warning.lower()
    ]
    score -= min(20, len(material_warnings) * 5)
    notes.extend(material_warnings[:4])

    if not cme_available:
        notes.append("CME missing is expected pre-license")

    if dislocations == 0 and deribit_rows > 0 and ibit_rows > 0:
        score -= 5
        notes.append("No dislocation candidates generated")

    if quote_review_candidates > dislocations:
        score -= 10
        notes.append("Quote-review count exceeds candidate count")

    freshness_grade = None
    if freshness:
        freshness_grade = str(freshness.get("grade") or "unknown").lower()
        stale_sources = [str(source) for source in freshness.get("stale_sources") or []]
        missing_sources = [str(source) for source in freshness.get("missing_sources") or []]
        if freshness_grade == "yellow":
            score -= 15
        elif freshness_grade == "red":
            score -= 55
        for source in stale_sources:
            notes.append(f"Freshness warning: {source} stale")
        for source in missing_sources:
            notes.append(f"Freshness missing: {source}")

    score = max(0, min(100, score))
    if score >= 80:
        grade = "green"
    elif score >= 50:
        grade = "yellow"
    else:
        grade = "red"
    if freshness_grade == "yellow" and grade == "green":
        grade = "yellow"
    if freshness_grade == "red":
        grade = "red"

    return {
        "score": score,
        "grade": grade,
        "core_sources_live": deribit_rows > 0 and ibit_rows > 0 and btc_per_share is not None,
        "cme_available": cme_available,
        "deribit_rows": deribit_rows,
        "ibit_rows": ibit_rows,
        "dislocations": dislocations,
        "quote_review_candidates": quote_review_candidates,
        "material_warning_count": len(material_warnings),
        "freshness_grade": freshness_grade,
        "notes": notes,
    }
