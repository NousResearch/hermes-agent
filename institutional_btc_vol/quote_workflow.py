from __future__ import annotations

from typing import Any

EVIDENCE_STATUS = "SCREEN-ONLY · NOT EXECUTABLE"
STAGES = [
    "screen_only",
    "reviewed",
    "rfq_package_drafted",
    "indicative_quote_1",
    "indicative_quote_2",
    "quote_verified",
    "trade_verified",
]


def _complete_quote(quote: dict[str, Any]) -> bool:
    return all(quote.get(key) not in (None, "") for key in ("counterparty", "bid_iv", "ask_iv", "evidence_ref"))


def _stage_for_candidate(candidate: dict[str, Any]) -> tuple[str, int]:
    quotes = [q for q in candidate.get("indicative_quotes") or [] if isinstance(q, dict) and _complete_quote(q)]
    quote_count = len(quotes)
    if candidate.get("trade_evidence_ref"):
        return "trade_verified", quote_count
    if quote_count >= 2:
        return "quote_verified", quote_count
    if quote_count == 1:
        return "indicative_quote_1", quote_count
    if candidate.get("rfq_package_ref"):
        return "rfq_package_drafted", quote_count
    if candidate.get("reviewed_by"):
        return "reviewed", quote_count
    return "screen_only", quote_count


def _next_action(stage: str) -> str:
    return {
        "screen_only": "Internal candidate review; draft RFQ package only after approval.",
        "reviewed": "Draft RFQ package for internal review; do not send automatically.",
        "rfq_package_drafted": "Collect first independent indicative quote with evidence reference.",
        "indicative_quote_1": "Collect second independent indicative quote before any quote-verified label.",
        "indicative_quote_2": "Confirm both evidence references and promote only after reviewer sign-off.",
        "quote_verified": "Prepare quote-verified diligence note; still not trade-verified.",
        "trade_verified": "Archive execution record and post-trade evidence; investor use still subject to wrapper approval.",
    }[stage]


def _publishability(stage: str) -> str:
    if stage == "trade_verified":
        return "wrapper-dependent"
    if stage == "quote_verified":
        return "diligence-review-only"
    return "internal-only"


def build_quote_verification_demo_board(candidates: list[dict[str, Any]], *, limit: int = 6) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates[:limit]:
        stage, quote_count = _stage_for_candidate(candidate)
        rows.append(
            {
                "rank": candidate.get("rank"),
                "candidate": candidate.get("candidate") or "Unnamed candidate",
                "priority": candidate.get("priority") or "review",
                "direction": candidate.get("direction") or "direction pending",
                "gross_iv_diff_vol_pts": candidate.get("gross_iv_diff_vol_pts"),
                "stage": stage,
                "quote_count": quote_count,
                "counterparty_quotes_required": 2,
                "next_action": _next_action(stage),
                "publishability": _publishability(stage),
                "trade_verified_requires": "execution record + post-trade evidence",
                "evidence_status": EVIDENCE_STATUS,
            }
        )
    summary = {stage: 0 for stage in STAGES}
    for row in rows:
        summary[row["stage"]] += 1
    return {
        "title": "Quote Verification Demo Board",
        "evidence_status": EVIDENCE_STATUS,
        "control": "Manual demo workflow only; no RFQ is sent and no executable quote is implied.",
        "summary": summary,
        "rows": rows,
        "lifecycle": [
            {"stage": "screen_only", "label": "Screen-only candidate", "description": "Detected by monitor from public/model-estimated data."},
            {"stage": "reviewed", "label": "Reviewed", "description": "Human review confirms candidate is worth preparing."},
            {"stage": "rfq_package_drafted", "label": "RFQ package drafted", "description": "Package exists but is not auto-sent."},
            {"stage": "indicative_quote_1", "label": "Indicative quote 1", "description": "One independent counterparty quote captured with evidence reference."},
            {"stage": "indicative_quote_2", "label": "Indicative quote 2", "description": "Second independent quote captured with evidence reference."},
            {"stage": "quote_verified", "label": "Quote-verified", "description": "Two complete indicative quotes; not trade-verified."},
            {"stage": "trade_verified", "label": "Trade-verified", "description": "Execution record and post-trade evidence archived."},
        ],
    }
