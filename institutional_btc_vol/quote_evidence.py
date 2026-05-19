from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

EVIDENCE_STATUS = "SCREEN-ONLY · NOT EXECUTABLE"
REQUIRED_FIELDS = [
    "rfq_id",
    "structure",
    "as_of_cst",
    "counterparty",
    "venue",
    "instrument",
    "side",
    "notional_btc",
    "execution_confidence",
    "source_confidence",
    "status",
    "evidence_ref",
]
ALLOWED_EXECUTION_CONFIDENCE = {"screen-only", "manual-indicative", "quote-verified", "trade-verified"}
FIRM_STATUSES = {"firm", "executable", "executed"}
DEMO_SOURCE_MARKERS = {"manual-rfq-demo", "synthetic-demo-only", "internal-demo-record"}
PROMOTION_ATTESTATION_FIELDS = ["promoted_by_operator", "promotion_timestamp", "promotion_basis"]


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None, places: int = 2) -> float | None:
    if value is None:
        return None
    return round(value, places)


def _publishability(execution_confidence: str, valid: bool) -> str:
    if not valid:
        return "invalid-not-publishable"
    if execution_confidence == "trade-verified":
        return "wrapper-dependent-not-automatic"
    return "not-investor-publishable"


def _counterparty_pseudonyms(counterparties: list[str]) -> dict[str, str]:
    return {name: f"Counterparty {chr(65 + idx)}" for idx, name in enumerate(sorted(set(counterparties)))}


def validate_quote_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in out or out.get(field) in (None, ""):
            if field == "evidence_ref":
                errors.append("evidence_ref is required")
            else:
                errors.append(f"Missing required field: {field}")

    execution_confidence = str(out.get("execution_confidence", "")).lower()
    if execution_confidence not in ALLOWED_EXECUTION_CONFIDENCE:
        errors.append("Invalid execution confidence")
    out["execution_confidence"] = execution_confidence

    status = str(out.get("status", "")).lower()
    out["status"] = status
    if status in FIRM_STATUSES and execution_confidence != "trade-verified":
        errors.append("firm/executable status requires trade-verified execution_confidence")
    if execution_confidence == "quote-verified":
        for field in PROMOTION_ATTESTATION_FIELDS:
            if not out.get(field):
                errors.append(f"{field} is required for quote-verified evidence")
        if not out.get("legal_review_ref"):
            errors.append("legal_review_ref is required for quote-verified evidence")
    if execution_confidence == "trade-verified" and not out.get("execution_record_ref"):
        errors.append("execution_record_ref is required for trade-verified evidence")
    if execution_confidence == "trade-verified":
        for field in PROMOTION_ATTESTATION_FIELDS:
            if not out.get(field):
                errors.append(f"{field} is required for trade-verified evidence")
        if not out.get("legal_review_ref"):
            errors.append("legal_review_ref is required for trade-verified evidence")

    venue = str(out.get("venue", "")).lower()
    source_confidence = str(out.get("source_confidence", "")).lower()
    evidence_ref = str(out.get("evidence_ref", "")).lower()
    notes = str(out.get("notes", "")).lower()
    if execution_confidence == "quote-verified" and (
        any(marker in venue for marker in DEMO_SOURCE_MARKERS)
        or any(marker in source_confidence for marker in DEMO_SOURCE_MARKERS)
        or "demo" in evidence_ref
        or "placeholder" in notes
        or "demo" in notes
    ):
        errors.append("demo/manual placeholder records cannot be quote-verified")

    notional_btc = _as_float(out.get("notional_btc"))
    if notional_btc is None or notional_btc <= 0:
        errors.append("Notional BTC must be positive")
    else:
        out["notional_btc"] = notional_btc

    bid_iv = _as_float(out.get("bid_iv"))
    ask_iv = _as_float(out.get("ask_iv"))
    mid_iv = _as_float(out.get("mid_iv"))
    if bid_iv is not None:
        out["bid_iv"] = bid_iv
    if ask_iv is not None:
        out["ask_iv"] = ask_iv
    if mid_iv is not None:
        out["mid_iv"] = mid_iv
    if bid_iv is not None and ask_iv is not None:
        if ask_iv < bid_iv:
            errors.append("ask_iv must be >= bid_iv")
        out["spread_vol_pts"] = _round((ask_iv - bid_iv) * 100)
        if mid_iv is None:
            mid_iv = (bid_iv + ask_iv) / 2
            out["mid_iv"] = _round(mid_iv, 6)
    else:
        out["spread_vol_pts"] = None

    valid = not errors
    out["valid"] = valid
    out["errors"] = errors
    out["evidence_status"] = EVIDENCE_STATUS
    out["publishability"] = _publishability(execution_confidence, valid)
    out["is_investor_publishable"] = False
    out["investor_note"] = (
        "Trade-verified evidence still requires the sponsor's counsel-approved business/legal wrapper before external use."
        if execution_confidence == "trade-verified" and valid
        else "Indicative quote evidence is internal diligence only and not investor-publishable."
    )
    return {"valid": valid, "record": out, "errors": errors}


def build_quote_evidence_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("candidate_id") or record.get("rfq_id") or "unknown")].append(record)

    candidates = []
    for candidate_id, rows in sorted(grouped.items()):
        quote_rows = [row for row in rows if row.get("execution_confidence") == "quote-verified"]
        manual_rows = [row for row in rows if row.get("execution_confidence") == "manual-indicative"]
        trade_rows = [row for row in rows if row.get("execution_confidence") == "trade-verified"]
        quote_counterparties = sorted({str(row.get("counterparty")) for row in quote_rows if row.get("counterparty")})
        manual_counterparties = sorted({str(row.get("counterparty")) for row in manual_rows if row.get("counterparty")})
        all_counterparties = sorted({str(row.get("counterparty")) for row in rows if row.get("counterparty")})
        pseudonyms = _counterparty_pseudonyms(all_counterparties)
        mids = [_as_float(row.get("mid_iv")) for row in rows]
        mids = [mid for mid in mids if mid is not None]
        if trade_rows:
            stage = "trade-verified"
            publishability = "wrapper-dependent-not-automatic"
        elif len(quote_counterparties) >= 2:
            stage = "quote-verified"
            publishability = "not-investor-publishable"
        elif len(quote_rows) >= 1:
            stage = "one-counterparty-indicative"
            publishability = "not-investor-publishable"
        elif len(manual_counterparties) >= 2:
            stage = "two-demo-indicative-quotes"
            publishability = "not-investor-publishable"
        elif len(manual_rows) == 1:
            stage = "one-demo-indicative-quote"
            publishability = "not-investor-publishable"
        else:
            stage = "screen-only-or-invalid"
            publishability = "not-investor-publishable"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "structure": rows[0].get("structure"),
                "indicative_quote_count": len(quote_rows) + len(manual_rows),
                "manual_indicative_count": len(manual_rows),
                "quote_verified_record_count": len(quote_rows),
                "distinct_quote_counterparty_count": len(quote_counterparties),
                "distinct_manual_counterparty_count": len(manual_counterparties),
                "counterparties_required": 2,
                "trade_verified_count": len(trade_rows),
                "record_count": len(rows),
                "stage": stage,
                "publishability": publishability,
                "avg_mid_iv": _round(sum(mids) / len(mids) * 100) if mids else None,
                "counterparty_pseudonyms": [pseudonyms[name] for name in all_counterparties],
                "raw_counterparties_redacted": True,
                "evidence_status": EVIDENCE_STATUS,
            }
        )

    return {
        "evidence_status": EVIDENCE_STATUS,
        "candidate_count": len(candidates),
        "quote_verified_candidates": sum(1 for row in candidates if row["stage"] == "quote-verified"),
        "trade_verified_candidates": sum(1 for row in candidates if row["stage"] == "trade-verified"),
        "controls": [
            "Two distinct counterparties required for quote-verified candidate status",
            "same-dealer duplicate quotes cannot promote a candidate",
            "Quote evidence remains not investor-publishable until legal wrapper approval",
        ],
        "candidates": candidates,
    }


def load_quote_evidence_ledger(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    records: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    if not target.exists():
        return {
            "path": str(target),
            "records": [],
            "invalid_rows": [],
            "valid_count": 0,
            "invalid_count": 0,
            "summary": {
                "total_records": 0,
                "quote_verified_records": 0,
                "manual_indicative_records": 0,
                "trade_verified_records": 0,
                "invalid_records": 0,
                "candidate_count": 0,
                "quote_verified_candidates": 0,
                "trade_verified_candidates": 0,
                "controls": [
                    "Two distinct counterparties required for quote-verified candidate status",
                    "same-dealer duplicate quotes cannot promote a candidate",
                    "Quote evidence remains not investor-publishable until legal wrapper approval",
                ],
                "candidates": [],
            },
            "evidence_status": EVIDENCE_STATUS,
        }

    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            invalid_rows.append({"line_number": line_number, "valid": False, "errors": [f"Invalid JSON: {exc}"]})
            continue
        if not isinstance(parsed, dict):
            invalid_rows.append({"line_number": line_number, "valid": False, "errors": ["Quote evidence line must be a JSON object"]})
            continue
        result = validate_quote_record(parsed)
        row = result["record"]
        row["line_number"] = line_number
        if result["valid"]:
            records.append(row)
        else:
            invalid_rows.append(row)

    candidate_summary = build_quote_evidence_summary(records)
    summary = {
        "total_records": len(records) + len(invalid_rows),
        "quote_verified_records": sum(1 for row in records if row.get("execution_confidence") == "quote-verified"),
        "manual_indicative_records": sum(1 for row in records if row.get("execution_confidence") == "manual-indicative"),
        "trade_verified_records": sum(1 for row in records if row.get("execution_confidence") == "trade-verified"),
        "invalid_records": len(invalid_rows),
        **{k: v for k, v in candidate_summary.items() if k not in {"evidence_status"}},
    }
    return {
        "path": str(target),
        "records": records,
        "invalid_rows": invalid_rows,
        "valid_count": len(records),
        "invalid_count": len(invalid_rows),
        "summary": summary,
        "evidence_status": EVIDENCE_STATUS,
    }


# Backward-compatible alias for earlier internal tests/callers.
def load_quote_evidence(path: str | Path) -> dict[str, Any]:
    ledger = load_quote_evidence_ledger(path)
    return {
        "valid_records": ledger["records"],
        "invalid_records": ledger["invalid_rows"],
        "summary": {
            "total": ledger["summary"]["total_records"],
            "quote_verified": ledger["summary"]["quote_verified_records"],
            "manual_indicative": ledger["summary"].get("manual_indicative_records", 0),
            "trade_verified": ledger["summary"]["trade_verified_records"],
            "invalid": ledger["summary"]["invalid_records"],
        },
    }
