from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVIDENCE_STATUS = "SCREEN-ONLY · QUOTE EVIDENCE TEMPLATE · NOT EXECUTABLE"


def quote_evidence_template() -> dict[str, Any]:
    return {
        "rfq_id": "replace-with-rfq-id",
        "candidate_id": "replace-with-candidate-id",
        "structure": "replace-with-structure",
        "as_of_cst": "YYYY-MM-DD HH:MM:SS CDT",
        "counterparty": "replace-with-external-counterparty-name",
        "venue": "manual-external-indicative-rfq",
        "instrument": "replace-with-instrument-description",
        "side": "two-way",
        "notional_btc": 0,
        "bid_iv": None,
        "ask_iv": None,
        "mid_iv": None,
        "execution_confidence": "quote-verified",
        "source_confidence": "external-indicative-rfq",
        "status": "indicative",
        "evidence_ref": "internal://evidence/<candidate>/<counterparty>/<timestamp>",
        "promoted_by_operator": "replace-with-operator-id-after-review",
        "promotion_timestamp": "YYYY-MM-DD HH:MM:SS CDT",
        "promotion_basis": "two real external indicative counterparty quotes captured and reviewed",
        "legal_review_ref": "internal://legal/<review-ref>",
        "notes": "Indicative only; not firm; not executable; internal diligence only.",
        "evidence_status": EVIDENCE_STATUS,
    }


def write_quote_evidence_template(output_path: str | Path) -> dict[str, Any]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(quote_evidence_template(), indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "template_path": str(path), "evidence_status": EVIDENCE_STATUS}


def append_quote_record(path: str | Path, record: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return target
