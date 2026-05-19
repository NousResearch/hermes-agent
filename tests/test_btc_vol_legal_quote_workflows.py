from __future__ import annotations

import json

from institutional_btc_vol.legal_wrapper import build_legal_wrapper_package, write_legal_wrapper_package
from institutional_btc_vol.quote_evidence import build_quote_evidence_summary, validate_quote_record
from institutional_btc_vol.quote_templates import quote_evidence_template, write_quote_evidence_template


def _quote(counterparty: str, **overrides):
    row = quote_evidence_template()
    row.update(
        {
            "rfq_id": f"rfq-{counterparty.lower().replace(' ', '-')}",
            "candidate_id": "candidate-two-counterparty-live",
            "structure": "IBIT 7D ATM vs Deribit 8D ATM",
            "as_of_cst": "2026-05-18 13:30:00 CDT",
            "counterparty": counterparty,
            "instrument": "IBIT option / Deribit BTC option comparison",
            "notional_btc": 25,
            "bid_iv": 0.40,
            "ask_iv": 0.44,
            "mid_iv": 0.42,
            "evidence_ref": f"internal://rfq/candidate-two-counterparty-live/{counterparty}/quote",
        }
    )
    row.update(overrides)
    return row


def test_quote_evidence_template_is_not_executable_and_valid_after_filling_required_fields(tmp_path):
    path = tmp_path / "quote-template.json"
    result = write_quote_evidence_template(path)
    payload = json.loads(path.read_text())

    assert result["ok"] is True
    assert payload["evidence_status"] == "SCREEN-ONLY · QUOTE EVIDENCE TEMPLATE · NOT EXECUTABLE"

    validated = validate_quote_record(_quote("Dealer A"))
    assert validated["valid"] is True
    assert validated["record"]["publishability"] == "not-investor-publishable"


def test_two_distinct_external_quote_records_promote_candidate_but_not_publishability():
    records = [validate_quote_record(_quote("Dealer A"))["record"], validate_quote_record(_quote("Dealer B", bid_iv=0.41, ask_iv=0.45, mid_iv=0.43))["record"]]
    summary = build_quote_evidence_summary(records)

    assert summary["quote_verified_candidates"] == 1
    candidate = summary["candidates"][0]
    assert candidate["stage"] == "quote-verified"
    assert candidate["distinct_quote_counterparty_count"] == 2
    assert candidate["publishability"] == "not-investor-publishable"


def test_legal_wrapper_defaults_to_blocked_until_counsel_approved(tmp_path):
    package = build_legal_wrapper_package()
    assert package["approved_by_counsel"] is False
    assert package["status"] == "draft-blocked"
    assert "external advisory/client/RFQ/fund use prohibited until wrapper approved" in package["blockers"]
    assert any(row["activity"] == "Execution / RFQ support" and row["allowed_now"] is False for row in package["boundary_matrix"])

    result = write_legal_wrapper_package(tmp_path / "legal.json", tmp_path / "legal.md")
    assert result["ok"] is True
    assert (tmp_path / "legal.json").exists()
    assert "DRAFT — NOT APPROVED FOR EXTERNAL USE" in (tmp_path / "legal.md").read_text()
