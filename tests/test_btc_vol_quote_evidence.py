import json
from pathlib import Path

from institutional_btc_vol.quote_evidence import (
    EVIDENCE_STATUS,
    build_quote_evidence_summary,
    load_quote_evidence_ledger,
    validate_quote_record,
)


def _valid_record(**overrides):
    record = {
        "rfq_id": "rfq-001",
        "candidate_id": "candidate-ibit-deribit-001",
        "structure": "IBIT call vs Deribit BTC call vol comparison",
        "as_of_cst": "2026-05-15 13:30:00 CDT",
        "counterparty": "Dealer A",
        "venue": "manual-rfq",
        "instrument": "IBIT Jun 80C / Deribit BTC Jun 80000C",
        "side": "two-way",
        "notional_btc": 25,
        "bid_iv": 0.41,
        "ask_iv": 0.45,
        "mid_iv": 0.43,
        "execution_confidence": "manual-indicative",
        "source_confidence": "manual-indicative-rfq",
        "status": "indicative",
        "evidence_ref": "internal://rfq/rfq-001/dealer-a-note",
        "notes": "Indicative only; not firm; not executable.",
    }
    record.update(overrides)
    return record


def _promotion_fields(**overrides):
    fields = {
        "promoted_by_operator": "operator-1",
        "promotion_timestamp": "2026-05-15 13:40:00 CDT",
        "promotion_basis": "two real external indicative counterparty quotes captured and reviewed",
        "legal_review_ref": "internal://legal/review-001",
    }
    fields.update(overrides)
    return fields


def test_valid_indicative_quote_record_computes_spread_and_publishability():
    result = validate_quote_record(_valid_record())

    assert result["valid"] is True
    assert result["record"]["spread_vol_pts"] == 4.0
    assert result["record"]["evidence_status"] == EVIDENCE_STATUS
    assert result["record"]["publishability"] == "not-investor-publishable"
    assert result["errors"] == []


def test_firm_or_executable_status_rejected_without_trade_verified_confidence():
    result = validate_quote_record(_valid_record(status="firm", execution_confidence="quote-verified"))

    assert result["valid"] is False
    assert "firm/executable status requires trade-verified execution_confidence" in result["errors"]


def test_missing_evidence_ref_and_crossed_market_are_rejected():
    result = validate_quote_record(_valid_record(evidence_ref="", bid_iv=0.48, ask_iv=0.44))

    assert result["valid"] is False
    assert "evidence_ref is required" in result["errors"]
    assert "ask_iv must be >= bid_iv" in result["errors"]


def test_loader_separates_valid_invalid_and_json_errors(tmp_path):
    path = tmp_path / "quote_evidence.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(_valid_record(rfq_id="rfq-ok")),
                json.dumps(_valid_record(rfq_id="rfq-bad", evidence_ref=None)),
                "{bad json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ledger = load_quote_evidence_ledger(path)

    assert ledger["valid_count"] == 1
    assert ledger["invalid_count"] == 2
    assert ledger["records"][0]["rfq_id"] == "rfq-ok"
    assert ledger["invalid_rows"][0]["rfq_id"] == "rfq-bad"
    assert ledger["invalid_rows"][1]["line_number"] == 3
    assert ledger["summary"]["quote_verified_records"] == 0
    assert ledger["summary"]["manual_indicative_records"] == 1
    assert ledger["summary"]["trade_verified_records"] == 0


def test_summary_groups_by_candidate_and_does_not_promote_demo_indicative_rows_to_quote_verified():
    records = [
        _valid_record(rfq_id="rfq-001", candidate_id="candidate-a", counterparty="Dealer A"),
        _valid_record(rfq_id="rfq-002", candidate_id="candidate-a", counterparty="Dealer B", bid_iv=0.42, ask_iv=0.46, mid_iv=0.44),
        _valid_record(rfq_id="rfq-003", candidate_id="candidate-b", counterparty="Dealer C"),
    ]

    summary = build_quote_evidence_summary(records)

    assert summary["candidate_count"] == 2
    assert summary["quote_verified_candidates"] == 0
    candidate_a = summary["candidates"][0]
    assert candidate_a["candidate_id"] == "candidate-a"
    assert candidate_a["indicative_quote_count"] == 2
    assert candidate_a["stage"] == "two-demo-indicative-quotes"
    assert candidate_a["publishability"] == "not-investor-publishable"
    assert candidate_a["avg_mid_iv"] == 43.5
    candidate_b = summary["candidates"][1]
    assert candidate_b["stage"] == "one-demo-indicative-quote"


def test_trade_verified_with_execution_record_is_wrapper_dependent_not_automatically_publishable():
    result = validate_quote_record(
        _valid_record(
            status="executed",
            execution_confidence="trade-verified",
            source_confidence="trade-blotter",
            evidence_ref="internal://blotter/trade-001",
            execution_record_ref="internal://execution-record/trade-001",
            **_promotion_fields(),
        )
    )

    assert result["valid"] is True
    assert result["record"]["publishability"] == "wrapper-dependent-not-automatic"
    assert result["record"]["investor_note"] == "Trade-verified evidence still requires the sponsor's counsel-approved business/legal wrapper before external use."


def test_trade_verified_requires_explicit_execution_record_reference():
    result = validate_quote_record(
        _valid_record(
            status="executed",
            execution_confidence="trade-verified",
            source_confidence="trade-blotter",
            evidence_ref="internal://blotter/trade-001",
        )
    )

    assert result["valid"] is False
    assert "execution_record_ref is required for trade-verified evidence" in result["errors"]


def test_quote_verified_requires_operator_and_legal_attestation():
    result = validate_quote_record(
        _valid_record(
            execution_confidence="quote-verified",
            source_confidence="dealer-indicative-rfq",
            evidence_ref="internal://rfq/real/dealer-a/quote",
        )
    )

    assert result["valid"] is False
    assert "promoted_by_operator is required for quote-verified evidence" in result["errors"]
    assert "promotion_timestamp is required for quote-verified evidence" in result["errors"]
    assert "promotion_basis is required for quote-verified evidence" in result["errors"]
    assert "legal_review_ref is required for quote-verified evidence" in result["errors"]


def test_quote_verified_stage_requires_two_distinct_counterparties():
    records = [
        validate_quote_record(
            _valid_record(
                rfq_id="rfq-quote-a",
                candidate_id="candidate-same-dealer",
                counterparty="Dealer A",
                execution_confidence="quote-verified",
                source_confidence="dealer-indicative-rfq",
                evidence_ref="internal://rfq/candidate-same-dealer/dealer-a/quote-1",
                **_promotion_fields(),
            )
        )["record"],
        validate_quote_record(
            _valid_record(
                rfq_id="rfq-quote-b",
                candidate_id="candidate-same-dealer",
                counterparty="Dealer A",
                execution_confidence="quote-verified",
                source_confidence="dealer-indicative-rfq",
                evidence_ref="internal://rfq/candidate-same-dealer/dealer-a/quote-2",
                bid_iv=0.42,
                ask_iv=0.46,
                mid_iv=0.44,
                **_promotion_fields(),
            )
        )["record"],
        validate_quote_record(
            _valid_record(
                rfq_id="rfq-quote-c",
                candidate_id="candidate-two-dealers",
                counterparty="Dealer A",
                execution_confidence="quote-verified",
                source_confidence="dealer-indicative-rfq",
                evidence_ref="internal://rfq/candidate-two-dealers/dealer-a/quote",
                **_promotion_fields(),
            )
        )["record"],
        validate_quote_record(
            _valid_record(
                rfq_id="rfq-quote-d",
                candidate_id="candidate-two-dealers",
                counterparty="Dealer B",
                execution_confidence="quote-verified",
                source_confidence="dealer-indicative-rfq",
                evidence_ref="internal://rfq/candidate-two-dealers/dealer-b/quote",
                bid_iv=0.42,
                ask_iv=0.46,
                mid_iv=0.44,
                **_promotion_fields(),
            )
        )["record"],
    ]

    summary = build_quote_evidence_summary(records)
    by_candidate = {row["candidate_id"]: row for row in summary["candidates"]}

    assert by_candidate["candidate-same-dealer"]["stage"] == "one-counterparty-indicative"
    assert by_candidate["candidate-same-dealer"]["distinct_quote_counterparty_count"] == 1
    assert by_candidate["candidate-same-dealer"]["counterparties_required"] == 2
    assert by_candidate["candidate-same-dealer"]["publishability"] == "not-investor-publishable"
    assert by_candidate["candidate-two-dealers"]["stage"] == "quote-verified"
    assert by_candidate["candidate-two-dealers"]["distinct_quote_counterparty_count"] == 2
    assert by_candidate["candidate-two-dealers"]["counterparty_pseudonyms"] == ["Counterparty A", "Counterparty B"]
    assert "counterparties" not in by_candidate["candidate-two-dealers"]
    assert by_candidate["candidate-two-dealers"]["raw_counterparties_redacted"] is True
    assert summary["quote_verified_candidates"] == 1
    assert "Two distinct counterparties required for quote-verified candidate status" in summary["controls"]
