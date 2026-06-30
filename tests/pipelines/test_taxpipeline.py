import sqlite3

import pytest

from pipelines.taxpipeline import learning
from pipelines.taxpipeline.graph import build_tax_pipeline_graph, route_after_validation
from pipelines.taxpipeline.models import (
    ClarificationRequest,
    Evidence,
    FieldMapping,
    IssueType,
)
from pipelines.taxpipeline.nodes import tax_field_mapper_node, tax_validator_node


def _evidence(document_id: str = "/home/tobi/Dokumente/hermes-dokuments/steuer/beleg.pdf") -> Evidence:
    return Evidence(
        document_id=document_id,
        chunk_id="chunk-1",
        document_hash="abc123",
        extracted_text="Betrag 42,00 EUR",
    )


def _mapping(
    value: object,
    *,
    evidence: Evidence | None = None,
    metadata: dict[str, object] | None = None,
) -> FieldMapping:
    return FieldMapping(
        value=value,
        extraction_confidence=1.0,
        mapping_confidence=1.0,
        validation_confidence=1.0,
        evidence=evidence or _evidence(),
        metadata=metadata or {},
    )


def test_missing_schema_returns_structured_clarification() -> None:
    result = tax_field_mapper_node({"form_id": "ESt2099", "form_version": "v999"})

    assert result["requires_clarification"] is True
    request = result["clarifications"][0]
    assert request.issue_type == IssueType.MISSING_REFERENCE
    assert request.field_id == "GLOBAL"


def test_validator_rejects_non_canonical_absolute_evidence_path() -> None:
    mapping = FieldMapping(
        value="42.00",
        extraction_confidence=0.95,
        mapping_confidence=0.95,
        validation_confidence=0.95,
        evidence=_evidence("/home/tobi/Downloads/beleg.pdf"),
    )

    result = tax_validator_node({"field_mappings": {"KZ_TEST": mapping}})

    assert result["requires_clarification"] is True
    assert result["clarifications"][0].issue_type == IssueType.VALIDATION_ERROR
    assert result["validation_issues"][0].field_id == "KZ_TEST"


def test_validator_requires_clarification_for_low_confidence() -> None:
    mapping = FieldMapping(
        value="42.00",
        extraction_confidence=0.79,
        mapping_confidence=0.95,
        validation_confidence=0.95,
        evidence=_evidence(),
    )

    result = tax_validator_node({"field_mappings": {"KZ_TEST": mapping}})

    assert result["requires_clarification"] is True
    assert result["clarifications"][0].issue_type == IssueType.DATA_UNCERTAINTY
    assert mapping.field_confidence == 0.79


def test_validator_rejects_invalid_amount_relationships() -> None:
    result = tax_validator_node(
        {
            "field_mappings": {
                "net_amount": _mapping("120,00"),
                "vat_amount": _mapping("125,00"),
                "gross_amount": _mapping("100,00"),
            }
        }
    )

    issue_ids = {issue.issue_id for issue in result["validation_issues"]}
    assert "net_amount:greater_than_gross" in issue_ids
    assert "vat_amount:greater_than_gross" in issue_ids
    assert "gross_amount:net_vat_mismatch" in issue_ids
    assert result["requires_clarification"] is True


def test_validator_rejects_percent_and_future_date() -> None:
    result = tax_validator_node(
        {
            "tax_year": 2025,
            "field_mappings": {
                "private_use_percent": _mapping(
                    "125",
                    metadata={"payment_date": "2099-01-01"},
                )
            },
        }
    )

    issue_ids = {issue.issue_id for issue in result["validation_issues"]}
    assert "private_use_percent:percent_out_of_range" in issue_ids
    assert "private_use_percent:payment_date_in_future" in issue_ids
    assert "private_use_percent:payment_date_outside_tax_year" in issue_ids


def test_validator_resolves_non_conflicting_source_priority_candidates() -> None:
    invoice = _mapping(
        "42.00",
        evidence=_evidence(),
    )
    invoice.evidence.source_type = "invoice"
    user_input = _mapping("99.00")
    user_input.evidence.source_type = "user_input"

    result = tax_validator_node({"field_mapping_candidates": {"KZ_TEST": [user_input, invoice]}})

    assert result["field_mappings"]["KZ_TEST"].value == "42.00"
    assert result["requires_clarification"] is False


def test_validator_blocks_same_priority_conflicts() -> None:
    first = _mapping("42.00")
    first.evidence.source_type = "invoice"
    second = _mapping("99.00", evidence=_evidence("/home/tobi/Dokumente/hermes-dokuments/steuer/beleg2.pdf"))
    second.evidence.source_type = "receipt"

    result = tax_validator_node({"field_mapping_candidates": {"KZ_TEST": [first, second]}})

    assert result["requires_clarification"] is True
    assert result["validation_issues"][0].issue_id == "KZ_TEST:same_priority_conflict"


def test_validator_detects_duplicate_extracted_transactions() -> None:
    result = tax_validator_node(
        {
            "raw_extracted_data": {
                "extracted": [
                    {
                        "document_id": "/home/tobi/Dokumente/hermes-dokuments/steuer/rechnung.pdf",
                        "amount": "42.00",
                        "payment_date": "2025-01-10",
                        "counterparty": "ACME GmbH",
                    },
                    {
                        "document_id": "/home/tobi/Dokumente/hermes-dokuments/steuer/konto.pdf",
                        "amount": "42.00",
                        "payment_date": "2025-01-10",
                        "counterparty": "ACME GmbH",
                    },
                ]
            }
        }
    )

    assert result["requires_clarification"] is True
    assert result["validation_issues"][0].issue_id.startswith("dedupe:")


def test_route_after_validation() -> None:
    assert route_after_validation({"requires_clarification": True}) == "human_review_node"
    assert route_after_validation({"requires_clarification": False}) == "tax_optimizer_node"


def test_learning_loop_writes_only_local_few_shot_db(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "few_shots.db"
    monkeypatch.setattr(learning, "LOCAL_FEW_SHOT_DB", db_path)
    request = ClarificationRequest(
        field_id="KZ_TEST",
        question="Welcher Betrag ist korrekt?",
        missing_information=["resolved_value"],
        issue_type=IssueType.DATA_UNCERTAINTY,
        evidence_refs=["chunk-1"],
    )
    mapping = FieldMapping(
        value="42.00",
        extraction_confidence=1.0,
        mapping_confidence=1.0,
        validation_confidence=1.0,
        evidence=_evidence(),
    )

    learning.save_user_correction_as_few_shot("user-1", request, mapping, user_consented=True)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT user_id, issue_type, field_id, resolved_value, evidence_hash FROM tax_few_shots"
        ).fetchall()
    assert rows == [("user-1", "DATA_UNCERTAINTY", "KZ_TEST", '"42.00"', "abc123")]


def test_learning_loop_anonymizes_pii_before_sqlite_insert(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "few_shots.db"
    monkeypatch.setattr(learning, "LOCAL_FEW_SHOT_DB", db_path)
    request = ClarificationRequest(
        field_id="KZ_TEST",
        question="Bitte IBAN DE89 3704 0044 0532 0130 00 pruefen.",
        missing_information=["resolved_value"],
        issue_type=IssueType.DATA_UNCERTAINTY,
        evidence_refs=["chunk-1"],
    )
    mapping = FieldMapping(
        value={"email": "tobias@example.org", "amount": "42.00"},
        extraction_confidence=1.0,
        mapping_confidence=1.0,
        validation_confidence=1.0,
        evidence=Evidence(
            document_id="/home/tobi/Dokumente/hermes-dokuments/steuer/beleg.pdf",
            chunk_id="chunk-1",
            document_hash="abc123",
            extracted_text="Steuer-ID 12 345 678 901 und Betrag 42,00 EUR",
        ),
    )

    learning.save_user_correction_as_few_shot("user-1", request, mapping, user_consented=True)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT question, resolved_value, evidence_text FROM tax_few_shots"
        ).fetchall()
    assert rows == [
        (
            "Bitte IBAN <IBAN> pruefen.",
            '{"amount": "42.00", "email": "<EMAIL>"}',
            "<TAX_ID> und Betrag 42,00 EUR",
        )
    ]


def test_learning_loop_requires_explicit_consent(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "few_shots.db"
    monkeypatch.setattr(learning, "LOCAL_FEW_SHOT_DB", db_path)
    request = ClarificationRequest(
        field_id="KZ_TEST",
        question="Welcher Betrag ist korrekt?",
        missing_information=["resolved_value"],
        issue_type=IssueType.DATA_UNCERTAINTY,
        evidence_refs=["chunk-1"],
    )
    mapping = FieldMapping(
        value="42.00",
        extraction_confidence=1.0,
        mapping_confidence=1.0,
        validation_confidence=1.0,
        evidence=_evidence(),
    )

    learning.save_user_correction_as_few_shot("user-1", request, mapping)

    assert not db_path.exists()


def test_graph_builder_requires_langgraph_when_missing() -> None:
    pytest.importorskip("langgraph", reason="LangGraph is optional in this checkout")
    assert build_tax_pipeline_graph() is not None
