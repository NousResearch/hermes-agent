from __future__ import annotations

from pathlib import Path

import pytest

from agent.security.admission import (
    AdmissionPromoter,
    AdmissionRecord,
    AdmissionStatus,
    AdmissionStore,
    CandidateKind,
    CandidateSource,
    find_records,
    InspectionReport,
    IntegrityVerifier,
    mark_record_approved,
    PromotionDecision,
    quarantine_mcp_server,
    reject_record,
    render_report,
)


def make_record() -> AdmissionRecord:
    return AdmissionRecord(
        record_id="rec-1",
        kind=CandidateKind.MCP_SERVER,
        source=CandidateSource(
            uri="https://example.com/demo.git",
            display_name="demo",
            version="1.0.0",
            installer="git",
        ),
    )


def test_transition_rules() -> None:
    record = make_record()
    record.transition_to(AdmissionStatus.APPROVED)
    assert record.status is AdmissionStatus.APPROVED
    record.transition_to(AdmissionStatus.REVOKED)
    assert record.status is AdmissionStatus.REVOKED
    with pytest.raises(ValueError):
        record.transition_to(AdmissionStatus.APPROVED)


def test_store_roundtrip(tmp_path: Path) -> None:
    store = AdmissionStore(tmp_path)
    record = make_record()
    store.save_record(record)
    loaded = store.load_record(record.record_id)
    assert loaded.record_id == record.record_id
    assert loaded.source.display_name == "demo"
    assert loaded.status is AdmissionStatus.QUARANTINED


def test_report_render_and_write(tmp_path: Path) -> None:
    store = AdmissionStore(tmp_path)
    record = make_record()
    report = InspectionReport(
        summary="Needs approval.",
        decision=PromotionDecision.HOLD,
        reasons=["network access"],
        warnings=["downloads code"],
    )
    content = render_report(record, report)
    path = store.write_report(record.record_id, content)
    assert path.read_text(encoding="utf-8").startswith("# Admission Report: demo")
    assert "network access" in content


def test_integrity_drift_detection(tmp_path: Path) -> None:
    verifier = IntegrityVerifier()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "file.txt").write_text("alpha", encoding="utf-8")
    state = verifier.capture(artifact)
    assert verifier.verify(artifact, state) is True
    (artifact / "file.txt").write_text("beta", encoding="utf-8")
    assert verifier.verify(artifact, state) is False


def test_promoter_requires_report_and_marks_approved(tmp_path: Path) -> None:
    store = AdmissionStore(tmp_path)
    promoter = AdmissionPromoter(store)
    record = make_record()
    with pytest.raises(ValueError):
        promoter.promote_record(record)
    report_path = store.write_report(record.record_id, "ok")
    record.report_path = str(report_path)
    promoter.promote_record(record)
    assert record.status is AdmissionStatus.APPROVED


def test_revision_lineage_increments_for_repeated_quarantine(tmp_path: Path) -> None:
    first = quarantine_mcp_server(
        "demo",
        {"url": "https://example.com/mcp"},
        [("search", "Search docs")],
        hermes_home=tmp_path,
    )
    second = quarantine_mcp_server(
        "demo",
        {"url": "https://example.com/mcp?v=2"},
        [("search", "Search docs")],
        hermes_home=tmp_path,
    )

    assert first.revision == 1
    assert second.revision == 2
    assert first.lineage_id == second.lineage_id
    assert second.record_id.startswith(f"{first.lineage_id}-rev-2")
    records = find_records(CandidateKind.MCP_SERVER, name="demo", hermes_home=tmp_path)
    assert [record.revision for record in records] == [2, 1]


def test_reject_and_approve_capture_terminal_and_timestamp(tmp_path: Path) -> None:
    record = make_record()
    reject_record(record, note="user rejected candidate", hermes_home=tmp_path)
    assert record.status is AdmissionStatus.REJECTED
    assert "user rejected candidate" in record.notes
    with pytest.raises(ValueError):
        record.transition_to(AdmissionStatus.APPROVED)

    approved = make_record()
    artifact = tmp_path / "approved.json"
    artifact.write_text("{}", encoding="utf-8")
    mark_record_approved(approved, approved_path=str(artifact), integrity_path=artifact, hermes_home=tmp_path)
    assert approved.status is AdmissionStatus.APPROVED
    assert approved.approved_at is not None
