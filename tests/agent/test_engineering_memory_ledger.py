from __future__ import annotations

from pathlib import Path

import pytest

from agent.engineering_memory_ledger import EngineeringMemoryLedger, EngineeringMemoryLedgerError
from agent.engineering_memory_schema import EngineeringMemoryRecord


def _record(record_id: str = "em-1", **changes) -> EngineeringMemoryRecord:
    value = {
        "record_id": record_id,
        "schema_name": "agent_engineering_record_v1",
        "sync_owned": True,
        "classification": "diagnostic-only",
        "authority": "source-index",
        "status": "candidate",
        "title": "A finding",
        "summary": "A bounded diagnostic finding.",
        "agent": "codex",
        "repository": "repo",
        "memory_area": "runtime",
        "task_type": "diagnosis",
        "component": "runner",
        "observed_at": "2026-09-21T00:00:00Z",
        "source_label": "test",
        "source_digest": "digest",
        "evidence_refs": ["tests/test_runner.py"],
        "body": "Evidence body.",
    }
    value.update(changes)
    return EngineeringMemoryRecord.from_mapping(value)


def test_candidate_round_trips_and_transition_is_audited(tmp_path: Path) -> None:
    ledger = EngineeringMemoryLedger(tmp_path / "vault")
    ledger.write_candidate(_record())

    loaded = ledger.get("em-1")
    assert loaded is not None and loaded.title == "A finding"
    approved = ledger.transition(
        "em-1", "approved", reviewer="human", reason="Receipt checked", evidence_refs=("receipt-1",)
    )
    assert approved.status == "approved"
    assert list(ledger.iter_records(statuses={"approved"}))[0].record_id == "em-1"
    audit = (tmp_path / "vault" / ".audit.jsonl").read_text(encoding="utf-8")
    assert '"to_status": "approved"' in audit


def test_duplicate_id_and_illegal_transition_fail_without_overwrite(tmp_path: Path) -> None:
    ledger = EngineeringMemoryLedger(tmp_path / "vault")
    ledger.write_candidate(_record())
    with pytest.raises(EngineeringMemoryLedgerError, match="duplicate"):
        ledger.write_candidate(_record())
    with pytest.raises(EngineeringMemoryLedgerError, match="transition"):
        ledger.transition("em-1", "candidate", reviewer="human", reason="bad")


def test_ledger_ignores_outside_files_and_reports_malformed_records(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "bad.md").write_text("not frontmatter", encoding="utf-8")
    (tmp_path / "outside.md").write_text("---\nrecord_id: x\n---\n", encoding="utf-8")
    ledger = EngineeringMemoryLedger(root)
    assert list(ledger.iter_records()) == []
    assert ledger.diagnostics and ledger.diagnostics[0].reason_code == "invalid_record"
