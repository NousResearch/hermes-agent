from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3

from tests.test_autonomous_contracts import sample_contract

from autonomous_contracts import (
    LedgerError,
    compile_ledger_seed,
    export_state,
    import_worker_closeout,
    initialize_ledger,
    ledger_check,
    ready_sprints,
    record_cleanup_entry,
    resolve_gate,
    transition_sprint,
    update_cleanup_state,
    verify_contract_lock,
    write_projection_files,
    write_schema_files,
)
from autonomous_contracts.models import CleanupRecord, WorkerCloseoutEnvelope
from autonomous_contracts.cli import main as contract_cli_main


def test_initialize_ledger_blocks_ready_sprint_until_gate_resolved(tmp_path) -> None:
    db = tmp_path / "state.sqlite"
    seed = compile_ledger_seed(sample_contract(), approved_by="galt")
    initialize_ledger(db, seed, actor="galt")

    assert ready_sprints(db) == []
    state = export_state(db)
    assert state["meta"]["contract_id"] == "sample-contract"
    assert state["sprints"][0]["unresolvedBlockingGates"] == ["G.APPROVED"]

    resolve_gate(db, "G.APPROVED", actor="galt", evidence=["approval-record:test"])
    assert ready_sprints(db) == ["PRE.1"]


def test_transition_sprint_requires_dependencies_gates_and_cleanup_closed(tmp_path) -> None:
    db = tmp_path / "state.sqlite"
    seed = compile_ledger_seed(sample_contract(), approved_by="galt")
    initialize_ledger(db, seed, actor="galt")

    try:
        transition_sprint(db, "PRE.1", "in_progress", actor="pm")
    except LedgerError as exc:
        assert "unresolved blocking gates" in str(exc)
    else:  # pragma: no cover - fail loudly in pytest output
        raise AssertionError("transition should have failed")

    resolve_gate(db, "G.APPROVED", actor="galt", evidence=["approval-record:test"])
    transition_sprint(db, "PRE.1", "in_progress", actor="pm", evidence={"packet": "PRE.1.implementer.001"})
    transition_sprint(db, "PRE.1", "completed", actor="pm", evidence={"tests": "pass"})
    state = export_state(db)
    assert state["sprints"][0]["state"] == "completed"
    assert any(event["event_type"] == "sprint_state_changed" for event in state["events"])


def test_write_projection_files_and_verify_lock(tmp_path) -> None:
    db = tmp_path / "state.sqlite"
    seed = compile_ledger_seed(sample_contract(), approved_by="galt")
    initialize_ledger(db, seed, actor="galt")
    written = write_projection_files(db, tmp_path / ".contract-ledger")
    assert {p.name for p in written} == {"current-state.json", "sprint-ledger.json", "events.jsonl"}
    assert verify_contract_lock(db, sample_contract()) is True


def test_cli_compile_init_ready_packet_roundtrip(tmp_path) -> None:
    import yaml

    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(yaml.safe_dump(sample_contract(), sort_keys=False), encoding="utf-8")
    seed_path = tmp_path / "seed.json"
    seed_path_2 = tmp_path / "seed-2.json"
    db = tmp_path / "state.sqlite"
    packet_path = tmp_path / "packet.json"

    assert contract_cli_main(["validate", str(contract_path)]) == 0
    assert contract_cli_main(["compile", str(contract_path), "--output", str(seed_path), "--approved-by", "galt"]) == 0
    assert contract_cli_main(["compile", str(contract_path), "--output", str(seed_path_2), "--approved-by", "galt"]) == 0
    assert seed_path.exists()
    assert seed_path.read_text(encoding="utf-8") == seed_path_2.read_text(encoding="utf-8")
    assert contract_cli_main(["init-ledger", str(contract_path), "--db", str(db), "--approved-by", "galt"]) == 0
    assert contract_cli_main(["verify-lock", str(contract_path), "--db", str(db)]) == 0
    assert contract_cli_main(["ready", "--db", str(db)]) == 0
    assert contract_cli_main(["resolve-gate", "--db", str(db), "--gate", "G.APPROVED", "--actor", "galt", "--evidence", "approval:test"]) == 0
    assert contract_cli_main([
        "packet",
        str(contract_path),
        "PRE.1",
        "--worker-role",
        "implementer",
        "--assigned-worker",
        "codex",
        "--output",
        str(packet_path),
    ]) == 0
    assert packet_path.exists()


def test_invalid_sprint_state_is_rejected(tmp_path) -> None:
    db = tmp_path / "state.sqlite"
    seed = compile_ledger_seed(sample_contract(), approved_by="galt")
    initialize_ledger(db, seed, actor="galt")

    try:
        transition_sprint(db, "PRE.1", "bogus", actor="pm")  # type: ignore[arg-type]
    except LedgerError as exc:
        assert "invalid sprint state" in str(exc)
    else:  # pragma: no cover - fail loudly in pytest output
        raise AssertionError("invalid sprint state should have failed")


def test_cleanup_cli_can_unblock_terminal_closeout(tmp_path) -> None:
    import yaml

    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(yaml.safe_dump(sample_contract(), sort_keys=False), encoding="utf-8")
    db = tmp_path / "state.sqlite"
    assert contract_cli_main(["init-ledger", str(contract_path), "--db", str(db), "--approved-by", "galt"]) == 0
    assert contract_cli_main(["resolve-gate", "--db", str(db), "--gate", "G.APPROVED", "--actor", "galt", "--evidence", "approval:test"]) == 0
    assert contract_cli_main(["transition", "--db", str(db), "--sprint", "PRE.1", "--state", "in_progress", "--actor", "pm"]) == 0
    assert contract_cli_main([
        "record-cleanup",
        "--db", str(db),
        "--id", "cleanup-1",
        "--type", "temp_file",
        "--sprint", "PRE.1",
        "--actor", "pm",
        "--identifier", "/tmp/ac-test",
        "--owner", "pm",
        "--close-condition", "remove temp file",
    ]) == 0
    assert contract_cli_main(["transition", "--db", str(db), "--sprint", "PRE.1", "--state", "completed", "--actor", "pm"]) == 1
    assert contract_cli_main(["update-cleanup", "--db", str(db), "--id", "cleanup-1", "--state", "closed", "--actor", "pm", "--notes", "removed"]) == 0
    assert contract_cli_main(["transition", "--db", str(db), "--sprint", "PRE.1", "--state", "completed", "--actor", "pm"]) == 0


def test_cleanup_api_can_unblock_terminal_closeout(tmp_path) -> None:
    db = tmp_path / "state.sqlite"
    seed = compile_ledger_seed(sample_contract(), approved_by="galt")
    initialize_ledger(db, seed, actor="galt")
    resolve_gate(db, "G.APPROVED", actor="galt", evidence=["approval-record:test"])
    transition_sprint(db, "PRE.1", "in_progress", actor="pm")
    record_cleanup_entry(
        db,
        CleanupRecord(
            id="cleanup-api-1",
            type="temp_file",
            createdBy="pm",
            sprintId="PRE.1",
            createdAt="2026-01-01T00:00:00+00:00",
            state="active_needed",
            identifier="/tmp/ac-api-test",
            owner="pm",
            closeCondition="remove temp file",
        ),
        actor="pm",
    )
    try:
        transition_sprint(db, "PRE.1", "completed", actor="pm")
    except LedgerError as exc:
        assert "unresolved cleanup records" in str(exc)
    else:  # pragma: no cover - fail loudly in pytest output
        raise AssertionError("open cleanup should block closeout")
    update_cleanup_state(db, "cleanup-api-1", "closed", actor="pm", notes="removed")
    transition_sprint(db, "PRE.1", "completed", actor="pm")



def _packet_and_db(tmp_path):
    db = tmp_path / "state.sqlite"
    seed = compile_ledger_seed(sample_contract(), approved_by="galt")
    initialize_ledger(db, seed, actor="galt")
    packet = __import__("autonomous_contracts").generate_worker_packet(
        sample_contract(),
        "PRE.1",
        worker_role="implementer",
        assigned_worker="codex",
        session_id="session-test",
    )
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet.model_dump(mode="json"), sort_keys=True) + "\n", encoding="utf-8")
    with contextlib.closing(sqlite3.connect(db)) as conn:
        conn.execute("UPDATE sprints SET worker_packet=? WHERE sprint_id=?", (str(packet_path), "PRE.1"))
        conn.commit()
    packet_hash = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    return db, packet_path, packet_hash


def _valid_closeout(packet_hash: str, **overrides):
    payload = {
        "schemaVersion": "worker-closeout/v1",
        "sprintId": "PRE.1",
        "workerPacketSha256": packet_hash,
        "resultStatus": "completed",
        "verificationEvidence": [
            {
                "id": "evidence-command-1",
                "type": "command",
                "commandId": "PRE.1.AC1",
                "command": "python3 --version",
                "provenance": "control_plane_captured",
                "passed": True,
                "exitCode": 0,
                "evidence": {"stdout": "Python 3.11.15"},
            }
        ],
        "warnings": [],
        "artifacts": [],
        "noLiveDbMutationProof": {
            "provenance": "sqlite_derived",
            "observedAt": "2026-01-01T00:00:00+00:00",
            "evidence": {"dbPath": "temp-test-db", "liveMutationRows": 0},
        },
        "summary": "done",
    }
    payload.update(overrides)
    return payload


def test_worker_closeout_schema_exported(tmp_path) -> None:
    written = write_schema_files(tmp_path / "schemas")
    assert "worker-closeout-envelope.schema.json" in {path.name for path in written}


def test_import_worker_closeout_dry_run_and_apply_records_structured_evidence(tmp_path) -> None:
    db, _packet_path, packet_hash = _packet_and_db(tmp_path)
    resolve_gate(db, "G.APPROVED", actor="galt", evidence=["approval-record:test"])
    transition_sprint(db, "PRE.1", "in_progress", actor="pm")
    report = import_worker_closeout(db, "PRE.1", _valid_closeout(packet_hash), actor="pm", apply=False)
    assert report["ok"] is True
    state_before = export_state(db)
    assert state_before["sprints"][0]["state"] == "in_progress"

    applied = import_worker_closeout(db, "PRE.1", _valid_closeout(packet_hash), actor="pm", apply=True)
    assert applied["ok"] is True
    state = export_state(db)
    assert state["sprints"][0]["state"] == "completed"
    assert state["sprints"][0]["completedBy"] == "pm"
    assert ledger_check(db, strict=True)["ok"] is True


def test_import_worker_closeout_rejects_worker_authored_no_live_db_proof(tmp_path) -> None:
    db, _packet_path, packet_hash = _packet_and_db(tmp_path)
    payload = _valid_closeout(
        packet_hash,
        noLiveDbMutationProof={
            "provenance": "worker_authored",
            "observedAt": "2026-01-01T00:00:00+00:00",
            "evidence": {"claim": "trust me"},
        },
    )
    report = import_worker_closeout(db, "PRE.1", payload, actor="pm", apply=False)
    assert report["ok"] is False
    assert any(issue["code"] == "untrusted_no_live_db_mutation_proof" for issue in report["issues"])


def test_import_worker_closeout_rejects_benjamin_acceptance_warning_terminal_success(tmp_path) -> None:
    db, _packet_path, packet_hash = _packet_and_db(tmp_path)
    payload = _valid_closeout(
        packet_hash,
        resultStatus="completed_with_warnings",
        warnings=[
            {
                "id": "warn-benjamin",
                "warningClass": "requires_benjamin_acceptance",
                "message": "Needs Benjamin acceptance.",
                "provenance": "supervisor_authored",
            }
        ],
    )
    try:
        WorkerCloseoutEnvelope.model_validate(payload)
    except ValueError as exc:
        assert "requires_benjamin_acceptance" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Benjamin-acceptance warning should not validate as terminal success")


def test_import_worker_closeout_requires_exact_trusted_command_evidence(tmp_path) -> None:
    db, _packet_path, packet_hash = _packet_and_db(tmp_path)
    payload = _valid_closeout(packet_hash)
    payload["verificationEvidence"][0]["command"] = "python --version"
    report = import_worker_closeout(db, "PRE.1", payload, actor="pm", apply=False)
    assert report["ok"] is False
    assert any(issue["code"] == "missing_command_evidence" for issue in report["issues"])

    payload = _valid_closeout(packet_hash)
    payload["verificationEvidence"][0]["provenance"] = "worker_authored"
    report = import_worker_closeout(db, "PRE.1", payload, actor="pm", apply=False)
    assert report["ok"] is False
    assert any(issue["code"] == "untrusted_command_evidence" for issue in report["issues"])


def test_import_worker_closeout_apply_failure_is_atomic(tmp_path) -> None:
    db, _packet_path, packet_hash = _packet_and_db(tmp_path)
    resolve_gate(db, "G.APPROVED", actor="galt", evidence=["approval-record:test"])
    transition_sprint(db, "PRE.1", "in_progress", actor="pm")
    bad = _valid_closeout(packet_hash)
    bad["workerPacketSha256"] = "0" * 64
    try:
        import_worker_closeout(db, "PRE.1", bad, actor="pm", apply=True)
    except LedgerError as exc:
        assert "worker closeout validation failed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid closeout should not apply")
    state = export_state(db)
    assert state["sprints"][0]["state"] == "in_progress"
    assert state["sprints"][0]["completedBy"] is None


def test_import_closeout_cli_and_ledger_check_strict(tmp_path) -> None:
    db, _packet_path, packet_hash = _packet_and_db(tmp_path)
    resolve_gate(db, "G.APPROVED", actor="galt", evidence=["approval-record:test"])
    transition_sprint(db, "PRE.1", "in_progress", actor="pm")
    envelope_path = tmp_path / "closeout.json"
    envelope_path.write_text(json.dumps(_valid_closeout(packet_hash)), encoding="utf-8")

    assert contract_cli_main(["ledger-check", "--db", str(db), "--strict"]) == 0
    assert contract_cli_main(["import-closeout", "--db", str(db), "--sprint", "PRE.1", "--envelope", str(envelope_path), "--actor", "pm", "--apply"]) == 0
    assert contract_cli_main(["ledger-check", "--db", str(db), "--strict"]) == 0
