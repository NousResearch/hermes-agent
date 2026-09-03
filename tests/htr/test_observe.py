"""Tests for Task 19 — read-only HTR run observation."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from htr import contracts, events, io, paths
from htr.observe import (
    EXIT_INTEGRITY,
    EXIT_INVOCATION,
    EXIT_OK,
    FINDING_ATTEMPT_TASK_IDENTITY_MISMATCH,
    FINDING_DUPLICATE_EVENT_ID,
    FINDING_EVENT_WITHOUT_JSON_SOT,
    FINDING_JSON_WITHOUT_MATCHING_EVENT,
    FINDING_MALFORMED_AUTHORITATIVE_JSON,
    FINDING_PHASE1_CHAIN_GAP,
    FINDING_POST_CLOSURE_ACTIVITY,
    FINDING_RECORD_FINGERPRINT_MISMATCH,
    FINDING_SOURCE_CORRESPONDENCE_FAILED,
    FINDING_SOURCE_FINGERPRINT_MISMATCH,
    FINDING_TASK_RUN_IDENTITY_MISMATCH,
    ObserveInvocationError,
    build_run_snapshot,
    compute_exit_code,
)

TASK_STATUS_RUNNING = "running"

RUN_A = "run_20260719_aaa001"
RUN_B = "run_20260719_aaa002"
TASK_A = "task_20260719_aaa001"
TASK_B = "task_20260719_aaa002"
ATTEMPT_A = "att_20260719_aaa001"
EVENT_A = "evt_20260719_aaa001"
EVENT_B = "evt_20260719_aaa002"


def _load_task16_helpers():
    """Load tracked baseline helper tests (test_run_final_closure.py)."""
    helper_path = Path(__file__).with_name("test_run_final_closure.py")
    spec = importlib.util.spec_from_file_location("task16_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TASK16 = _load_task16_helpers()


def _run_root(run_id: str, base_dir: Path) -> Path:
    return contracts.run_completion_record_json_path(run_id, base_dir).parent


def _task_events_path(run_id: str, base_dir: Path) -> Path:
    return _run_root(run_id, base_dir) / "task_events.jsonl"


def _task_status_path(run_id: str, task_id: str, base_dir: Path) -> Path:
    return _run_root(run_id, base_dir) / "tasks" / task_id / "task_status.json"


def _attempt_status_path(
    run_id: str, task_id: str, attempt_id: str, base_dir: Path
) -> Path:
    return (
        _run_root(run_id, base_dir)
        / "tasks"
        / task_id
        / "attempts"
        / attempt_id
        / "attempt_status.json"
    )


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _finalize_run_with_closure(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    closure = TASK16._run_final_closure_record(
        run_id,
        chain[3],
        chain[4],
        chain[5],
        chain[6],
        chain[7],
        chain[8],
        chain[9],
        chain[10],
        chain[11],
        chain[12],
    )
    events.record_run_final_closure(tmp_path, run_id, closure, actor="human")
    return run_id, chain[1], chain[2]


def _finding_codes(snapshot: dict) -> set[str]:
    return {f["code"] for f in snapshot["integrity"]["findings"]}


def _capture_run_tree(run_root: Path) -> dict[str, str]:
    digest: dict[str, str] = {}
    if not run_root.exists():
        return digest
    for path in sorted(run_root.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(run_root))
            digest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _observe_read_only(run_id: str, base_dir: Path) -> dict:
    run_root = _run_root(run_id, base_dir)
    before = _capture_run_tree(run_root)
    snapshot = build_run_snapshot(
        run_id, base_dir=base_dir, observed_at="2026-07-19T00:00:00+00:00"
    )
    after = _capture_run_tree(run_root)
    assert before == after
    return snapshot


def test_observe_test_module_does_not_import_untracked_htr_modules():
    repo_root = Path(__file__).resolve().parents[2]
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {
        "htr.io",
        "htr.paths",
        "htr.ids",
        "htr.state",
        "htr.artifacts",
        "htr.audit",
    }
    assert forbidden.isdisjoint(imported)


def test_complete_trusted_phase1_run(tmp_path):
    run_id, _, _ = _finalize_run_with_closure(tmp_path)
    snapshot = _observe_read_only(run_id, tmp_path)
    assert snapshot["integrity"]["status"] == "pass"
    assert snapshot["phase1_chain"]["chain_complete"] is True
    assert snapshot["phase1_chain"]["terminal_reached"] is True
    assert snapshot["decision_support"]["snapshot_trustworthy"] is True
    assert snapshot["decision_support"]["integrity_fully_clean"] is True
    assert compute_exit_code(snapshot) == EXIT_OK
    for entry in snapshot["phase1_chain"]["records"]:
        assert not Path(entry["json_path"]).is_absolute()
        assert ".." not in entry["json_path"]


def test_partial_chain_reports_gap(tmp_path):
    run_id, *_ = TASK16._run_with_planned_run(tmp_path)
    contracts.run_review_record_json_path(run_id, tmp_path).unlink()
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    assert FINDING_PHASE1_CHAIN_GAP in _finding_codes(snapshot)
    assert snapshot["phase1_chain"]["chain_complete"] is False


def test_event_present_json_missing(tmp_path):
    run_id, *_ = TASK16._run_with_planned_run(tmp_path)
    contracts.run_followup_plan_record_json_path(run_id, tmp_path).unlink()
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    assert FINDING_EVENT_WITHOUT_JSON_SOT in _finding_codes(snapshot)
    assert snapshot["integrity"]["status"] == "fail"
    assert snapshot["decision_support"]["snapshot_trustworthy"] is False
    assert snapshot["decision_support"]["lifecycle_action_eligible"] is False
    assert compute_exit_code(snapshot) == EXIT_INTEGRITY


def test_json_present_required_event_missing(tmp_path):
    run_id, *_ = TASK16._run_with_planned_run(tmp_path)
    log_path = _task_events_path(run_id, tmp_path)
    kept: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event_type") != events.EVENT_TYPE_MANUAL_RUN_FOLLOWUP_PLANNED:
            kept.append(event)
    log_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
        encoding="utf-8",
    )
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    assert FINDING_JSON_WITHOUT_MATCHING_EVENT in _finding_codes(snapshot)


def test_fingerprint_mismatch_between_event_and_json(tmp_path):
    run_id, *_ = TASK16._run_with_reviewed_run(tmp_path)
    review_path = contracts.run_review_record_json_path(run_id, tmp_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["notes"] = "tampered notes"
    review_path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    assert FINDING_RECORD_FINGERPRINT_MISMATCH in _finding_codes(snapshot)


def test_source_fingerprint_mismatch(tmp_path):
    run_id, *_ = TASK16._run_with_planned_run(tmp_path)
    request = contracts.make_run_execution_request_record(
        run_id=run_id,
        source_followup_plan_fingerprint="wrong-fingerprint",
        execution_items=[TASK16._sample_execution_item()],
    )
    request_path = contracts.run_execution_request_record_json_path(run_id, tmp_path)
    _write_json(request_path, request)
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    assert FINDING_SOURCE_FINGERPRINT_MISMATCH in _finding_codes(snapshot)


def test_source_correspondence_mismatch(tmp_path):
    run_id, *_ = TASK16._run_with_execution_result(tmp_path)
    result_path = contracts.run_execution_result_record_json_path(run_id, tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["item_results"][0]["item_status"] = contracts.EXECUTION_ITEM_FAILED
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    verification = contracts.make_run_execution_verification_record(
        run_id=run_id,
        source_execution_result_fingerprint=contracts.run_execution_result_fingerprint(
            result
        ),
        verification_decision=contracts.EXECUTION_VERIFICATION_ACCEPTED,
        item_verifications=[
            TASK16._item_verification_from_result(
                result["item_results"][0],
                item_status=contracts.EXECUTION_ITEM_COMPLETED,
            )
        ],
    )
    verification_path = contracts.run_execution_verification_record_json_path(
        run_id, tmp_path
    )
    _write_json(verification_path, verification)
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    assert FINDING_SOURCE_CORRESPONDENCE_FAILED in _finding_codes(snapshot)


def test_duplicate_event_id(tmp_path):
    run_id, task_ids, *_ = TASK16._run_with_reviewed_run(tmp_path)
    del task_ids
    log_path = _task_events_path(run_id, tmp_path)
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    duplicate = dict(rows[-1])
    log_path.write_text(
        log_path.read_text(encoding="utf-8")
        + json.dumps(duplicate, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    assert FINDING_DUPLICATE_EVENT_ID in _finding_codes(snapshot)


def test_malformed_authoritative_json(tmp_path):
    run_root = tmp_path / RUN_A
    run_root.mkdir(parents=True)
    (run_root / "run_manifest.json").write_text("{not-json", encoding="utf-8")
    snapshot = build_run_snapshot(RUN_A, base_dir=tmp_path, observed_at="fixed")
    assert FINDING_MALFORMED_AUTHORITATIVE_JSON in _finding_codes(snapshot)


def test_task_run_identity_mismatch(tmp_path):
    run_id, task_ids, *_ = TASK16._run_with_reviewed_run(tmp_path)
    task_id = task_ids[0]
    status_path = _task_status_path(run_id, task_id, tmp_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["run_id"] = RUN_B
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    assert FINDING_TASK_RUN_IDENTITY_MISMATCH in _finding_codes(snapshot)


def test_attempt_task_identity_mismatch(tmp_path):
    run_id, task_ids, attempt_ids, *_ = TASK16._run_with_reviewed_run(tmp_path)
    task_id = task_ids[0]
    attempt_id = attempt_ids[0]
    status_path = _attempt_status_path(run_id, task_id, attempt_id, tmp_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["task_id"] = TASK_B
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    assert FINDING_ATTEMPT_TASK_IDENTITY_MISMATCH in _finding_codes(snapshot)


def test_final_closure_semantics_and_post_closure_advisory(tmp_path):
    run_id, task_ids, _ = _finalize_run_with_closure(tmp_path)
    task_id = task_ids[0]
    post_event = events.make_event(
        run_id=run_id,
        task_id=task_id,
        event_type=events.EVENT_TYPE_TASK_STATUS_CHANGED,
        previous_status=TASK_STATUS_RUNNING,
        new_status=TASK_STATUS_RUNNING,
        actor="human",
        event_id=EVENT_A,
        payload={},
    )
    post_event["created_at"] = "2099-01-01T00:00:00+00:00"
    io.append_jsonl(paths.task_events_path(run_id, tmp_path), post_event)
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    assert snapshot["phase1_chain"]["terminal_reached"] is True
    assert snapshot["policy_hints"]["global_hard_lock_enforced"] is False
    post_finding = next(
        f for f in snapshot["integrity"]["findings"] if f["code"] == FINDING_POST_CLOSURE_ACTIVITY
    )
    assert post_finding["severity"] == "warning"
    assert "illegal" not in post_finding["message"].lower()
    assert "hard lock" not in post_finding["message"].lower() or "no global" in post_finding["message"].lower()
    assert snapshot["decision_support"]["integrity_fully_clean"] is False
    assert snapshot["decision_support"]["human_checkpoint_recommended"] is True
    assert compute_exit_code(snapshot) == EXIT_OK
    assert compute_exit_code(snapshot, strict=True) == EXIT_INTEGRITY


def test_deterministic_semantic_snapshot(tmp_path):
    run_id, _, _ = _finalize_run_with_closure(tmp_path)
    first = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed-ts")
    second = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed-ts")
    assert first == second
    record_types = [entry["record_type"] for entry in first["phase1_chain"]["records"]]
    assert record_types == list(contracts.PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN)


def test_stable_ordering(tmp_path):
    run_id, *_ = TASK16._run_with_reviewed_run(tmp_path, task_count=2)
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    task_ids = [task["task_id"] for task in snapshot["tasks"]]
    assert task_ids == sorted(task_ids)
    for task in snapshot["tasks"]:
        attempt_ids = [attempt["attempt_id"] for attempt in task["attempts"]]
        assert attempt_ids == sorted(attempt_ids)
    finding_keys = [
        (f["severity"], f["code"], json.dumps(f["subject"], sort_keys=True))
        for f in snapshot["integrity"]["findings"]
    ]
    assert finding_keys == sorted(finding_keys)


def test_invocation_error_missing_run(tmp_path):
    with pytest.raises(ObserveInvocationError):
        build_run_snapshot(RUN_B, base_dir=tmp_path)


def test_invocation_error_path_traversal_run_id(tmp_path):
    with pytest.raises(ObserveInvocationError):
        build_run_snapshot("../escape", base_dir=tmp_path)


def test_run_workspace_symlink_outside_runs_root_rejected(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    external = tmp_path / "external_run"
    external.mkdir()
    (external / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "run_id": RUN_A,
                "status": "created",
                "created_at": "2026-07-19T00:00:00+00:00",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    link_path = runs_root / RUN_A
    link_path.symlink_to(external, target_is_directory=True)
    with pytest.raises(ObserveInvocationError, match="outside configured runs root"):
        build_run_snapshot(RUN_A, base_dir=runs_root)


def test_cli_stdout_is_single_json_document(capsys, tmp_path):
    run_id, *_ = TASK16._run_with_planned_run(tmp_path)
    contracts.run_review_record_json_path(run_id, tmp_path).unlink()
    from hermes_cli.htr import htr_command

    code = htr_command(
        SimpleNamespace(
            htr_command="observe",
            run_id=run_id,
            runs_root=str(tmp_path),
            summary=False,
            strict=False,
        )
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["run_id"] == run_id
    assert code == EXIT_INTEGRITY


def test_cli_summary_on_stderr_only(capsys, tmp_path):
    run_id, _, _ = _finalize_run_with_closure(tmp_path)
    from hermes_cli.htr import htr_command

    code = htr_command(
        SimpleNamespace(
            htr_command="observe",
            run_id=run_id,
            runs_root=str(tmp_path),
            summary=True,
            strict=False,
        )
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["run_id"] == run_id
    assert "integrity=" in captured.err
    assert "integrity=" not in captured.out
    assert code == EXIT_OK


def test_cli_read_only_on_integrity_failure(capsys, tmp_path):
    run_id, *_ = TASK16._run_with_planned_run(tmp_path)
    plan_path = contracts.run_followup_plan_record_json_path(run_id, tmp_path)
    plan_path.unlink()
    before = _capture_run_tree(_run_root(run_id, tmp_path))
    from hermes_cli.htr import htr_command

    code = htr_command(
        SimpleNamespace(
            htr_command="observe",
            run_id=run_id,
            runs_root=str(tmp_path),
            summary=False,
            strict=False,
        )
    )
    captured = capsys.readouterr()
    json.loads(captured.out)
    after = _capture_run_tree(_run_root(run_id, tmp_path))
    assert before == after
    assert code == EXIT_INTEGRITY


def test_no_mutation_on_failed_observation(tmp_path):
    run_id, *_ = TASK16._run_with_planned_run(tmp_path)
    contracts.run_followup_plan_record_json_path(run_id, tmp_path).unlink()
    _observe_read_only(run_id, tmp_path)


def test_invalid_run_id_invocation(capsys, tmp_path):
    from hermes_cli.htr import htr_command

    code = htr_command(
        SimpleNamespace(
            htr_command="observe",
            run_id="../escape",
            runs_root=str(tmp_path),
            summary=False,
            strict=False,
        )
    )
    captured = capsys.readouterr()
    assert code == EXIT_INVOCATION
    assert captured.out == ""
    assert captured.err
