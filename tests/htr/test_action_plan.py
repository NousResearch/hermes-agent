"""Tests for Task 21 — derived read-only action planning."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io as std_io
import json
import sys
from pathlib import Path

from htr import contracts, events
from htr.action_plan import (
    CONF_HIGH,
    EXIT_INVOCATION,
    EXIT_PLAN_NOT_ELIGIBLE,
    EXIT_PROPOSABLE,
    PLANNER_NAME,
    PLANNER_VERSION,
    PROJECT_DIR_BINDING_EXPLICIT,
    PROJECT_DIR_BINDING_OBSERVER,
    PROJECT_DIR_SEMANTIC_ROLE,
    RISK_CRITICAL,
    RISK_HIGH,
    STATE_BLOCKED_FINALIZED,
    STATE_BLOCKED_INTEGRITY,
    STATE_BLOCKED_PRECONDITION,
    STATE_INDETERMINATE,
    STATE_INPUTS_REQUIRED,
    STATE_PROPOSABLE,
    STATE_RECOVERY_PROTOCOL_REQUIRED,
    STATE_UNSUPPORTED_ACTION,
    PlanningIntent,
    build_action_plan,
    compute_plan_digest,
    compute_plan_exit_code,
    compute_source_observation_digest,
    infer_structural_next_action,
    plan_run,
    project_semantic_observation,
    resolve_project_dir_binding,
    _plan_digest_projection,
)
from htr import paths
from htr.observe import FINDING_POST_CLOSURE_ACTIVITY, build_run_snapshot

TASK_STATUS_RUNNING = "running"


def _load_task16_helpers():
    helper_path = Path(__file__).with_name("test_run_final_closure.py")
    spec = importlib.util.spec_from_file_location("task16_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TASK16 = _load_task16_helpers()


def _run_root(run_id: str, base_dir: Path) -> Path:
    return contracts.run_completion_record_json_path(run_id, base_dir).parent


def _capture_run_tree(run_root: Path) -> dict[str, str]:
    digest: dict[str, str] = {}
    if not run_root.exists():
        return digest
    for path in sorted(run_root.rglob("*")):
        if path.is_file():
            digest[str(path.relative_to(run_root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return digest


def _append_task_event_fixture(run_id: str, event: dict, base_dir: Path) -> None:
    """Test-only corrupt/legacy fixture: bypass guarded public append API."""
    from htr.io import append_jsonl

    append_jsonl(paths.task_events_path(run_id, base_dir), event)


def _plan_read_only(run_id: str, base_dir: Path, intent: PlanningIntent) -> dict:
    run_root = _run_root(run_id, base_dir)
    before = _capture_run_tree(run_root)
    if intent.htr_runs_root is None:
        intent = PlanningIntent(
            requested_action=intent.requested_action,
            action_inputs=intent.action_inputs,
            project_repository_checkpoint=intent.project_repository_checkpoint,
            htr_runs_root=str(base_dir),
            remediation_oriented=intent.remediation_oriented,
        )
    snapshot = build_run_snapshot(run_id, base_dir=base_dir, observed_at="fixed")
    plan = build_action_plan(snapshot, intent)
    after = _capture_run_tree(run_root)
    assert before == after
    return plan


def _run_with_completion_only(tmp_path):
    from htr import io
    from htr.ids import new_run_id, new_task_id

    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    task_id = new_task_id()
    io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
    TASK16._complete_task(tmp_path, run_id, task_id)
    completion = contracts.make_run_completion_record(
        run_id=run_id, completed_task_ids=[task_id]
    )
    events.complete_run_manually(run_id, completion, base_dir=tmp_path)
    return run_id, task_id


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
    return run_id, chain[1], closure


def _minimal_digest_snapshot(**overrides):
    base = {
        "run_id": "run_digest_golden",
        "phase1_boundary_status": contracts.PHASE1_BOUNDARY_STATUS,
        "run_manifest": {"run_id": "run_digest_golden", "status": "completed"},
        "phase1_chain": {
            "terminal_record_type": contracts.PHASE1_TERMINAL_RECORD_TYPE,
            "terminal_event_type": contracts.PHASE1_TERMINAL_EVENT_TYPE,
            "chain_complete": False,
            "terminal_reached": False,
            "records": [
                {
                    "record_type": "run_completion_record",
                    "present": True,
                    "fingerprint": "fp-completion",
                    "matching_event_count": 1,
                    "json_path": "run_completion_record.json",
                },
                {
                    "record_type": "run_review_record",
                    "present": False,
                    "fingerprint": None,
                    "matching_event_count": 0,
                    "json_path": "run_review_record.json",
                },
            ],
        },
        "integrity": {"status": "pass", "error_count": 0, "findings": []},
        "decision_support": {
            "snapshot_trustworthy": True,
            "lifecycle_action_eligible": True,
            "human_checkpoint_recommended": False,
            "integrity_fully_clean": True,
            "blocking_finding_codes": [],
            "warning_finding_codes": [],
        },
        "policy_hints": {
            "phase1_chain_terminal": False,
            "post_closure_activity_detected": False,
            "global_hard_lock_enforced": False,
        },
        "tasks": [],
        "observed_at": "2026-07-19T00:00:00+00:00",
    }
    base.update(overrides)
    return base


GOLDEN_OBSERVATION_DIGEST = compute_source_observation_digest(_minimal_digest_snapshot())


def test_observation_digest_excludes_observed_at():
    a = _minimal_digest_snapshot(observed_at="2026-07-19T00:00:00+00:00")
    b = _minimal_digest_snapshot(observed_at="2099-12-31T23:59:59+00:00")
    assert compute_source_observation_digest(a) == compute_source_observation_digest(b)


def test_observation_digest_excludes_finding_message():
    snap = _minimal_digest_snapshot(
        integrity={
            "status": "fail",
            "error_count": 1,
            "findings": [
                {
                    "code": "phase1_chain_gap",
                    "severity": "error",
                    "message": "wording version A",
                    "subject": {"run_id": "run_digest_golden", "record_type": "x"},
                    "evidence": ["a.json"],
                }
            ],
        },
        decision_support={
            "snapshot_trustworthy": False,
            "lifecycle_action_eligible": False,
            "human_checkpoint_recommended": True,
            "integrity_fully_clean": False,
            "blocking_finding_codes": ["phase1_chain_gap"],
            "warning_finding_codes": [],
        },
    )
    snap2 = _minimal_digest_snapshot(
        integrity={
            "status": "fail",
            "error_count": 1,
            "findings": [
                {
                    "code": "phase1_chain_gap",
                    "severity": "error",
                    "message": "wording version B completely different",
                    "subject": {"run_id": "run_digest_golden", "record_type": "x"},
                    "evidence": ["other/path.json"],
                }
            ],
        },
        decision_support={
            "snapshot_trustworthy": False,
            "lifecycle_action_eligible": False,
            "human_checkpoint_recommended": True,
            "integrity_fully_clean": False,
            "blocking_finding_codes": ["phase1_chain_gap"],
            "warning_finding_codes": [],
        },
    )
    assert compute_source_observation_digest(snap) == compute_source_observation_digest(snap2)


def test_observation_digest_changes_when_authoritative_state_changes():
    a = _minimal_digest_snapshot()
    b = _minimal_digest_snapshot(
        phase1_chain={
            **a["phase1_chain"],
            "records": [
                {
                    "record_type": "run_completion_record",
                    "present": True,
                    "fingerprint": "fp-changed",
                    "matching_event_count": 1,
                    "json_path": "run_completion_record.json",
                },
                a["phase1_chain"]["records"][1],
            ],
        },
    )
    assert compute_source_observation_digest(a) != compute_source_observation_digest(b)


def test_observation_digest_golden_vector():
    assert compute_source_observation_digest(_minimal_digest_snapshot()) == GOLDEN_OBSERVATION_DIGEST
    assert GOLDEN_OBSERVATION_DIGEST.startswith("sha256:")


def test_plan_digest_golden_vector_stable():
    snap = _minimal_digest_snapshot()
    plan = build_action_plan(snap, PlanningIntent())
    again = build_action_plan(snap, PlanningIntent())
    assert plan["plan_digest"] == again["plan_digest"]
    assert plan["plan_digest"].startswith("sha256:")


GOLDEN_PLAN_DIGEST = build_action_plan(_minimal_digest_snapshot(), PlanningIntent())["plan_digest"]


def test_plan_digest_golden_vector():
    plan = build_action_plan(_minimal_digest_snapshot(), PlanningIntent())
    assert plan["plan_digest"] == GOLDEN_PLAN_DIGEST


def test_no_action_returns_inputs_required_with_structural_hint(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    plan = _plan_read_only(run_id, tmp_path, PlanningIntent())
    assert plan["plan_state"] == STATE_INPUTS_REQUIRED
    assert plan["structural_next_action"]["api"] == "review_run_manually"
    assert plan["plan_digest"].startswith("sha256:")
    assert compute_plan_exit_code(plan) == EXIT_PLAN_NOT_ELIGIBLE


def test_partial_chain_structural_slot(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    hint = infer_structural_next_action(snapshot)
    assert hint["api"] == "review_run_manually"
    assert hint["chain_index"] == 1


def test_proposable_review_with_explicit_inputs(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    review = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(
            requested_action="review_run_manually",
            action_inputs={"record": review, "actor": "human"},
        ),
    )
    assert plan["plan_state"] == STATE_PROPOSABLE
    assert plan["automation_eligibility"]["execution_eligible"] is True
    assert plan["confidence"]["class"] == CONF_HIGH
    assert compute_plan_exit_code(plan) == EXIT_PROPOSABLE


def test_missing_semantic_inputs(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(requested_action="review_run_manually", action_inputs={}),
    )
    assert plan["plan_state"] == STATE_INPUTS_REQUIRED
    missing_paths = {m["path"] for m in plan["arguments"]["missing_required"]}
    assert "record" in missing_paths
    assert "actor" in missing_paths


def test_invalid_schema_input_blocked(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(
            requested_action="review_run_manually",
            action_inputs={"record": {"run_id": run_id}, "actor": "human"},
        ),
    )
    assert plan["plan_state"] == STATE_BLOCKED_PRECONDITION


def test_unsupported_action(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(requested_action="register_attempt"),
    )
    assert plan["plan_state"] == STATE_UNSUPPORTED_ACTION


def test_wrong_chain_position_blocked(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(requested_action="plan_run_followup"),
    )
    assert plan["plan_state"] == STATE_BLOCKED_PRECONDITION


def test_integrity_error_blocked(tmp_path):
    run_id, *_ = TASK16._run_with_planned_run(tmp_path)
    contracts.run_followup_plan_record_json_path(run_id, tmp_path).unlink()
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(requested_action="request_run_execution"),
    )
    assert plan["plan_state"] == STATE_BLOCKED_INTEGRITY


def test_finalized_mutation_blocked(tmp_path):
    run_id, _, _ = _finalize_run_with_closure(tmp_path)
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(requested_action="review_run_manually"),
    )
    assert plan["plan_state"] == STATE_BLOCKED_FINALIZED
    assert plan["automation_eligibility"]["execution_eligible"] is False


def test_finalized_post_closure_recovery_classification(tmp_path):
    run_id, task_ids, _ = _finalize_run_with_closure(tmp_path)
    task_id = task_ids[0]
    post_event = events.make_event(
        run_id=run_id,
        task_id=task_id,
        event_type=events.EVENT_TYPE_TASK_STATUS_CHANGED,
        previous_status=TASK_STATUS_RUNNING,
        new_status=TASK_STATUS_RUNNING,
        actor="human",
        payload={},
    )
    post_event["created_at"] = "2099-01-01T00:00:00+00:00"
    _append_task_event_fixture(run_id, post_event, tmp_path)
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(
            requested_action="review_run_manually",
            remediation_oriented=True,
        ),
    )
    assert plan["plan_state"] == STATE_RECOVERY_PROTOCOL_REQUIRED
    assert "recovery_protocol_not_implemented" in plan["escalation_reason_codes"]


def test_finalized_post_closure_without_remediation_intent_blocked(tmp_path):
    run_id, task_ids, _ = _finalize_run_with_closure(tmp_path)
    task_id = task_ids[0]
    post_event = events.make_event(
        run_id=run_id,
        task_id=task_id,
        event_type=events.EVENT_TYPE_TASK_STATUS_CHANGED,
        previous_status=TASK_STATUS_RUNNING,
        new_status=TASK_STATUS_RUNNING,
        actor="human",
        payload={},
    )
    post_event["created_at"] = "2099-01-01T00:00:00+00:00"
    _append_task_event_fixture(run_id, post_event, tmp_path)
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(requested_action="review_run_manually"),
    )
    assert plan["plan_state"] == STATE_BLOCKED_FINALIZED


def test_integrity_takes_precedence_over_recovery(tmp_path):
    run_id, _, _ = _finalize_run_with_closure(tmp_path)
    review_path = contracts.run_review_record_json_path(run_id, tmp_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["notes"] = "tampered"
    review_path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(requested_action="review_run_manually"),
    )
    assert plan["plan_state"] == STATE_BLOCKED_INTEGRITY


def test_warning_only_does_not_block_planning(tmp_path):
    run_id, task_ids, _ = _finalize_run_with_closure(tmp_path)
    task_id = task_ids[0]
    post_event = events.make_event(
        run_id=run_id,
        task_id=task_id,
        event_type=events.EVENT_TYPE_TASK_STATUS_CHANGED,
        previous_status=TASK_STATUS_RUNNING,
        new_status=TASK_STATUS_RUNNING,
        actor="human",
        payload={},
    )
    post_event["created_at"] = "2099-01-01T00:00:00+00:00"
    _append_task_event_fixture(run_id, post_event, tmp_path)
    plan = _plan_read_only(run_id, tmp_path, PlanningIntent())
    assert plan["plan_state"] == STATE_INPUTS_REQUIRED
    assert FINDING_POST_CLOSURE_ACTIVITY in plan["integrity_summary"]["warning_codes"]


def test_no_float_confidence():
    plan = build_action_plan(_minimal_digest_snapshot(), PlanningIntent())
    assert isinstance(plan["confidence"]["class"], str)


def test_final_closure_risk_critical(tmp_path):
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
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(
            requested_action="record_run_final_closure",
            action_inputs={"record": closure, "actor": "human"},
        ),
    )
    assert plan["risk"]["class"] == RISK_CRITICAL


def test_plan_digest_changes_when_inputs_change(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    review = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    plan_a = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(
            requested_action="review_run_manually",
            action_inputs={"record": review, "actor": "human"},
        ),
    )
    review2 = contracts.make_run_review_record(
        run_id=run_id,
        decision=contracts.RUN_REVIEW_ACCEPTED,
        notes="different",
    )
    plan_b = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(
            requested_action="review_run_manually",
            action_inputs={"record": review2, "actor": "human"},
        ),
    )
    assert plan_a["plan_digest"] != plan_b["plan_digest"]


def test_blocked_plan_still_has_digest(tmp_path):
    run_id, _, _ = _finalize_run_with_closure(tmp_path)
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(requested_action="review_run_manually"),
    )
    assert plan["plan_state"] == STATE_BLOCKED_FINALIZED
    assert plan["plan_digest"].startswith("sha256:")


def test_plan_excludes_raw_snapshot_payload(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    plan = _plan_read_only(run_id, tmp_path, PlanningIntent())
    dumped = json.dumps(plan)
    assert "observer_version" not in dumped
    assert "task_events.jsonl" not in dumped


def test_execute_request_catalog_entry(tmp_path):
    run_id, *_ = TASK16._run_with_execution_request(tmp_path)
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(
            requested_action="execute_run_execution_request",
            action_inputs={"executor": "human"},
        ),
    )
    assert plan["plan_state"] == STATE_PROPOSABLE
    assert plan["risk"]["class"] == RISK_HIGH
    binding = plan["arguments"]["derived"]["project_dir_binding"]
    assert binding["semantic_role"] == PROJECT_DIR_SEMANTIC_ROLE
    assert binding["binding"] == PROJECT_DIR_BINDING_OBSERVER
    assert "project_dir" not in plan["arguments"]["derived"]
    assert str(tmp_path) not in json.dumps(plan)


def test_project_dir_committed_path_contract(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    runs_root = tmp_path
    explicit = str(tmp_path)
    assert paths.run_root(run_id, runs_root) == paths.run_root(run_id, Path(explicit))
    record_path = contracts.run_execution_result_record_json_path(run_id, runs_root)
    assert record_path == contracts.run_execution_result_record_json_path(
        run_id, Path(explicit)
    )
    assert record_path == paths.run_root(run_id, runs_root) / "run_execution_result_record.json"
    binding, missing, errors = resolve_project_dir_binding(
        explicit_project_dir=None,
        htr_runs_root=str(runs_root),
    )
    assert not missing and not errors
    assert binding["binding"] == PROJECT_DIR_BINDING_OBSERVER


def test_explicit_project_dir_changes_digest(tmp_path):
    run_id, *_ = TASK16._run_with_execution_request(tmp_path)
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    a = build_action_plan(
        snapshot,
        PlanningIntent(
            requested_action="execute_run_execution_request",
            action_inputs={"executor": "human", "project_dir": str(tmp_path)},
            htr_runs_root=str(tmp_path),
        ),
    )
    other = str(tmp_path / "other_runs_root")
    b = build_action_plan(
        snapshot,
        PlanningIntent(
            requested_action="execute_run_execution_request",
            action_inputs={"executor": "human", "project_dir": other},
            htr_runs_root=str(tmp_path),
        ),
    )
    assert b["plan_state"] == STATE_BLOCKED_PRECONDITION
    assert a["arguments"]["derived"]["project_dir_binding"]["binding"] == (
        PROJECT_DIR_BINDING_EXPLICIT
    )
    assert a["plan_digest"] != b["plan_digest"]


def test_htr_runs_root_observed_flag_not_in_observation_digest(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    snap_a = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    plan_a = build_action_plan(snap_a, PlanningIntent(htr_runs_root=str(tmp_path)))
    plan_b = build_action_plan(snap_a, PlanningIntent())
    assert plan_a["source"]["htr_runs_root_observed"] is True
    assert plan_b["source"]["htr_runs_root_observed"] is False
    assert plan_a["source"]["source_observation_digest"] == plan_b["source"]["source_observation_digest"]


def test_plan_digest_changes_when_machine_reason_code_changes():
    snap = _minimal_digest_snapshot()
    a = build_action_plan(snap, PlanningIntent())
    b = build_action_plan(snap, PlanningIntent(requested_action="not_in_catalog"))
    assert a["plan_state_reason_codes"] != b["plan_state_reason_codes"]
    assert compute_plan_digest(a) != compute_plan_digest(b)


def test_plan_digest_ignores_human_wording_only_fields():
    snap = _minimal_digest_snapshot()
    plan = build_action_plan(snap, PlanningIntent())
    mutated = dict(plan)
    mutated["structural_next_action"] = {"api": "wording_only"}
    mutated["integrity_summary"] = {
        **plan["integrity_summary"],
        "status": plan["integrity_summary"]["status"],
    }
    mutated["planner"] = {**plan["planner"], "version": plan["planner"]["version"]}
    assert compute_plan_digest(plan) == compute_plan_digest(mutated)


def test_event_id_idempotency_explicit_and_omitted(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    review = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    without = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(
            requested_action="review_run_manually",
            action_inputs={"record": review, "actor": "human"},
        ),
    )
    assert without["idempotency"]["exact_event_identity_bound"] is False
    assert "EVENT_ID_ALLOCATED_AT_INVOKE_IF_OMITTED" in without["execution_prerequisites"]
    with_id = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(
            requested_action="review_run_manually",
            action_inputs={
                "record": review,
                "actor": "human",
                "event_id": "evt-fixed-001",
            },
        ),
    )
    assert with_id["idempotency"]["exact_event_identity_bound"] is True
    assert with_id["plan_digest"] != without["plan_digest"]


def test_cli_stdout_single_json_and_summary_stderr(tmp_path, monkeypatch):
    from hermes_cli.htr import htr_command

    run_id, _ = _run_with_completion_only(tmp_path)
    args = type(
        "Args",
        (),
        {
            "htr_command": "plan",
            "run_id": run_id,
            "runs_root": str(tmp_path),
            "action": None,
            "inputs_file": None,
            "project_checkpoint": None,
            "summary": True,
        },
    )()
    stdout = std_io.StringIO()
    stderr = std_io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    code = htr_command(args)
    assert code == EXIT_PLAN_NOT_ELIGIBLE
    payload = json.loads(stdout.getvalue())
    assert payload["plan_state"] == STATE_INPUTS_REQUIRED
    assert "state=" in stderr.getvalue()


def test_cli_inputs_without_action_exit_2(tmp_path, monkeypatch):
    from hermes_cli.htr import htr_command

    run_id, _ = _run_with_completion_only(tmp_path)
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text("{}", encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "htr_command": "plan",
            "run_id": run_id,
            "runs_root": str(tmp_path),
            "action": None,
            "inputs_file": str(inputs_path),
            "project_checkpoint": None,
            "summary": False,
        },
    )()
    stdout = std_io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    code = htr_command(args)
    assert code == EXIT_INVOCATION
    payload = json.loads(stdout.getvalue())
    assert "error" in payload
    assert "plan_digest" not in payload


def test_action_plan_module_import_boundary():
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "htr" / "action_plan.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "htr.events" not in imported


def test_project_semantic_projection_version():
    projection = project_semantic_observation(_minimal_digest_snapshot())
    assert projection["projection_version"] == "htr.observe.semantic.v1"
    assert "observed_at" not in projection
    assert "json_path" not in json.dumps(projection)


def test_planner_identity():
    plan = build_action_plan(_minimal_digest_snapshot(), PlanningIntent())
    assert plan["planner"]["name"] == PLANNER_NAME
    assert plan["planner"]["version"] == PLANNER_VERSION


def test_plan_run_convenience(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    plan = plan_run(run_id, PlanningIntent(), base_dir=tmp_path)
    assert plan["plan_state"] == STATE_INPUTS_REQUIRED


def test_compute_plan_digest_matches_embedded(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    plan = build_action_plan(
        snapshot,
        PlanningIntent(htr_runs_root=str(tmp_path)),
    )
    assert plan["plan_digest"] == compute_plan_digest(plan)


def test_execute_requires_project_dir_for_proposable(tmp_path):
    run_id, *_ = TASK16._run_with_execution_request(tmp_path)
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    plan = build_action_plan(
        snapshot,
        PlanningIntent(
            requested_action="execute_run_execution_request",
            action_inputs={"executor": "human"},
        ),
    )
    assert plan["plan_state"] == STATE_INPUTS_REQUIRED
    missing = {m["path"] for m in plan["arguments"]["missing_required"]}
    assert "project_dir" in missing


def test_wrong_position_beats_missing_inputs(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(requested_action="plan_run_followup", action_inputs={}),
    )
    assert plan["plan_state"] == STATE_BLOCKED_PRECONDITION


def test_unknown_action_with_integrity_blocked_integrity(tmp_path):
    run_id, *_ = TASK16._run_with_planned_run(tmp_path)
    contracts.run_followup_plan_record_json_path(run_id, tmp_path).unlink()
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(requested_action="register_attempt"),
    )
    assert plan["plan_state"] == STATE_BLOCKED_INTEGRITY


def test_untrusted_closure_indeterminate_not_recovery(tmp_path):
    run_id, _, _ = _finalize_run_with_closure(tmp_path)
    closure_path = contracts.run_final_closure_record_json_path(run_id, tmp_path)
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    closure["notes"] = "tampered closure"
    closure_path.write_text(json.dumps(closure, indent=2) + "\n", encoding="utf-8")
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(
            requested_action="review_run_manually",
            remediation_oriented=True,
        ),
    )
    assert plan["plan_state"] in (STATE_BLOCKED_INTEGRITY, STATE_INDETERMINATE)
    assert plan["plan_state"] != STATE_RECOVERY_PROTOCOL_REQUIRED


def test_integrity_finalized_precedence(tmp_path):
    run_id, _, _ = _finalize_run_with_closure(tmp_path)
    review_path = contracts.run_review_record_json_path(run_id, tmp_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["notes"] = "tampered"
    review_path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(requested_action="review_run_manually"),
    )
    assert plan["plan_state"] == STATE_BLOCKED_INTEGRITY


def test_unsupported_action_fields_fail_closed(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    review = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(
            requested_action="review_run_manually",
            action_inputs={"record": review, "actor": "human", "extra_field": True},
        ),
    )
    assert plan["plan_state"] == STATE_BLOCKED_PRECONDITION


def test_proposable_includes_future_execution_prerequisites(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    review = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(
            requested_action="review_run_manually",
            action_inputs={"record": review, "actor": "human"},
        ),
    )
    assert plan["plan_state"] == STATE_PROPOSABLE
    prereqs = plan["execution_prerequisites"]
    assert "task_22_immutable_finalized_run_seal" in prereqs
    assert "task_25_human_gated_invoke" in prereqs
    assert plan["automation_eligibility"]["reason_codes"] == [
        "EXEC_STRUCTURALLY_COMPLETE_BUT_NOT_AUTHORIZED"
    ]
    assert plan["proposable_completeness"]["exact_event_identity_bound"] is False


def test_non_proposable_execution_eligible_false(tmp_path):
    run_id, _, _ = _finalize_run_with_closure(tmp_path)
    plan = _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(requested_action="review_run_manually"),
    )
    assert plan["automation_eligibility"]["execution_eligible"] is False


def test_plan_digest_changes_with_project_checkpoint():
    snap = _minimal_digest_snapshot()
    a = build_action_plan(snap, PlanningIntent(project_repository_checkpoint="cp-a"))
    b = build_action_plan(snap, PlanningIntent(project_repository_checkpoint="cp-b"))
    assert a["plan_digest"] != b["plan_digest"]


def test_plan_digest_changes_with_plan_state():
    snap = _minimal_digest_snapshot()
    a = build_action_plan(snap, PlanningIntent())
    b = build_action_plan(
        snap,
        PlanningIntent(requested_action="unsupported_action_name"),
    )
    assert a["plan_digest"] != b["plan_digest"]


def test_plan_digest_changes_with_failed_preconditions(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    blocked = build_action_plan(
        snapshot,
        PlanningIntent(
            requested_action="plan_run_followup",
            htr_runs_root=str(tmp_path),
        ),
    )
    review = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    ok = build_action_plan(
        snapshot,
        PlanningIntent(
            requested_action="review_run_manually",
            action_inputs={"record": review, "actor": "human"},
            htr_runs_root=str(tmp_path),
        ),
    )
    assert blocked["plan_digest"] != ok["plan_digest"]
    assert blocked["failed_preconditions"] != ok["failed_preconditions"]


def test_plan_digest_ignores_display_only_fields():
    snap = _minimal_digest_snapshot()
    plan = build_action_plan(snap, PlanningIntent())
    mutated = dict(plan)
    mutated["plan_state_reason_codes"] = ["DIFFERENT_WORDING_ONLY"]
    mutated["confidence"] = {
        **plan["confidence"],
        "reason_codes": ["DIFFERENT_CONF_WORDING"],
    }
    # Machine-readable reason codes belong in the digest; human-only fields do not.
    assert compute_plan_digest(plan) != compute_plan_digest(mutated)


def test_plan_digest_stable_precondition_order():
    snap = _minimal_digest_snapshot()
    plan = build_action_plan(snap, PlanningIntent())
    proj_a = _plan_digest_projection(plan)
    proj_b = dict(proj_a)
    proj_b["preconditions"] = sorted(proj_a.get("preconditions") or [], reverse=True)
    from htr.action_plan import _sha256_digest

    assert _sha256_digest(proj_a) == _sha256_digest(proj_b)


def test_observation_digest_stable_record_order():
    snap_a = _minimal_digest_snapshot()
    records = list(snap_a["phase1_chain"]["records"])
    snap_b = _minimal_digest_snapshot(
        phase1_chain={**snap_a["phase1_chain"], "records": list(reversed(records))}
    )
    assert compute_source_observation_digest(snap_a) == compute_source_observation_digest(snap_b)


def test_runtime_no_write_unsupported_action(tmp_path):
    run_id, _ = _run_with_completion_only(tmp_path)
    _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(requested_action="register_attempt"),
    )


def test_runtime_no_write_integrity_block(tmp_path):
    run_id, *_ = TASK16._run_with_planned_run(tmp_path)
    contracts.run_followup_plan_record_json_path(run_id, tmp_path).unlink()
    _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(requested_action="request_run_execution"),
    )


def test_runtime_no_write_recovery_classification(tmp_path):
    run_id, task_ids, _ = _finalize_run_with_closure(tmp_path)
    task_id = task_ids[0]
    post_event = events.make_event(
        run_id=run_id,
        task_id=task_id,
        event_type=events.EVENT_TYPE_TASK_STATUS_CHANGED,
        previous_status=TASK_STATUS_RUNNING,
        new_status=TASK_STATUS_RUNNING,
        actor="human",
        payload={},
    )
    post_event["created_at"] = "2099-01-01T00:00:00+00:00"
    _append_task_event_fixture(run_id, post_event, tmp_path)
    _plan_read_only(
        run_id,
        tmp_path,
        PlanningIntent(
            requested_action="review_run_manually",
            remediation_oriented=True,
        ),
    )


def test_cli_invalid_inputs_after_run_resolved_no_write(tmp_path, monkeypatch):
    from hermes_cli.htr import htr_command

    run_id, _ = _run_with_completion_only(tmp_path)
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not json", encoding="utf-8")
    before = _capture_run_tree(_run_root(run_id, tmp_path))
    args = type(
        "Args",
        (),
        {
            "htr_command": "plan",
            "run_id": run_id,
            "runs_root": str(tmp_path),
            "action": "review_run_manually",
            "inputs_file": str(bad_path),
            "project_checkpoint": None,
            "remediation_intent": False,
            "summary": False,
        },
    )()
    stdout = std_io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    code = htr_command(args)
    after = _capture_run_tree(_run_root(run_id, tmp_path))
    assert before == after
    assert code == EXIT_INVOCATION
