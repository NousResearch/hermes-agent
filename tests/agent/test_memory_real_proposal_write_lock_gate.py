from copy import deepcopy

from agent.memory_block_review_queue import create_review_queue_item
from agent.memory_blocks import create_memory_block_candidate
from agent.memory_governance_submission_packet import create_governance_submission_packet
from agent.memory_human_review_outcome_gate import create_human_review_outcome_candidate
from agent.memory_proposal_draft_builder import create_memory_proposal_draft
from agent.memory_proposal_governance_gate import create_governance_submission_candidate
from agent.memory_real_proposal_creation_plan import create_real_proposal_creation_plan
from agent.memory_real_proposal_dry_run import create_real_proposal_dry_run
from agent.memory_real_proposal_write_lock_gate import (
    MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ELIGIBLE,
    MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_KIND,
    MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED,
    MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_POLICY,
    MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ROUTING,
    create_real_proposal_write_lock_gate,
    explain_real_proposal_write_lock_gate,
    recommend_real_proposal_write_lock_action,
    summarize_real_proposal_write_lock_gates,
    validate_real_proposal_write_lock_gate,
)
from agent.memory_review_decision_gate import evaluate_review_queue_item


def _plan(block_type="procedural_rules", outcome=None, source_pattern_ids=None, source_fact_ids=None):
    block = create_memory_block_candidate(
        block_type,
        {"rules": ["Create write-lock candidates only."], "nested": {"value": "preserved"}},
        project_scope="memory-fabric",
        source_pattern_ids=source_pattern_ids if source_pattern_ids is not None else ["pattern-1"],
        source_fact_ids=source_fact_ids if source_fact_ids is not None else ["fact-1"],
        metadata={"source": "test"},
    )
    queue_item = create_review_queue_item(block, reviewer="memory-reviewer")
    decision = evaluate_review_queue_item(queue_item, reviewer="memory-reviewer")
    draft = create_memory_proposal_draft(decision, author="proposal-drafter")
    submission = create_governance_submission_candidate(draft, reviewer="governance-reviewer")
    packet = create_governance_submission_packet(submission, reviewer="packet-reviewer")
    outcome_candidate = create_human_review_outcome_candidate(packet, reviewer="human-reviewer", outcome=outcome)
    return create_real_proposal_creation_plan(outcome_candidate, planner="plan-reviewer")


def _dry_run(block_type="procedural_rules", outcome=None, source_pattern_ids=None, source_fact_ids=None):
    return create_real_proposal_dry_run(
        _plan(
            block_type=block_type,
            outcome=outcome,
            source_pattern_ids=source_pattern_ids,
            source_fact_ids=source_fact_ids,
        ),
        operator="final-preflight-operator",
    )


def test_valid_dry_run_creates_eligible_human_approval_token_gate():
    dry_run = _dry_run()

    gate = create_real_proposal_write_lock_gate(dry_run, operator="write-lock-operator")

    assert gate["gate_kind"] == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_KIND
    assert gate["gate_status"] == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ELIGIBLE
    assert gate["routing"] == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_ROUTING
    assert gate["lock_reason"] is None
    assert gate["source_dry_run_id"] == dry_run["dry_run_id"]
    assert gate["source_plan_id"] == dry_run["source_plan_id"]
    assert gate["source_outcome_id"] == dry_run["source_outcome_id"]
    assert gate["source_packet_id"] == dry_run["source_packet_id"]
    assert gate["source_submission_id"] == dry_run["source_submission_id"]
    assert gate["source_draft_id"] == dry_run["source_draft_id"]
    assert gate["source_decision_id"] == dry_run["source_decision_id"]
    assert gate["source_queue_item_id"] == dry_run["source_queue_item_id"]
    assert gate["operator"] == "write-lock-operator"
    assert gate["dry_run_validation"] == {"valid": True, "errors": []}
    assert gate["gate_validation"] == {"valid": True, "errors": []}
    assert validate_real_proposal_write_lock_gate(gate) == {"valid": True, "errors": []}
    assert gate["next_step_recommendation"]["action"] == "request_separate_human_approval_token_before_real_proposal_write"
    assert gate["next_step_recommendation"]["creates_real_proposals"] is False
    assert gate["next_step_recommendation"]["writes_operation_ledger"] is False


def test_invalid_dry_run_creates_locked_gate():
    dry_run = _dry_run(outcome="reject")

    gate = create_real_proposal_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "invalid_real_proposal_dry_run"
    assert gate["dry_run_validation"]["valid"] is False
    assert gate["gate_validation"] == {"valid": True, "errors": []}
    assert gate["next_step_recommendation"]["action"] == "keep_real_proposal_write_locked"


def test_missing_proposal_record_preview_locks_gate():
    dry_run = _dry_run()
    dry_run.pop("proposal_record_preview")

    gate = create_real_proposal_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "missing_proposal_record_preview"


def test_missing_operation_ledger_preview_locks_gate():
    dry_run = _dry_run()
    dry_run.pop("operation_ledger_preview")

    gate = create_real_proposal_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "missing_operation_ledger_preview"


def test_missing_target_paths_preview_locks_gate():
    dry_run = _dry_run()
    dry_run.pop("target_paths_preview")

    gate = create_real_proposal_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "missing_target_paths_preview"


def test_missing_source_evidence_locks_gate():
    dry_run = _dry_run(source_pattern_ids=[], source_fact_ids=[])

    gate = create_real_proposal_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "missing_source_evidence"


def test_preview_integrity_failed_when_preview_only_false_or_written_created_flags_true():
    cases = []
    for field in ("proposal_record_preview", "operation_ledger_preview", "target_paths_preview"):
        preview_only_false = _dry_run()
        preview_only_false[field]["preview_only"] = False
        cases.append(preview_only_false)

        written_true = _dry_run()
        written_true[field]["written"] = True
        cases.append(written_true)

        created_true = _dry_run()
        created_true[field]["created_real_proposal"] = True
        cases.append(created_true)

    for dry_run in cases:
        gate = create_real_proposal_write_lock_gate(dry_run)
        assert gate["gate_status"] == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED
        assert gate["lock_reason"] == "preview_integrity_failed"


def test_write_lock_checklist_is_deterministic():
    gate_a = create_real_proposal_write_lock_gate(_dry_run())
    gate_b = create_real_proposal_write_lock_gate(_dry_run())

    assert gate_a["write_lock_checklist"] == gate_b["write_lock_checklist"]
    assert [check["id"] for check in gate_a["write_lock_checklist"]] == [
        "verify_dry_run_validation",
        "verify_preview_artifacts_only",
        "verify_no_written_or_created_flags",
        "verify_payload_and_source_evidence",
        "require_separate_human_approval_token",
    ]


def test_input_dry_run_is_not_mutated():
    dry_run = _dry_run()
    before = deepcopy(dry_run)

    gate = create_real_proposal_write_lock_gate(dry_run)
    gate["source_dry_run_snapshot"]["payload_preview"]["content"]["nested"]["value"] = "changed"

    assert dry_run == before


def test_policy_proves_no_memory_config_graph_proposal_or_ledger_writes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    gate = create_real_proposal_write_lock_gate(_dry_run())
    explanation = explain_real_proposal_write_lock_gate(gate)
    recommendation = recommend_real_proposal_write_lock_action(gate)

    assert gate["policy"] == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_POLICY
    assert gate["policy"]["read_only"] is True
    assert gate["policy"]["would_write_memory"] is False
    assert gate["policy"]["would_modify_config"] is False
    assert gate["policy"]["would_write_graph"] is False
    assert gate["policy"]["does_not_create_operation_events"] is True
    assert gate["policy"]["creates_write_lock_candidates_only"] is True
    assert gate["policy"]["creates_real_proposals"] is False
    assert gate["policy"]["writes_proposal_files"] is False
    assert gate["policy"]["writes_operation_ledger"] is False
    assert gate["policy"]["applies_proposals"] is False
    assert gate["policy"]["persists_approvals"] is False
    assert gate["policy"]["submits_to_governance"] is False
    assert gate["policy"]["converts_to_real_proposal"] is False
    assert explanation["written"] is False
    assert explanation["submitted"] is False
    assert explanation["applied"] is False
    assert explanation["persisted"] is False
    assert explanation["approved"] is False
    assert explanation["created_real_proposal"] is False
    assert explanation["created_operation_event"] is False
    assert explanation["writes_proposal_files"] is False
    assert explanation["writes_operation_ledger"] is False
    assert explanation["converted_to_real_proposal"] is False
    assert recommendation["creates_real_proposals"] is False
    assert recommendation["writes_proposal_files"] is False
    assert recommendation["writes_operation_ledger"] is False
    assert not (hermes_home / "memory" / "graph" / "memory_graph.sqlite").exists()
    assert not (hermes_home / "memory" / "audit" / "memory_operation_ledger.jsonl").exists()
    assert not (hermes_home / "memory" / "proposals" / "memory_write_proposals.jsonl").exists()


def test_gate_must_never_be_marked_written_submitted_applied_persisted_approved_or_converted():
    gate = create_real_proposal_write_lock_gate(_dry_run())

    for forbidden_key in (
        "written",
        "submitted",
        "applied",
        "persisted",
        "approved",
        "created_real_proposal",
        "created_operation_event",
        "writes_proposal_files",
        "writes_operation_ledger",
        "converted_to_real_proposal",
    ):
        mutated = deepcopy(gate)
        mutated[forbidden_key] = True
        validation = validate_real_proposal_write_lock_gate(mutated)
        assert validation["valid"] is False
        assert f"{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_summary_counts_locked_eligible_gates_and_by_block_type_status():
    gates = [
        create_real_proposal_write_lock_gate(_dry_run("procedural_rules")),
        create_real_proposal_write_lock_gate(_dry_run("project_context")),
        create_real_proposal_write_lock_gate(_dry_run("procedural_rules", outcome="reject")),
    ]

    summary = summarize_real_proposal_write_lock_gates(gates)

    assert summary["total"] == 3
    assert summary["locked_count"] == 1
    assert summary["eligible_count"] == 2
    assert summary["valid_count"] == 3
    assert summary["invalid_count"] == 0
    assert summary["by_block_type"] == {"procedural_rules": 2, "project_context": 1}
    assert summary["by_status"] == {
        "eligible_for_human_approval_token": 2,
        "locked": 1,
    }
    assert summary["policy"] == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_POLICY
