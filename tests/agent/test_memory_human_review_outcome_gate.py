from copy import deepcopy

from agent.memory_block_review_queue import create_review_queue_item
from agent.memory_blocks import create_memory_block_candidate
from agent.memory_governance_submission_packet import create_governance_submission_packet
from agent.memory_human_review_outcome_gate import (
    MEMORY_HUMAN_REVIEW_OUTCOME_GATE_POLICY,
    MEMORY_HUMAN_REVIEW_OUTCOME_KIND,
    MEMORY_HUMAN_REVIEW_OUTCOME_ROUTING,
    MEMORY_HUMAN_REVIEW_OUTCOME_STATUS,
    create_human_review_outcome_candidate,
    evaluate_human_review_packet,
    explain_human_review_outcome_candidate,
    recommend_human_review_outcome_action,
    summarize_human_review_outcomes,
    validate_human_review_outcome_candidate,
)
from agent.memory_proposal_draft_builder import create_memory_proposal_draft
from agent.memory_proposal_governance_gate import create_governance_submission_candidate
from agent.memory_review_decision_gate import evaluate_review_queue_item


def _packet(block_type="procedural_rules", source_pattern_ids=None, source_fact_ids=None):
    block = create_memory_block_candidate(
        block_type,
        {"rules": ["Create outcome candidates only."], "nested": {"value": "preserved"}},
        project_scope="memory-fabric",
        source_pattern_ids=source_pattern_ids if source_pattern_ids is not None else ["pattern-1"],
        source_fact_ids=source_fact_ids if source_fact_ids is not None else ["fact-1"],
        metadata={"source": "test"},
    )
    queue_item = create_review_queue_item(block, reviewer="memory-reviewer")
    decision = evaluate_review_queue_item(queue_item, reviewer="memory-reviewer")
    draft = create_memory_proposal_draft(decision, author="proposal-drafter")
    submission = create_governance_submission_candidate(draft, reviewer="governance-reviewer")
    return create_governance_submission_packet(submission, reviewer="packet-reviewer")


def test_valid_packet_becomes_approve_real_proposal_creation_outcome_candidate():
    packet = _packet()

    candidate = create_human_review_outcome_candidate(packet, reviewer="human-reviewer")

    assert candidate["outcome_kind"] == MEMORY_HUMAN_REVIEW_OUTCOME_KIND
    assert candidate["outcome_status"] == MEMORY_HUMAN_REVIEW_OUTCOME_STATUS
    assert candidate["routing"] == MEMORY_HUMAN_REVIEW_OUTCOME_ROUTING
    assert candidate["source_packet_id"] == packet["packet_id"]
    assert candidate["source_submission_id"] == packet["source_submission_id"]
    assert candidate["source_draft_id"] == packet["source_draft_id"]
    assert candidate["source_decision_id"] == packet["source_decision_id"]
    assert candidate["source_queue_item_id"] == packet["source_queue_item_id"]
    assert candidate["reviewer"] == "human-reviewer"
    assert candidate["outcome"] == "approve_real_proposal_creation"
    assert candidate["outcome_validation"] == {"valid": True, "errors": []}
    assert validate_human_review_outcome_candidate(candidate) == {"valid": True, "errors": []}
    assert candidate["next_step_recommendation"]["action"] == "manual_real_proposal_creation_required"
    assert candidate["next_step_recommendation"]["creates_real_proposals"] is False
    assert candidate["next_step_recommendation"]["converts_to_real_proposal"] is False


def test_invalid_packet_rejects():
    packet = _packet()
    packet["packet_status"] = "submitted"

    evaluation = evaluate_human_review_packet(packet, reviewer="human-reviewer")
    candidate = create_human_review_outcome_candidate(packet, reviewer="human-reviewer")

    assert evaluation["outcome"] == "reject"
    assert candidate["outcome"] == "reject"
    assert candidate["packet_validation"]["valid"] is False
    assert candidate["outcome_validation"] == {"valid": True, "errors": []}
    assert candidate["next_step_recommendation"]["action"] == "do_not_create_real_proposal"


def test_missing_payload_preview_requests_changes():
    packet = _packet()
    packet.pop("payload_preview")

    candidate = create_human_review_outcome_candidate(packet)

    assert candidate["outcome"] == "request_changes"
    assert "missing_payload_preview" in candidate["packet_validation"]["errors"]
    assert candidate["next_step_recommendation"]["action"] == "return_packet_for_changes"


def test_missing_source_evidence_requests_changes():
    packet = _packet(source_pattern_ids=[], source_fact_ids=[])

    candidate = create_human_review_outcome_candidate(packet)

    assert candidate["outcome"] == "request_changes"
    assert "missing_source_evidence" in candidate["packet_validation"]["errors"]
    assert candidate["next_step_recommendation"]["action"] == "return_packet_for_changes"


def test_explicit_supported_outcome_override_is_accepted_but_not_applied():
    packet = _packet()

    candidate = create_human_review_outcome_candidate(
        packet,
        reviewer="human-reviewer",
        outcome="defer",
        rationale="Reviewer wants another human to inspect packet wording.",
    )
    recommendation = recommend_human_review_outcome_action(candidate)

    assert candidate["outcome"] == "defer"
    assert validate_human_review_outcome_candidate(candidate) == {"valid": True, "errors": []}
    assert recommendation["action"] == "defer_manual_review_outcome"
    assert recommendation["creates_real_proposals"] is False
    assert recommendation["applies_proposals"] is False
    assert recommendation["persists_approvals"] is False


def test_unsupported_outcome_fails_validation():
    candidate = create_human_review_outcome_candidate(_packet(), outcome="ship_it")

    validation = validate_human_review_outcome_candidate(candidate)

    assert validation["valid"] is False
    assert "unsupported_human_review_outcome" in validation["errors"]
    assert candidate["next_step_recommendation"]["action"] == "do_not_use_outcome_candidate"


def test_input_packet_is_not_mutated():
    packet = _packet()
    before = deepcopy(packet)

    candidate = create_human_review_outcome_candidate(packet)
    candidate["source_packet_snapshot"]["payload_preview"]["content"]["nested"]["value"] = "changed"

    assert packet == before


def test_policy_proves_no_memory_config_graph_proposal_or_ledger_writes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    candidate = create_human_review_outcome_candidate(_packet())
    explanation = explain_human_review_outcome_candidate(candidate)
    recommendation = recommend_human_review_outcome_action(candidate)

    assert candidate["policy"] == MEMORY_HUMAN_REVIEW_OUTCOME_GATE_POLICY
    assert candidate["policy"]["read_only"] is True
    assert candidate["policy"]["would_write_memory"] is False
    assert candidate["policy"]["would_modify_config"] is False
    assert candidate["policy"]["would_write_graph"] is False
    assert candidate["policy"]["does_not_create_operation_events"] is True
    assert candidate["policy"]["creates_outcome_candidates_only"] is True
    assert candidate["policy"]["creates_real_proposals"] is False
    assert candidate["policy"]["applies_proposals"] is False
    assert candidate["policy"]["persists_approvals"] is False
    assert candidate["policy"]["submits_to_governance"] is False
    assert candidate["policy"]["converts_to_real_proposal"] is False
    assert explanation["submitted"] is False
    assert explanation["applied"] is False
    assert explanation["persisted"] is False
    assert explanation["approved"] is False
    assert explanation["converted_to_real_proposal"] is False
    assert explanation["created_real_proposal"] is False
    assert explanation["created_operation_event"] is False
    assert explanation["submitted_to_governance"] is False
    assert recommendation["creates_real_proposals"] is False
    assert recommendation["submits_to_governance"] is False
    assert recommendation["converts_to_real_proposal"] is False
    assert not (hermes_home / "memory" / "graph" / "memory_graph.sqlite").exists()
    assert not (hermes_home / "memory" / "audit" / "memory_operation_ledger.jsonl").exists()
    assert not (hermes_home / "memory" / "proposals" / "memory_write_proposals.jsonl").exists()


def test_outcome_candidate_must_never_be_marked_submitted_applied_persisted_approved_or_converted():
    candidate = create_human_review_outcome_candidate(_packet())

    for forbidden_key in (
        "submitted",
        "applied",
        "persisted",
        "approved",
        "converted_to_real_proposal",
        "created_real_proposal",
        "created_operation_event",
        "created_proposal_record",
        "submitted_to_governance",
    ):
        mutated = deepcopy(candidate)
        mutated[forbidden_key] = True
        validation = validate_human_review_outcome_candidate(mutated)
        assert validation["valid"] is False
        assert f"{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_summary_counts_by_outcome_and_block_type_status():
    candidates = [
        create_human_review_outcome_candidate(_packet("procedural_rules")),
        create_human_review_outcome_candidate(_packet("project_context"), outcome="defer"),
        create_human_review_outcome_candidate(_packet("procedural_rules", source_pattern_ids=[], source_fact_ids=[])),
    ]

    summary = summarize_human_review_outcomes(candidates)

    assert summary["total"] == 3
    assert summary["valid_count"] == 3
    assert summary["invalid_count"] == 0
    assert summary["by_outcome"] == {
        "approve_real_proposal_creation": 1,
        "defer": 1,
        "request_changes": 1,
    }
    assert summary["by_block_type"] == {"procedural_rules": 2, "project_context": 1}
    assert summary["by_status"] == {"review_outcome_candidate": 3}
    assert summary["policy"] == MEMORY_HUMAN_REVIEW_OUTCOME_GATE_POLICY
