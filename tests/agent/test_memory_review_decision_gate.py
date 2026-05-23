from copy import deepcopy

from agent.memory_block_review_queue import create_review_queue_item
from agent.memory_blocks import create_memory_block_candidate
from agent.memory_review_decision_gate import (
    MEMORY_REVIEW_DECISION_GATE_POLICY,
    SUPPORTED_REVIEW_DECISIONS,
    create_review_decision_candidate,
    evaluate_review_queue_item,
    explain_review_decision_candidate,
    recommend_review_decision_action,
    summarize_review_decisions,
    validate_review_decision_candidate,
)


def _queue_item(block_type, content="content", source_pattern_ids=None, source_fact_ids=None):
    block = create_memory_block_candidate(
        block_type,
        content,
        project_scope="memory-fabric",
        source_pattern_ids=source_pattern_ids,
        source_fact_ids=source_fact_ids,
    )
    return create_review_queue_item(block, reviewer="memory-reviewer")


def test_valid_low_risk_project_context_gets_approve_to_proposal():
    item = _queue_item("project_context", {"text": "foundation context"})

    candidate = evaluate_review_queue_item(item)

    assert candidate["decision"] == "approve_to_proposal"
    assert candidate["risk_level"] == "low"
    assert candidate["decision_validation"] == {"valid": True, "errors": []}
    assert candidate["next_step_recommendation"]["creates_real_proposals"] is False


def test_valid_procedural_rules_with_source_ids_gets_approve_to_proposal():
    item = _queue_item(
        "procedural_rules",
        {"rules": ["Create decision candidates only."]},
        source_pattern_ids=["pattern-1"],
        source_fact_ids=["fact-1"],
    )

    candidate = evaluate_review_queue_item(item)

    assert candidate["decision"] == "approve_to_proposal"
    assert candidate["source_pattern_ids"] == ["pattern-1"]
    assert candidate["source_fact_ids"] == ["fact-1"]


def test_procedural_rules_without_source_ids_requests_more_evidence():
    item = _queue_item("procedural_rules", {"rules": ["Needs sources."]})

    candidate = evaluate_review_queue_item(item)

    assert candidate["decision"] == "request_more_evidence"
    assert candidate["risk_level"] == "medium"


def test_safety_policy_requests_more_evidence():
    item = _queue_item("safety_policy", "Never write memory directly.")

    candidate = evaluate_review_queue_item(item)

    assert candidate["decision"] == "request_more_evidence"
    assert candidate["risk_level"] == "high"


def test_invalid_queue_item_rejects():
    item = _queue_item("project_context", {"text": "valid"})
    item["status"] = "approved"

    candidate = evaluate_review_queue_item(item)

    assert candidate["decision"] == "reject"
    assert candidate["queue_item_validation"]["valid"] is False
    assert "status_must_be_pending_review" in candidate["queue_item_validation"]["errors"]


def test_unsupported_invalid_block_rejects():
    item = _queue_item("unsupported", "invalid")

    candidate = evaluate_review_queue_item(item)

    assert candidate["decision"] == "reject"
    assert candidate["risk_level"] == "high"
    assert candidate["queue_item_snapshot"]["validation"]["valid"] is False
    assert "unsupported_block_type:unsupported" in candidate["queue_item_snapshot"]["validation"]["errors"]


def test_explicit_decision_override_is_validated_but_not_applied():
    item = _queue_item("safety_policy", "Requires review.")

    candidate = create_review_decision_candidate(
        item,
        reviewer="override-reviewer",
        decision="approve_to_proposal",
        rationale="Manual validation scenario.",
    )
    invalid = create_review_decision_candidate(item, decision="apply_now")

    assert candidate["decision"] == "approve_to_proposal"
    assert candidate["reviewer"] == "override-reviewer"
    assert candidate["decision_validation"] == {"valid": True, "errors": []}
    assert candidate["next_step_recommendation"]["applies_decisions"] is False
    assert invalid["decision_validation"]["valid"] is False
    assert "unsupported_decision:apply_now" in invalid["decision_validation"]["errors"]


def test_summary_counts_by_decision_and_risk():
    candidates = [
        evaluate_review_queue_item(_queue_item("project_context", "low")),
        evaluate_review_queue_item(
            _queue_item("procedural_rules", {"rules": ["source"]}, source_fact_ids=["fact-1"])
        ),
        evaluate_review_queue_item(_queue_item("procedural_rules", {"rules": ["no source"]})),
        evaluate_review_queue_item(_queue_item("safety_policy", "high")),
        evaluate_review_queue_item(_queue_item("unsupported", "invalid")),
    ]

    summary = summarize_review_decisions(candidates)

    assert summary["total"] == 5
    assert summary["by_decision"]["approve_to_proposal"] == 2
    assert summary["by_decision"]["request_more_evidence"] == 2
    assert summary["by_decision"]["reject"] == 1
    assert summary["by_decision"]["defer"] == 0
    assert summary["by_risk"] == {"high": 2, "medium": 2, "low": 1}
    assert set(summary["by_decision"]) == set(SUPPORTED_REVIEW_DECISIONS)


def test_input_queue_item_is_not_mutated():
    item = _queue_item("project_context", {"nested": ["original"]})
    before = deepcopy(item)

    candidate = evaluate_review_queue_item(item)
    candidate["queue_item_snapshot"]["block_snapshot"]["content"]["nested"].append("mutated-copy")

    assert item == before


def test_policy_proves_no_memory_config_graph_proposal_or_ledger_writes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    item = _queue_item("procedural_rules", {"rules": ["Only create decision candidates."]}, source_fact_ids=["fact-1"])

    candidate = evaluate_review_queue_item(item)
    explanation = explain_review_decision_candidate(candidate)
    recommendation = recommend_review_decision_action(candidate)

    assert candidate["policy"] == MEMORY_REVIEW_DECISION_GATE_POLICY
    assert validate_review_decision_candidate(candidate) == {"valid": True, "errors": []}
    assert candidate["policy"]["read_only"] is True
    assert candidate["policy"]["would_write_memory"] is False
    assert candidate["policy"]["would_modify_config"] is False
    assert candidate["policy"]["would_write_graph"] is False
    assert candidate["policy"]["does_not_create_operation_events"] is True
    assert candidate["policy"]["creates_decision_candidates_only"] is True
    assert candidate["policy"]["applies_decisions"] is False
    assert candidate["policy"]["creates_real_proposals"] is False
    assert explanation["applied"] is False
    assert explanation["created_real_proposal"] is False
    assert explanation["created_operation_event"] is False
    assert recommendation["creates_real_proposals"] is False
    assert not (hermes_home / "memory" / "graph" / "memory_graph.sqlite").exists()
    assert not (hermes_home / "memory" / "audit" / "memory_operation_ledger.jsonl").exists()
    assert not (hermes_home / "memory" / "proposals" / "memory_write_proposals.jsonl").exists()
