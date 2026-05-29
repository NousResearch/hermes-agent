"""Rule-based write gate tests for Memory v2 online ingestion."""

from __future__ import annotations

from plugins.memory.memory_v2.write_gate import RuleBasedWriteGate, WriteGateOutcome


def classify(text: str):
    return RuleBasedWriteGate().classify(text)


def test_write_gate_archives_ordinary_or_ephemeral_turns_without_candidate():
    ordinary = classify("hello, what is 2 + 2?")
    ephemeral = classify("Remember that today I parked by the blue sign.")

    assert ordinary.outcome == WriteGateOutcome.ARCHIVE_ONLY
    assert ordinary.should_create_candidate is False
    assert ephemeral.outcome == WriteGateOutcome.ARCHIVE_ONLY
    assert ephemeral.should_create_candidate is False
    assert "ephemeral" in ephemeral.reason.lower()


def test_write_gate_classifies_explicit_user_preference_as_core_update_candidate():
    decision = classify("Remember that Dylan prefers direct, no-BS, tool-grounded help.")

    assert decision.outcome == WriteGateOutcome.CORE_UPDATE
    assert decision.memory_type == "preference"
    assert decision.claim == "Dylan prefers direct, no-BS, tool-grounded help."
    assert decision.proposed_destination == "semantic/items"
    assert decision.should_create_candidate is True
    assert decision.importance >= 0.8


def test_write_gate_classifies_project_update_as_project_update_candidate():
    decision = classify("Remember that Memory v2 current plan is to add a write gate classifier.")

    assert decision.outcome == WriteGateOutcome.PROJECT_UPDATE
    assert decision.memory_type == "project_state"
    assert "write gate classifier" in decision.claim
    assert decision.should_create_candidate is True


def test_write_gate_classifies_environment_conflict_as_review_candidate():
    decision = classify("Remember that Hermes is running on macOS, not WSL.")

    assert decision.outcome == WriteGateOutcome.SUPERSEDE_EXISTING
    assert decision.memory_type == "environment"
    assert decision.requires_review is True
    assert "conflict" in decision.reason.lower() or "contradict" in decision.reason.lower()


def test_write_gate_routes_procedures_to_skill_candidate_not_semantic_fact():
    decision = classify("Remember that when modifying Hermes providers, load the hermes-agent skill first.")

    assert decision.outcome == WriteGateOutcome.SKILL_CANDIDATE
    assert decision.memory_type == "procedure_ref"
    assert decision.should_create_candidate is True
    assert "skill" in decision.reason.lower()
