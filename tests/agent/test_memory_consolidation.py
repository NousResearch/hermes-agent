from agent.memory_consolidation import (
    classify_memory_candidate,
    semantic_record_fields_from_decision,
)


def test_user_preference_classifies_as_high_salience_profile_fact():
    decision = classify_memory_candidate("user", "User prefers direct blunt feedback.")

    assert decision.action == "semantic_add"
    assert decision.target == "user"
    assert decision.salience >= 0.8
    assert decision.confidence >= 0.8

    fields = semantic_record_fields_from_decision(decision, "user")
    assert fields["kind"] == "user_profile_fact"
    assert fields["salience"] == decision.salience


def test_project_fact_classifies_as_semantic_fact_not_procedure():
    decision = classify_memory_candidate("memory", "Project uses pytest with xdist.")

    assert decision.action == "semantic_add"
    assert decision.target == "memory"

    fields = semantic_record_fields_from_decision(decision, "memory")
    assert fields["kind"] == "semantic_fact"


def test_workflow_candidate_is_advisory_procedural_skill_candidate():
    decision = classify_memory_candidate(
        "memory",
        "When deploying, first run pytest, then check the release notes.",
    )

    assert decision.action == "procedural_skill_candidate"
    assert decision.target == "skill"
    assert decision.warnings

    fields = semantic_record_fields_from_decision(decision, "memory")
    assert fields["kind"] == "procedural_candidate"
    assert "consolidation_warnings" in fields


def test_task_progress_candidate_is_low_salience_episodic_only():
    decision = classify_memory_candidate(
        "memory",
        "Fixed PR #123 today and all tests passed in 4.2s.",
    )

    assert decision.action == "episodic_only"
    assert decision.target == "none"
    assert decision.salience < 0.3
    assert decision.warnings

    fields = semantic_record_fields_from_decision(decision, "memory")
    assert fields["kind"] == "episodic_note"
