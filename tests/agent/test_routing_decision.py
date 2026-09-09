from __future__ import annotations

from agent.routing_decision import (
    add_routing_summary,
    build_routing_decision,
    evaluate_reviewer_independence,
    initialize_agent_routing_decision,
    normalize_fallback_reason,
    record_agent_fallback,
    record_agent_primary_restore,
    record_fallback,
    routing_summary,
)


def _codex_decision():
    return build_routing_decision(
        task_id="HSK-005",
        board="automation",
        task_type="implementation",
        capability="implement",
        risk="normal",
        code_change=True,
        independent_review=True,
        preferred_profile="rozmilo-codex",
        reviewer_profile="rozmilo-claude",
        selected_profile="rozmilo-codex",
        selected_provider="openai-codex",
        selected_model="gpt-5.6-sol",
        human_gate_required=False,
        independence_valid=True,
        policy_digest=None,
        selected_at="2026-09-09T12:00:00Z",
        selected_by="dispatcher",
        run_id=42,
        session_id="session-1",
    )


def test_normal_codex_route_records_initial_provider_and_model():
    decision = _codex_decision()

    assert decision["routing_contract_version"] == "phase-c-v1"
    assert decision["selected_profile"] == "rozmilo-codex"
    assert decision["initial_provider"] == "openai-codex"
    assert decision["initial_model"] == "gpt-5.6-sol"
    assert decision["selected_provider"] == "openai-codex"
    assert decision["selected_model"] == "gpt-5.6-sol"
    assert decision["fallback_used"] is False
    assert decision["fallback_reason"] is None
    assert decision["routing_history"] == [
        {
            "phase": "initial",
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
            "reason": None,
            "recorded_at": "2026-09-09T12:00:00Z",
        }
    ]


def test_declared_codex_to_copilot_fallback_records_both_sides_and_known_reason():
    decision = record_fallback(
        _codex_decision(),
        from_provider="openai-codex",
        from_model="gpt-5.6-sol",
        to_provider="copilot",
        to_model="gpt-5.6-luna",
        reason="rate_limit",
        recorded_at="2026-09-09T12:01:00Z",
    )

    assert decision["fallback_used"] is True
    assert decision["fallback_from_provider"] == "openai-codex"
    assert decision["fallback_from_model"] == "gpt-5.6-sol"
    assert decision["selected_provider"] == "copilot"
    assert decision["selected_model"] == "gpt-5.6-luna"
    assert decision["fallback_reason"] == "rate_limit"
    assert decision["routing_history"][-1] == {
        "phase": "fallback",
        "provider": "copilot",
        "model": "gpt-5.6-luna",
        "reason": "rate_limit",
        "recorded_at": "2026-09-09T12:01:00Z",
        "from_provider": "openai-codex",
        "from_model": "gpt-5.6-sol",
    }


def test_unknown_fallback_reason_remains_explicit_unknown():
    decision = record_fallback(
        _codex_decision(),
        from_provider="openai-codex",
        from_model="gpt-5.6-sol",
        to_provider="copilot",
        to_model="gpt-5.6-luna",
        reason=None,
        recorded_at="2026-09-09T12:01:00Z",
    )

    assert decision["fallback_reason"] == "unknown"
    assert decision["routing_history"][-1]["reason"] == "unknown"


def test_stable_fallback_categories_are_preserved():
    for category in (
        "quota",
        "rate_limit",
        "auth",
        "timeout",
        "provider_error",
        "model_unavailable",
        "policy_fallback",
        "unknown",
    ):
        assert normalize_fallback_reason(category) == category
    assert normalize_fallback_reason("content_policy_blocked") == "policy_fallback"


def test_mixed_provider_summary_preserves_primary_and_final_identity():
    decision = record_fallback(
        _codex_decision(),
        from_provider="openai-codex",
        from_model="gpt-5.6-sol",
        to_provider="copilot",
        to_model="gpt-5.6-luna",
        reason="quota",
        recorded_at="2026-09-09T12:01:00Z",
    )

    summary = routing_summary(decision, final_provider="copilot", final_model="gpt-5.6-luna")

    assert summary == {
        "routing_provenance": "phase-c-v1",
        "routing_decision_id": decision["decision_id"],
        "primary_profile": "rozmilo-codex",
        "initial_provider": "openai-codex",
        "initial_model": "gpt-5.6-sol",
        "final_provider": "copilot",
        "final_model": "gpt-5.6-luna",
        "fallback_used": True,
        "fallback_reason": "quota",
    }


def test_legacy_summary_is_readable_and_explicitly_unknown():
    assert routing_summary(None, final_provider="copilot", final_model="gpt-5.6-luna") == {
        "routing_provenance": "legacy_unknown",
        "routing_decision_id": None,
        "primary_profile": None,
        "initial_provider": None,
        "initial_model": None,
        "final_provider": "copilot",
        "final_model": "gpt-5.6-luna",
        "fallback_used": None,
        "fallback_reason": None,
    }


def test_reviewer_independence_accepts_codex_to_claude():
    assert evaluate_reviewer_independence(
        implementation_profile="rozmilo-codex",
        reviewer_profile="rozmilo-claude",
        independent_review=True,
    ) is True


def test_reviewer_independence_accepts_claude_to_codex():
    assert evaluate_reviewer_independence(
        implementation_profile="rozmilo-claude",
        reviewer_profile="rozmilo-codex",
        independent_review=True,
    ) is True


def test_reviewer_independence_rejects_required_self_review():
    assert evaluate_reviewer_independence(
        implementation_profile="rozmilo-codex",
        reviewer_profile="rozmilo-codex",
        independent_review=True,
    ) is False


def test_implementation_provider_fallback_does_not_invalidate_opposite_profile():
    assert evaluate_reviewer_independence(
        implementation_profile="rozmilo-codex",
        reviewer_profile="rozmilo-claude",
        independent_review=True,
        implementation_provider="copilot",
    ) is True


def test_reviewer_provider_fallback_to_implementer_provider_fails_closed():
    assert evaluate_reviewer_independence(
        implementation_profile="rozmilo-codex",
        reviewer_profile="rozmilo-claude",
        independent_review=True,
        implementation_provider="copilot",
        reviewer_provider="copilot",
    ) is False


class _SessionDB:
    def __init__(self):
        self.patches = []

    def patch_session_model_config(self, session_id, patch):
        self.patches.append((session_id, patch))


class _Agent:
    def __init__(self):
        self.provider = "openai-codex"
        self.model = "gpt-5.6-sol"
        self.session_id = "session-1"
        self._session_db = _SessionDB()
        self._session_init_model_config = {}


def test_agent_initial_selection_is_staged_for_session_persistence():
    agent = _Agent()
    decision = initialize_agent_routing_decision(
        agent,
        routing_context={
            "task_id": "HSK-005",
            "board": "automation",
            "run_id": 42,
            "task_type": "implementation",
            "capability": "implement",
            "risk": "normal",
            "code_change": True,
            "independent_review": True,
            "preferred_profile": "rozmilo-codex",
            "reviewer_profile": "rozmilo-claude",
            "selected_profile": "rozmilo-codex",
            "human_gate_required": False,
            "selected_by": "dispatcher",
        },
        selected_at="2026-09-09T12:00:00Z",
    )

    assert decision["initial_provider"] == "openai-codex"
    assert decision["initial_model"] == "gpt-5.6-sol"
    assert decision["independence_valid"] is True
    assert agent._session_init_model_config["routing_decision"] == decision
    assert agent._session_db.patches == [("session-1", {"routing_decision": decision})]


def test_agent_initial_selection_persists_in_real_session_store(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-1", source="cli")
    agent = _Agent()
    agent._session_db = db

    try:
        decision = initialize_agent_routing_decision(
            agent,
            routing_context={"selected_profile": "rozmilo-codex"},
            selected_at="2026-09-09T12:00:00Z",
        )

        assert db.get_session_model_config_value(
            "session-1", "routing_decision"
        ) == decision
    finally:
        db.close()


def test_resumed_session_keeps_original_decision_and_fallback_history(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-1", source="cli")
    first_agent = _Agent()
    first_agent._session_db = db

    try:
        initial = initialize_agent_routing_decision(
            first_agent,
            routing_context={"selected_profile": "rozmilo-codex"},
            selected_at="2026-09-09T12:00:00Z",
        )
        first_agent.provider = "copilot"
        first_agent.model = "gpt-5.6-luna"
        fallback = record_agent_fallback(
            first_agent,
            from_provider="openai-codex",
            from_model="gpt-5.6-sol",
            reason="rate_limit",
            recorded_at="2026-09-09T12:01:00Z",
        )

        resumed_agent = _Agent()
        resumed_agent._session_db = db
        resumed = initialize_agent_routing_decision(
            resumed_agent,
            routing_context={"selected_profile": "rozmilo-codex"},
            selected_at="2026-09-09T13:00:00Z",
        )

        assert resumed == fallback
        assert resumed["decision_id"] == initial["decision_id"]
        assert len(resumed["routing_history"]) == 2
    finally:
        db.close()


def test_delegated_child_does_not_claim_parent_kanban_routing_context(monkeypatch):
    from agent.delegation_context import delegated_child_context, scrub_kanban_env

    routing_context = '{"task_id":"parent-task","run_id":42,"board":"automation"}'
    monkeypatch.setenv("HERMES_ROUTING_CONTEXT", routing_context)
    agent = _Agent()

    with delegated_child_context():
        decision = initialize_agent_routing_decision(
            agent,
            selected_at="2026-09-09T12:00:00Z",
        )

    assert decision["task_id"] is None
    assert decision["run_id"] is None
    assert "HERMES_ROUTING_CONTEXT" not in scrub_kanban_env(
        {"HERMES_ROUTING_CONTEXT": routing_context}
    )


def test_agent_fallback_updates_session_decision_with_classifier_reason():
    from agent.error_classifier import FailoverReason

    agent = _Agent()
    initialize_agent_routing_decision(agent, routing_context={}, selected_at="2026-09-09T12:00:00Z")
    agent.provider = "copilot"
    agent.model = "gpt-5.6-luna"

    decision = record_agent_fallback(
        agent,
        from_provider="openai-codex",
        from_model="gpt-5.6-sol",
        reason=FailoverReason.rate_limit,
        recorded_at="2026-09-09T12:01:00Z",
    )

    assert decision["selected_provider"] == "copilot"
    assert decision["selected_model"] == "gpt-5.6-luna"
    assert decision["fallback_reason"] == "rate_limit"
    assert agent._session_db.patches[-1] == ("session-1", {"routing_decision": decision})


def test_usage_summary_adds_route_fields_without_changing_legacy_provider_model():
    agent = _Agent()
    initialize_agent_routing_decision(agent, routing_context={}, selected_at="2026-09-09T12:00:00Z")
    agent.provider = "copilot"
    agent.model = "gpt-5.6-luna"
    record_agent_fallback(
        agent,
        from_provider="openai-codex",
        from_model="gpt-5.6-sol",
        reason=None,
        recorded_at="2026-09-09T12:01:00Z",
    )
    legacy = {"provider": "copilot", "model": "gpt-5.6-luna"}

    result = add_routing_summary(legacy, agent)

    assert result["provider"] == "copilot"
    assert result["model"] == "gpt-5.6-luna"
    assert result["initial_provider"] == "openai-codex"
    assert result["final_provider"] == "copilot"
    assert result["fallback_used"] is True
    assert result["fallback_reason"] == "unknown"


def test_primary_restore_is_append_only_and_keeps_fallback_provenance():
    agent = _Agent()
    initialize_agent_routing_decision(agent, routing_context={}, selected_at="2026-09-09T12:00:00Z")
    agent.provider = "copilot"
    agent.model = "gpt-5.6-luna"
    record_agent_fallback(
        agent,
        from_provider="openai-codex",
        from_model="gpt-5.6-sol",
        reason="rate_limit",
        recorded_at="2026-09-09T12:01:00Z",
    )
    agent.provider = "openai-codex"
    agent.model = "gpt-5.6-sol"

    decision = record_agent_primary_restore(
        agent,
        from_provider="copilot",
        from_model="gpt-5.6-luna",
        recorded_at="2026-09-09T12:02:00Z",
    )

    assert decision["selected_provider"] == "openai-codex"
    assert decision["selected_model"] == "gpt-5.6-sol"
    assert decision["fallback_used"] is True
    assert decision["fallback_from_provider"] == "openai-codex"
    assert decision["fallback_reason"] == "rate_limit"
    assert decision["routing_history"][-1] == {
        "phase": "primary_restored",
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "reason": None,
        "recorded_at": "2026-09-09T12:02:00Z",
        "from_provider": "copilot",
        "from_model": "gpt-5.6-luna",
    }
