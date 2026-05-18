from agent.model_routing.policy import RoutingContext, load_policy
from agent.model_routing.policy_router import recommend_model


def _decision(**overrides):
    defaults = {
        "task_type": "daily_ops",
        "agent_role": "COO",
        "risk_level": "medium",
        "client_facing": False,
        "sensitive_data": False,
        "final_authority": False,
        "complexity": "routine",
    }
    defaults.update(overrides)
    context = RoutingContext(**defaults)
    return recommend_model(context, load_policy())


def test_client_facing_final_authority_never_uses_free_model():
    decision = _decision(
        task_type="client_facing_draft",
        agent_role="CMO",
        risk_level="high",
        client_facing=True,
        final_authority=True,
    )

    assert decision.tier == "S"
    assert decision.model == "qwen/qwen3.6-plus"
    assert decision.free_model_allowed is False
    assert "client-facing" in decision.reason


def test_security_privacy_final_decision_routes_to_strong_model():
    decision = _decision(
        task_type="security_privacy_review",
        agent_role="CISO",
        risk_level="high",
        sensitive_data=True,
        final_authority=True,
    )

    assert decision.tier == "S"
    assert decision.model == "qwen/qwen3.6-plus"
    assert decision.approval_required is True


def test_daily_operations_default_to_paid_budget_not_strong():
    decision = _decision(task_type="daily_ops", agent_role="COO", risk_level="medium")

    assert decision.tier == "B"
    assert decision.model == "deepseek/deepseek-v4-flash"
    assert decision.estimated_cost_class == "budget"


def test_code_explanation_can_use_free_coding_model():
    decision = _decision(
        task_type="code_explanation",
        agent_role="Front-End Engineer",
        risk_level="low",
        final_authority=False,
        complexity="simple",
    )

    assert decision.tier == "F"
    assert decision.model == "qwen/qwen3-coder:free"
    assert decision.free_model_allowed is True


def test_coding_implementation_planning_routes_to_budget_or_strong_not_free():
    decision = _decision(
        task_type="coding_implementation_planning",
        agent_role="CTO",
        risk_level="medium",
        final_authority=False,
        complexity="moderate",
    )

    assert decision.tier == "B"
    assert decision.model == "qwen/qwen3-coder-next"
    assert decision.free_model_allowed is False


def test_openrouter_free_is_never_selected_for_governed_workflows():
    for task_type in ["daily_ops", "strategy", "client_facing_draft", "code_explanation"]:
        decision = _decision(task_type=task_type)
        assert decision.model != "openrouter/free"


def test_strong_model_decision_includes_escalation_reason():
    decision = _decision(task_type="strategy", agent_role="Hermes", risk_level="high")

    assert decision.tier == "S"
    assert decision.escalation_reason
    assert "strategy" in decision.escalation_reason.lower()


def test_unknown_task_fails_safe_to_budget_model_not_free_or_random():
    decision = _decision(task_type="unknown_new_task", agent_role="Unknown", risk_level="medium")

    assert decision.tier == "B"
    assert decision.model == "deepseek/deepseek-v4-flash"
    assert decision.model != "openrouter/free"
    assert decision.policy_warnings
