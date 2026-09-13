from types import SimpleNamespace

from agent.reasoning_escalation import (
    apply_turn_reasoning_escalation,
    decide_reasoning_escalation,
    effective_reasoning_config,
)


BASE_POLICY = {
    "enabled": True,
    "providers": ["openai-codex"],
    "models": ["gpt-5.6-sol"],
    "base_effort": "medium",
    "escalated_effort": "high",
    "threshold": 9,
    "min_dimensions": 3,
    "announce": False,
    "log_decisions": False,
}


def _agent(*, provider="openai-codex", model="gpt-5.6-sol", effort="medium"):
    return SimpleNamespace(
        provider=provider,
        model=model,
        reasoning_config={"enabled": True, "effort": effort},
        session_id="test-session",
        _turn_reasoning_config_override=None,
        _turn_reasoning_escalation_target=None,
        _turn_reasoning_escalation_decision=None,
        _emit_status=lambda _message: None,
    )


def test_simple_prompt_stays_at_medium():
    decision = decide_reasoning_escalation(
        "What time is it?",
        threshold=9,
        min_dimensions=3,
        base_effort="medium",
        escalated_effort="high",
    )
    assert decision.escalate is False
    assert decision.selected_effort == "medium"


def test_long_but_simple_prompt_does_not_escalate_on_length_alone():
    prompt = "Summarize this sentence. " + ("ordinary text " * 350)
    decision = decide_reasoning_escalation(
        prompt,
        threshold=9,
        min_dimensions=3,
        base_effort="medium",
        escalated_effort="high",
    )
    assert decision.escalate is False
    assert decision.dimensions == ("large_prompt",)


def test_complex_multi_boundary_prompt_escalates_to_high():
    prompt = """
    Perform an end-to-end production architecture audit and implementation. Inspect the
    model-routing config, runtime gateway, source code, tests, security and privacy
    boundaries, and provider documentation. Diagnose root causes, compare designs,
    implement the safest option, preserve unrelated routes, create backups and rollback,
    add unit and integration tests, benchmark the result, validate live logs and database
    evidence, and deliver a cited launch report with every requirement verified.
    """
    decision = decide_reasoning_escalation(
        prompt,
        threshold=9,
        min_dimensions=3,
        base_effort="medium",
        escalated_effort="high",
    )
    assert decision.escalate is True
    assert decision.selected_effort == "high"
    assert decision.score >= 9
    assert len(decision.dimensions) >= 3


def test_explicit_medium_only_instruction_blocks_high():
    prompt = "Audit the production stack, but stay on medium only and do not escalate to high."
    decision = decide_reasoning_escalation(
        prompt,
        threshold=1,
        min_dimensions=1,
        base_effort="medium",
        escalated_effort="high",
    )
    assert decision.escalate is False
    assert decision.selected_effort == "medium"
    assert "explicit_opt_out" in decision.reasons


def test_policy_applies_only_to_target_provider_model_and_medium_baseline():
    complex_prompt = (
        "End-to-end production audit: design, implement, debug, research, test, verify, "
        "secure, back up, roll back, and validate the gateway, repo, routing config, "
        "provider docs, logs, and database with many requirements."
    )
    wrong_provider = _agent(provider="openrouter")
    wrong_model = _agent(model="gpt-5.5")
    wrong_baseline = _agent(effort="high")

    assert apply_turn_reasoning_escalation(wrong_provider, complex_prompt, BASE_POLICY).escalate is False
    assert apply_turn_reasoning_escalation(wrong_model, complex_prompt, BASE_POLICY).escalate is False
    assert apply_turn_reasoning_escalation(wrong_baseline, complex_prompt, BASE_POLICY).escalate is False
    assert effective_reasoning_config(wrong_provider)["effort"] == "medium"
    assert effective_reasoning_config(wrong_model)["effort"] == "medium"
    assert effective_reasoning_config(wrong_baseline)["effort"] == "high"


def test_high_override_is_ephemeral_and_resets_on_next_turn():
    agent = _agent()
    complex_prompt = """
    End-to-end production migration: audit architecture, research official sources,
    compare designs, implement code across runtime boundaries, preserve security and
    privacy constraints, create backups and rollback, run unit and integration tests,
    verify live gateway logs and database state, and produce a comprehensive report.
    """

    first = apply_turn_reasoning_escalation(agent, complex_prompt, BASE_POLICY)
    assert first.escalate is True
    assert effective_reasoning_config(agent) == {"enabled": True, "effort": "high"}

    second = apply_turn_reasoning_escalation(agent, "What time is it?", BASE_POLICY)
    assert second.escalate is False
    assert agent._turn_reasoning_config_override is None
    assert agent._turn_reasoning_escalation_target is None
    assert effective_reasoning_config(agent) == {"enabled": True, "effort": "medium"}


def test_fallback_model_cannot_inherit_high_override():
    agent = _agent()
    agent._turn_reasoning_config_override = {"enabled": True, "effort": "high"}
    agent._turn_reasoning_escalation_target = ("openai-codex", "gpt-5.6-sol")
    agent.provider = "openrouter"
    agent.model = "~deepseek/deepseek-v4-flash-latest"
    assert effective_reasoning_config(agent) == {"enabled": True, "effort": "medium"}


def test_disabled_policy_fails_closed_to_medium():
    agent = _agent()
    policy = {**BASE_POLICY, "enabled": False}
    decision = apply_turn_reasoning_escalation(
        agent,
        "Complex production audit with tests, backups, rollback, security, and research.",
        policy,
    )
    assert decision.escalate is False
    assert effective_reasoning_config(agent) == {"enabled": True, "effort": "medium"}
