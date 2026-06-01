from agent.managed_agents.capability_matrix import build_capability_matrix, preview_route
from agent.managed_agents.failure_reroute import decide_failure_reroute
from agent.managed_agents.model_tier_router import resolve_model_tier
from agent.managed_agents.registry import AgentRegistry


def _registry() -> AgentRegistry:
    return AgentRegistry.from_yaml(
        {
            "version": "test",
            "agents": [
                {
                    "agent_id": "claude",
                    "name": "Claude",
                    "role": "lead_implementer",
                    "model_ref": "claude_sonnet",
                    "runtime": "claude_code_cli",
                    "model_strategy": {
                        "mode": "external",
                        "primary": "claude_sonnet",
                        "chain": ["claude_sonnet", "claude_opus"],
                    },
                    "tools": ["file", "terminal", "git"],
                    "permission": "ask",
                    "capabilities": ["code_edit", "test_run", "refactor"],
                    "risk_allowed": ["R1", "R2", "R3"],
                },
                {
                    "agent_id": "deepseek-tui",
                    "name": "DeepSeek",
                    "role": "fast_worker",
                    "model_ref": "deepseek_pro",
                    "runtime": "deepseek_tui_cli",
                    "model_strategy": {
                        "mode": "fallback",
                        "primary": "opencode_go_deepseek_flash",
                        "chain": ["opencode_go_deepseek_flash", "opencode_go_deepseek_pro", "deepseek_flash", "deepseek_pro"],
                        "fallback_on": ["timeout", "rate_limited"],
                    },
                    "tools": ["file", "terminal"],
                    "permission": "ask",
                    "capabilities": ["small_fix", "test_generation", "bug_reproduction"],
                    "risk_allowed": ["R0", "R1", "R2"],
                },
                {
                    "agent_id": "opencode",
                    "name": "OpenCode",
                    "role": "external_collaboration_worker",
                    "model_ref": "opencode_go_deepseek_flash",
                    "runtime": "opencode_cli",
                    "tools": ["file", "terminal"],
                    "permission": "ask",
                    "capabilities": ["code_review", "external_quick_check", "bug_reproduction"],
                    "risk_allowed": ["R0", "R1", "R2"],
                },
                {
                    "agent_id": "codex",
                    "name": "Codex",
                    "role": "principal_engineer",
                    "model_ref": "codex_cli",
                    "runtime": "codex_cli",
                    "tools": ["file", "terminal"],
                    "permission": "read_only",
                    "capabilities": ["architecture_review", "code_review", "implementation_planning"],
                    "risk_allowed": ["R0", "R1", "R2", "R3", "R4"],
                },
            ],
        }
    )


def test_capability_matrix_derives_profiles_from_agent_metadata() -> None:
    matrix = build_capability_matrix(_registry())

    assert matrix["deepseek-tui"].model_tier == "quick"
    assert matrix["deepseek-tui"].failure_policy == "one_retry_then_escalate"
    assert "tests" in matrix["deepseek-tui"].task_types
    assert matrix["deepseek-tui"].runtime == "deepseek_tui_cli"

    assert matrix["opencode"].runtime == "opencode_cli"
    assert "code_review" in matrix["opencode"].task_types
    assert "tests" not in matrix["opencode"].task_types

    assert matrix["claude"].model_tier == "strong"
    assert matrix["claude"].failure_policy == "timeout_then_switch_agent"
    assert "implementation" in matrix["claude"].task_types

    assert matrix["codex"].failure_policy == "timeout_then_switch_agent"
    assert "cannot_apply_changes" in matrix["codex"].weak_spots
    assert "architecture_review" in matrix["codex"].task_types


def test_preview_route_filters_by_task_type_and_risk() -> None:
    registry = _registry()

    tests_route = preview_route(registry, task_type="tests", risk_level="R1")
    assert tests_route["primary_agent"] == "deepseek-tui"
    assert tests_route["reason"] == "capability_and_risk_match"

    high_risk_tests = preview_route(registry, task_type="tests", risk_level="R4")
    assert high_risk_tests["primary_agent"] is None
    assert high_risk_tests["reason"] == "no_capability_match"

    architecture_route = preview_route(registry, task_type="architecture_review", risk_level="R3")
    assert architecture_route["primary_agent"] == "codex"


def test_preview_route_uses_effectiveness_to_deprioritize_unstable_agent() -> None:
    registry = _registry()
    effectiveness = {
        "deepseek-tui": {
            "run_count": 10,
            "handoff_count": 0,
            "effectiveness_score": 15.0,
            "timeout_rate": 80.0,
            "failed_rate": 10.0,
            "revision_needed_count": 1,
        },
        "claude": {
            "run_count": 5,
            "handoff_count": 0,
            "effectiveness_score": 92.0,
            "timeout_rate": 0.0,
            "failed_rate": 0.0,
            "revision_needed_count": 0,
        },
    }

    route = preview_route(
        registry,
        task_type="tests",
        risk_level="R1",
        effectiveness=effectiveness,
    )

    assert route["primary_agent"] == "claude"
    assert route["candidate_agents"][:2] == ["claude", "deepseek-tui"]
    assert route["reason"] == "adaptive_effectiveness_and_capability_match"


def test_model_tier_router_prefers_agent_strategy_chain() -> None:
    models = {
        "opencode_go_deepseek_pro": {"role": "experimental_deepseek_pool", "status": "experimental", "tokens_per_million": 0.2},
        "opencode_go_deepseek_flash": {"role": "experimental_cheap_task", "status": "experimental", "tokens_per_million": 0.05},
        "deepseek_flash": {"role": "primary_hermes", "status": "active", "tokens_per_million": 0.1},
        "deepseek_pro": {"role": "complex_reasoning", "status": "active", "tokens_per_million": 0.5},
    }

    decision = resolve_model_tier(
        _registry(),
        models,
        task_type="tests",
        risk_level="R1",
    )

    assert decision.agent_id == "deepseek-tui"
    assert decision.model_tier == "quick"
    assert decision.model_ref == "opencode_go_deepseek_flash"
    assert decision.fallback_chain == ("opencode_go_deepseek_flash", "deepseek_flash", "opencode_go_deepseek_pro", "deepseek_pro")
    assert decision.fallback_on == ("timeout", "rate_limited")
    assert decision.source == "agent_model_strategy"


def test_model_tier_router_uses_adaptive_route_when_no_agent_is_forced() -> None:
    models = {
        "opencode_go_deepseek_flash": {"role": "experimental_cheap_task", "status": "experimental", "tokens_per_million": 0.05},
        "deepseek_flash": {"role": "primary_hermes", "status": "active", "tokens_per_million": 0.1},
        "claude_sonnet": {"role": "complex_coding", "status": "active", "tokens_per_million": 3.0},
        "claude_opus": {"role": "primary_claude_code", "status": "active", "tokens_per_million": 15.0},
    }
    effectiveness = {
        "deepseek-tui": {"run_count": 8, "effectiveness_score": 10.0, "timeout_rate": 90.0, "failed_rate": 0.0},
        "claude": {"run_count": 4, "effectiveness_score": 95.0, "timeout_rate": 0.0, "failed_rate": 0.0},
    }

    decision = resolve_model_tier(
        _registry(),
        models,
        task_type="tests",
        risk_level="R1",
        effectiveness=effectiveness,
    )

    assert decision.agent_id == "claude"
    assert decision.model_ref == "claude_sonnet"


def test_model_tier_router_filters_deprecated_models_from_chain() -> None:
    models = {
        "opencode_go_deepseek_pro": {"role": "experimental_deepseek_pool", "status": "deprecated", "tokens_per_million": 0.2},
        "opencode_go_deepseek_flash": {"role": "experimental_cheap_task", "status": "experimental", "tokens_per_million": 0.05},
        "deepseek_flash": {"role": "primary_hermes", "status": "active", "tokens_per_million": 0.1},
        "deepseek_pro": {"role": "complex_reasoning", "status": "active", "tokens_per_million": 0.5},
    }

    decision = resolve_model_tier(
        _registry(),
        models,
        task_type="tests",
        risk_level="R1",
    )

    assert decision.model_ref == "opencode_go_deepseek_flash"
    assert decision.fallback_chain == ("opencode_go_deepseek_flash", "deepseek_flash", "deepseek_pro")


def test_model_tier_router_uses_tier_pool_without_agent_strategy() -> None:
    models = {
        "deepseek_flash": {"role": "primary_hermes", "status": "active", "tokens_per_million": 0.1},
        "opencode_go_deepseek_flash": {"role": "experimental_cheap_task", "status": "experimental", "tokens_per_million": 0.05},
        "claude_sonnet": {"role": "complex_coding", "status": "active", "tokens_per_million": 3.0},
        "claude_opus": {"role": "primary_claude_code", "status": "active", "tokens_per_million": 15.0},
    }

    decision = resolve_model_tier(
        _registry(),
        models,
        agent_id="codex",
        task_type="architecture_review",
        risk_level="R3",
    )

    assert decision.agent_id == "codex"
    assert decision.model_ref == "claude_sonnet"
    assert decision.source == "tier_pool"


def test_model_tier_router_returns_unresolved_without_candidate() -> None:
    decision = resolve_model_tier(
        _registry(),
        {},
        task_type="tests",
        risk_level="R4",
    )

    assert decision.model_ref is None
    assert decision.reason == "no_agent_for_task_type_and_risk"


def test_failure_reroute_switches_agent_on_timeout() -> None:
    models = {
        "opencode_go_deepseek_pro": {"role": "experimental_deepseek_pool", "status": "experimental", "tokens_per_million": 0.2},
        "opencode_go_deepseek_flash": {"role": "experimental_cheap_task", "status": "experimental", "tokens_per_million": 0.05},
        "deepseek_flash": {"role": "primary_hermes", "status": "active", "tokens_per_million": 0.1},
        "deepseek_pro": {"role": "complex_reasoning", "status": "active", "tokens_per_million": 0.5},
        "claude_sonnet": {"role": "complex_coding", "status": "active", "tokens_per_million": 3.0},
        "claude_opus": {"role": "primary_claude_code", "status": "active", "tokens_per_million": 15.0},
    }

    decision = decide_failure_reroute(
        _registry(),
        models,
        task_type="tests",
        risk_level="R1",
        failure="timeout",
        failed_agent_id="deepseek-tui",
        failed_model_ref="opencode_go_deepseek_flash",
    )

    assert decision.action == "switch_agent"
    assert decision.next_agent_id == "claude"
    assert decision.next_model_ref == "claude_sonnet"


def test_failure_reroute_uses_effectiveness_rank_for_agent_switch() -> None:
    models = {
        "opencode_go_deepseek_flash": {"role": "experimental_cheap_task", "status": "experimental", "tokens_per_million": 0.05},
        "deepseek_flash": {"role": "primary_hermes", "status": "active", "tokens_per_million": 0.1},
        "claude_sonnet": {"role": "complex_coding", "status": "active", "tokens_per_million": 3.0},
        "claude_opus": {"role": "primary_claude_code", "status": "active", "tokens_per_million": 15.0},
    }
    effectiveness = {
        "deepseek-tui": {"run_count": 4, "effectiveness_score": 20.0, "timeout_rate": 70.0, "failed_rate": 0.0},
        "claude": {"run_count": 4, "effectiveness_score": 90.0, "timeout_rate": 0.0, "failed_rate": 0.0},
    }

    decision = decide_failure_reroute(
        _registry(),
        models,
        task_type="tests",
        risk_level="R1",
        failure="timeout",
        failed_agent_id="opencode",
        failed_model_ref="opencode_go_deepseek_flash",
        effectiveness=effectiveness,
    )

    assert decision.action == "switch_agent"
    assert decision.next_agent_id == "claude"


def test_failure_reroute_switches_model_on_rate_limit() -> None:
    models = {
        "opencode_go_deepseek_pro": {"role": "experimental_deepseek_pool", "status": "experimental", "tokens_per_million": 0.2},
        "opencode_go_deepseek_flash": {"role": "experimental_cheap_task", "status": "experimental", "tokens_per_million": 0.05},
        "deepseek_pro": {"role": "complex_reasoning", "status": "active", "tokens_per_million": 0.5},
    }

    decision = decide_failure_reroute(
        _registry(),
        models,
        task_type="tests",
        risk_level="R1",
        failure="rate_limited",
        failed_agent_id="deepseek-tui",
        failed_model_ref="opencode_go_deepseek_flash",
    )

    assert decision.action == "switch_model"
    assert decision.next_agent_id == "deepseek-tui"
    assert decision.next_model_ref == "opencode_go_deepseek_pro"


def test_failure_reroute_retries_same_agent_for_revision_needed() -> None:
    models = {
        "opencode_go_deepseek_pro": {"role": "experimental_deepseek_pool", "status": "experimental", "tokens_per_million": 0.2},
        "opencode_go_deepseek_flash": {"role": "experimental_cheap_task", "status": "experimental", "tokens_per_million": 0.05},
    }

    decision = decide_failure_reroute(
        _registry(),
        models,
        task_type="tests",
        risk_level="R1",
        failure="revision_needed",
        failed_agent_id="deepseek-tui",
        failed_model_ref="opencode_go_deepseek_flash",
    )

    assert decision.action == "retry_same_agent"
    assert decision.next_agent_id == "deepseek-tui"
    assert decision.reason == "revision_needed_is_recoverable"


def test_failure_reroute_sends_auth_error_to_manual_review() -> None:
    decision = decide_failure_reroute(
        _registry(),
        {},
        task_type="tests",
        risk_level="R1",
        failure="auth_error",
        failed_agent_id="deepseek-tui",
        failed_model_ref="deepseek_pro",
    )

    assert decision.action == "manual_review"
    assert decision.requires_human_approval is True
