import textwrap
from pathlib import Path

import pytest

from agent.managed_agents.router import (
    ManagedAgentRouterError,
    load_managed_agent_router,
)


def write_managed_agents_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "managed-agents.yaml"
    path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    return path


def write_router_config(tmp_path: Path) -> Path:
    return write_managed_agents_yaml(
        tmp_path,
        """
        version: "2026-05-21"
        agents:
          - agent_id: nesta
            name: Nesta
            role: technical_analyst
            tools: [file, terminal]
            permission: ask
            capabilities: [technical_analysis, code_review]
            risk_allowed: [R0, R1, R2]
          - agent_id: claude
            name: Claude Code
            role: lead_implementer
            tools: [file, terminal, git]
            permission: ask
            capabilities: [code_edit, test_run, refactor]
            risk_allowed: [R1, R2, R3]
          - agent_id: codex
            name: Codex
            role: principal_engineer
            tools: [file, terminal]
            permission: read_only
            capabilities: [architecture_review, code_review, implementation_planning]
            risk_allowed: [R0, R1, R2, R3]
          - agent_id: deepseek-tui
            name: DeepSeek TUI
            role: fast_worker
            tools: [file, terminal]
            permission: ask
            capabilities: [small_fix, test_generation, bug_reproduction]
            risk_allowed: [R0, R1, R2]
        routing:
          default_route:
            mode: single_agent
            agents: [claude]
            reason: normal_code_change
          fallback_route:
            mode: review_only
            agents: [codex]
            reason: unresolved_route_requires_review
          rules:
            - id: high_risk_pipeline
              when:
                risk_level: R4
              mode: pipeline
              agents: [codex, claude, nesta]
              require_gate: review
              reason: risk_escalation
            - id: feature_implementation
              when:
                task_category: feature
              mode: single_agent
              agents: [claude]
              reason: default_feature_implementation
            - id: test_slice
              when:
                task_category: tests
              mode: single_agent
              agents: [deepseek-tui]
              reason: targeted_test_generation
            - id: architecture_review
              when:
                task_category: architecture_review
              mode: single_agent
              agents: [codex]
              reason: design_review
        """,
    )


def test_loads_route_rules_from_managed_agents_config_shape(tmp_path):
    router = load_managed_agent_router(write_router_config(tmp_path))

    decision = router.route(
        {
            "task_id": "T-route-load",
            "task_category": "tests",
            "risk_level": "R1",
        }
    )

    assert decision.mode == "single_agent"
    assert decision.agents == ["deepseek-tui"]
    assert decision.matched_rule == "test_slice"
    assert decision.routing_basis == ["rule:test_slice", "risk:R1"]


def test_default_routes_normal_code_change_to_implementer(tmp_path):
    router = load_managed_agent_router(write_router_config(tmp_path))

    decision = router.route(
        {
            "task_id": "T-feature-default",
            "task_category": "feature",
            "risk_level": "R2",
            "summary": "Add registry-backed router tests",
        }
    )

    assert decision.mode == "single_agent"
    assert decision.agents == ["claude"]
    assert decision.reason == "default_feature_implementation"
    assert decision.requires_review is True
    assert decision.requires_human_approval is False
    assert decision.routing_basis == ["rule:feature_implementation", "risk:R2"]


def test_user_override_wins_when_agent_exists_and_allows_risk(tmp_path):
    router = load_managed_agent_router(write_router_config(tmp_path))

    decision = router.route(
        {
            "task_id": "T-user-override",
            "task_category": "feature",
            "risk_level": "R2",
            "user_override": "deepseek-tui",
        }
    )

    assert decision.mode == "single_agent"
    assert decision.agents == ["deepseek-tui"]
    assert decision.matched_rule == "user_override"
    assert decision.reason == "user_explicit_instruction"
    assert decision.routing_basis == [
        "user_override:deepseek-tui",
        "rule:feature_implementation",
        "risk:R2",
    ]


def test_high_risk_escalates_to_pipeline_even_with_user_override(tmp_path):
    router = load_managed_agent_router(write_router_config(tmp_path))

    decision = router.route(
        {
            "task_id": "T-risk-escalation",
            "task_category": "feature",
            "risk_level": "R4",
            "user_override": "claude",
        }
    )

    assert decision.mode == "pipeline"
    assert decision.agents == ["codex", "claude", "nesta"]
    assert decision.matched_rule == "high_risk_pipeline"
    assert decision.require_gate == "review"
    assert decision.requires_plan is True
    assert decision.requires_review is True
    assert decision.requires_human_approval is True
    assert decision.routing_basis == [
        "risk_escalation:R4",
        "user_override:claude",
        "rule:high_risk_pipeline",
    ]


@pytest.mark.parametrize(
    ("task", "expected_basis"),
    [
        (
            {
                "task_id": "T-unknown-category",
                "task_category": "unknown",
                "risk_level": "R1",
            },
            ["fallback:unmatched_route", "risk:R1"],
        ),
        (
            {
                "task_id": "T-unknown-override",
                "task_category": "feature",
                "risk_level": "R1",
                "user_override": "missing-agent",
            },
            [
                "fallback:unknown_user_override",
                "user_override:missing-agent",
                "rule:feature_implementation",
                "risk:R1",
            ],
        ),
        (
            {
                "task_id": "T-risk-mismatch-override",
                "task_category": "feature",
                "risk_level": "R3",
                "user_override": "deepseek-tui",
            },
            [
                "fallback:user_override_risk_mismatch",
                "user_override:deepseek-tui",
                "rule:feature_implementation",
                "risk:R3",
            ],
        ),
    ],
)
def test_falls_back_to_review_only_when_route_cannot_be_resolved(
    tmp_path, task, expected_basis
):
    router = load_managed_agent_router(write_router_config(tmp_path))

    decision = router.route(task)

    assert decision.mode == "review_only"
    assert decision.agents == ["codex"]
    assert decision.reason == "unresolved_route_requires_review"
    assert decision.fallback_used in {
        "unmatched_route",
        "unknown_user_override",
        "user_override_risk_mismatch",
    }
    assert decision.routing_basis == expected_basis


def test_structured_decision_fields_feed_task_card_plan_and_routing_basis(tmp_path):
    router = load_managed_agent_router(write_router_config(tmp_path))

    decision = router.route(
        {
            "task_id": "T-task-card-fields",
            "task_category": "feature",
            "risk_level": "R2",
        }
    )

    assert decision.to_execution_plan() == {
        "mode": "single_agent",
        "agents": ["claude"],
        "delegation_reason": "default_feature_implementation",
        "require_gate": None,
    }
    assert decision.to_task_card_updates() == {
        "execution_plan": {
            "mode": "single_agent",
            "agents": ["claude"],
            "delegation_reason": "default_feature_implementation",
            "require_gate": None,
        },
        "routing_basis": ["rule:feature_implementation", "risk:R2"],
        "fallback_used": None,
    }
    assert decision.to_dict() == {
        "task_id": "T-task-card-fields",
        "mode": "single_agent",
        "agents": ["claude"],
        "reason": "default_feature_implementation",
        "matched_rule": "feature_implementation",
        "routing_basis": ["rule:feature_implementation", "risk:R2"],
        "fallback_used": None,
        "risk_level": "R2",
        "require_gate": None,
        "requires_plan": False,
        "requires_review": True,
        "requires_human_approval": False,
    }


def test_route_rules_must_resolve_to_known_agents(tmp_path):
    path = write_managed_agents_yaml(
        tmp_path,
        """
        version: "2026-05-21"
        agents:
          - agent_id: codex
            name: Codex
            role: principal_engineer
            tools: [file, terminal]
            permission: read_only
            capabilities: [code_review]
            risk_allowed: [R0, R1, R2]
        routing:
          default_route:
            mode: single_agent
            agents: [missing-agent]
            reason: broken_default
          rules: []
        """,
    )

    with pytest.raises(ManagedAgentRouterError, match="missing-agent"):
        load_managed_agent_router(path)
