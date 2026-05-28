"""Tests for AgentRouter — registry-backed routing (Phase A)."""

import json
import os
from pathlib import Path

import pytest

from agent.agent_router import (
    AgentRouter,
    _load_agent_registry,
    _get_agents_map,
    _get_routing_rules,
    TASK_CATEGORY_REQUIRED_CAPABILITY,
)


def _write_test_registry(hermes_home: Path) -> Path:
    """Write a minimal test agent-registry.json for router tests."""
    config_dir = hermes_home / "config"
    config_dir.mkdir(exist_ok=True)
    registry = {
        "schema_version": "1.0",
        "agents": {
            "kimi": {
                "id": "kimi",
                "type": "researcher",
                "status": "deprecated",
                "capabilities": ["web_search", "file_reading"],
            },
            "claude": {
                "id": "claude",
                "display_name": "Claude 主程执行官",
                "aliases": ["主程", "主执行人"],
                "type": "file_executor",
                "capabilities": ["file_modification", "script_execution", "system_operations"],
            },
            "codex": {
                "id": "codex",
                "type": "code_executor",
                "capabilities": ["code_review", "implementation_planning", "file_modification"],
            },
            "agent-tars": {
                "id": "agent-tars",
                "type": "desktop_operator",
                "capabilities": ["desktop_control", "app_operation", "screenshot"],
            },
            "hermes-internal": {
                "id": "hermes-internal",
                "display_name": "Hermes 技术翻译官",
                "aliases": ["技术翻译官", "技术中间层", "nesta"],
                "type": "analyst",
                "capabilities": ["analysis", "decision_making", "creative_planning", "web_search", "file_reading", "technical_decomposition"],
            },
            "designer": {
                "id": "designer",
                "display_name": "Designer 视觉设计师",
                "aliases": ["视觉设计师", "设计师", "designer"],
                "type": "visual_designer",
                "capabilities": ["creative_direction", "visual_design", "web_visual_design", "ui_design"],
            },
            "intelligence": {
                "id": "intelligence",
                "type": "researcher",
                "capabilities": ["web_research", "file_reading_analysis"],
            },
            "deepseek-worker": {
                "id": "deepseek-worker",
                "type": "persistent_worker",
                "capabilities": ["background_execution", "file_operations"],
            },
            "deepseek-tui": {
                "id": "deepseek-tui",
                "display_name": "DeepSeek 低成本快工",
                "aliases": ["低成本快工", "快工", "deepseek"],
                "type": "fast_worker",
                "capabilities": ["small_fix", "test_generation"],
            },
            "hybrid-worker": {
                "id": "hybrid-worker",
                "type": "hybrid_worker",
                "capabilities": ["cap_a", "cap_b"],
            },
            "cap-c-worker": {
                "id": "cap-c-worker",
                "type": "cap_c_worker",
                "capabilities": ["cap_c"],
            },
        },
        "routing_rules": {
            "web_research": "intelligence",
            "file_reading_analysis": "intelligence",
            "file_modification": "claude",
            "script_execution": "claude",
            "code_review": "codex",
            "desktop_control": "agent-tars",
            "strategy_decision": "hermes-internal",
            "creative_direction": "designer",
            "technical_decomposition": "hermes-internal",
            "test_generation": "deepseek-tui",
        },
    }
    path = config_dir / "agent-registry.json"
    path.write_text(json.dumps(registry))
    return path


@pytest.fixture
def hermetic_registry(_hermetic_environment):
    """Hermetic env with a test registry written to HERMES_HOME/config/."""
    hermes_home = Path(os.environ["HERMES_HOME"])
    _write_test_registry(hermes_home)
    yield hermes_home


class TestRegistryLoading:

    def test_load_registry_returns_dict(self, hermetic_registry):
        registry = _load_agent_registry()
        assert isinstance(registry, dict)
        assert "agents" in registry
        assert "routing_rules" in registry

    def test_get_agents_map(self):
        registry = {"agents": {"kimi": {"id": "kimi"}}, "routing_rules": {}}
        agents = _get_agents_map(registry)
        assert "kimi" in agents

    def test_get_routing_rules(self):
        registry = {"agents": {}, "routing_rules": {"web_research": "kimi"}}
        rules = _get_routing_rules(registry)
        assert rules["web_research"] == "kimi"


class TestRouting:

    def test_route_self_execute_category(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route("architecture_review")
        assert decision.mode == "self_execute"
        assert decision.agents == []

    def test_route_research_uses_registry(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route("research")
        assert decision.mode == "single_agent"
        assert "intelligence" in decision.agents

    def test_route_unknown_category_self_execute(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route("nonexistent_category")
        assert decision.mode == "self_execute"

    def test_user_override_valid_agent(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route("research", user_agent_override="claude")
        assert decision.mode == "single_agent"
        assert decision.agents == ["claude"]
        assert "user_override" in decision.routing_basis

    def test_user_override_resolves_chinese_alias(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route("research", user_agent_override="技术翻译官")
        assert decision.mode == "single_agent"
        assert decision.agents == ["hermes-internal"]
        assert "user_override" in decision.routing_basis

    def test_user_override_resolves_legacy_nesta_alias(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route("research", user_agent_override="nesta")
        assert decision.mode == "single_agent"
        assert decision.agents == ["hermes-internal"]

    def test_user_override_resolves_low_cost_worker_alias(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route("research", user_agent_override="低成本快工")
        assert decision.mode == "single_agent"
        assert decision.agents == ["deepseek-tui"]

    def test_user_override_unknown_agent(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route("research", user_agent_override="nonexistent")
        assert decision.mode == "review_only"
        assert decision.agents == []
        assert "user_override_unknown" in decision.routing_basis

    def test_high_risk_escalation(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route("architecture_review", risk_level="high")
        assert decision.mode == "pipeline"
        assert "risk_level" in decision.routing_basis
        assert decision.risk_level == "high"

    def test_routing_decision_to_dict(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route("document")
        d = decision.to_dict()
        assert "mode" in d
        assert "agents" in d
        assert "delegation_reason" in d

    def test_get_available_agents(self, hermetic_registry):
        router = AgentRouter()
        agents = router.get_available_agents()
        assert "kimi" in agents
        assert "claude" in agents
        assert "codex" in agents
        assert "agent-tars" in agents

    def test_get_agent_info(self, hermetic_registry):
        router = AgentRouter()
        info = router.get_agent_info("kimi")
        assert info is not None
        assert info.get("id") == "kimi"

    def test_get_agent_info_resolves_alias(self, hermetic_registry):
        router = AgentRouter()
        info = router.get_agent_info("技术中间层")
        assert info is not None
        assert info.get("id") == "hermes-internal"

    def test_get_agent_info_unknown(self, hermetic_registry):
        router = AgentRouter()
        info = router.get_agent_info("nonexistent")
        assert info is None

    def test_required_capabilities_expand_agents(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route(
            "research",
            required_capabilities=["system_operations"],
        )
        # system_operations is in claude's capabilities
        assert "claude" in decision.agents
        assert decision.mode == "pipeline"

    def test_registry_missing_route(self, hermetic_registry):
        """visual_design: capability creative_direction maps to designer, should work."""
        router = AgentRouter()
        decision = router.route("visual_design")
        assert "designer" in decision.agents

    def test_code_review_routes_to_codex(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route("other", required_capabilities=["code_review"])
        assert decision.mode == "single_agent"
        assert decision.agents == ["codex"]

    def test_desktop_control_route_available(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route(
            "other",
            user_agent_override="agent-tars",
            required_capabilities=["desktop_control"],
        )
        assert decision.agents == ["agent-tars"]

    def test_user_override_is_not_modified_by_missing_capabilities(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route(
            "research",
            user_agent_override="claude",
            required_capabilities=["web_search"],
        )
        assert decision.mode == "single_agent"
        assert decision.agents == ["claude"]
        assert "capability_gap_for_user_override" in decision.overrides

    def test_high_risk_promotes_user_override_to_pipeline_without_dropping_agent(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route(
            "research",
            user_agent_override="claude",
            risk_level="HIGH ",
        )
        assert decision.mode == "pipeline"
        assert "claude" in decision.agents
        assert "hermes-internal" in decision.agents
        assert decision.risk_level == "high"

    def test_single_agent_with_missing_route_returns_explicit_error(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route("code_analysis")
        assert decision.mode == "single_agent"
        assert decision.agents

        # Remove route in the live router instance to exercise the invariant.
        router._routing_rules["file_reading_analysis"] = "missing-agent"
        decision = router.route("code_analysis")
        assert decision.mode == "review_only"
        assert decision.agents == []

    def test_required_capabilities_can_be_covered_by_agent_combination(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route(
            "other",
            required_capabilities=["cap_a", "cap_c"],
        )
        assert decision.mode == "pipeline"
        assert "hybrid-worker" in decision.agents
        assert "cap-c-worker" in decision.agents

    def test_invalid_risk_level_is_normalized(self, hermetic_registry):
        router = AgentRouter()
        decision = router.route("architecture_review", risk_level="Highish")
        assert decision.risk_level == "low"
        assert "invalid_risk_level" in decision.routing_basis

    def test_fallback_for_initializes_registry_when_called_directly(self, hermetic_registry):
        router = AgentRouter()
        assert "claude" in router._fallback_for("single_agent", ["claude"])

    def test_validate_delegation_prompt_covers_codex(self):
        issues = AgentRouter.validate_delegation_prompt("codex", "review")
        assert issues
        assert "Codex CLI" in issues[0]
        assert AgentRouter.validate_delegation_prompt(
            "codex",
            "Review /tmp/example.py for routing bugs without editing files",
        ) == []

    def test_validate_delegation_prompt_covers_tars_and_kimi(self):
        assert AgentRouter.validate_delegation_prompt("agent-tars", "do it")
        assert AgentRouter.validate_delegation_prompt("agent-tars", "截图当前桌面") == []
        assert AgentRouter.validate_delegation_prompt("kimi", "搜索")

    def test_task_category_required_capability_complete(self):
        from agent.agent_router import DEFAULT_ROUTES
        for cat in DEFAULT_ROUTES:
            assert cat in TASK_CATEGORY_REQUIRED_CAPABILITY, \
                f"Missing capability mapping for {cat}"
