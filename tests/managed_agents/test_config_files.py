from __future__ import annotations

from pathlib import Path

import yaml

from agent.managed_agents.policy import load_policy_engine
from agent.managed_agents.registry import PermissionMode, RiskLevel, load_agent_registry
from agent.managed_agents.router import load_managed_agent_router


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "managed_agents"


def test_managed_agents_config_loads_all_declared_runtime_agents():
    registry = load_agent_registry(CONFIG_DIR / "agents.yaml")

    assert set(registry.agents) == {
        "claude",
        "codex",
        "deepseek-tui",
        "pirlo",
        "intelligence",
        "ambrosini",
        "agent-tars",
        "hermes-internal",
    }
    assert all(agent.can_delegate is False for agent in registry.agents.values())
    assert registry.get("codex").permission is PermissionMode.READ_ONLY
    assert registry.get("ambrosini").permission is PermissionMode.READ_ONLY
    assert not registry.get("deepseek-tui").allows_risk(RiskLevel.R4)
    assert "visual_gui_automation" in registry.get("agent-tars").capabilities
    assert registry.resolve_agent_id("nesta") == "hermes-internal"
    assert registry.resolve_agent_id("技术翻译官") == "hermes-internal"
    assert registry.resolve_agent_id("低成本快工") == "deepseek-tui"
    assert registry.resolve_agent_id("kanban") is None
    assert registry.get("claude").runtime == "claude_code_cli"
    assert registry.get("codex").runtime == "codex_cli"


def test_managed_agents_aliases_and_user_facing_fields_are_consistent():
    registry = load_agent_registry(CONFIG_DIR / "agents.yaml")
    alias_map = registry.alias_map()

    assert alias_map["nesta"] == "hermes-internal"
    assert "kanban" not in registry.agents
    assert "nesta" not in registry.agents
    for agent in registry.agents.values():
        assert agent.name
        assert agent.aliases
        assert agent.role_summary
        assert agent.model_ref


def test_managed_agents_model_refs_are_declared_in_models_config():
    registry = load_agent_registry(CONFIG_DIR / "agents.yaml")
    models_path = Path("/Users/gu/.hermes/config/models.yaml")
    models = yaml.safe_load(models_path.read_text(encoding="utf-8"))["models"]

    expected = {
        "claude": "claude_opus",
        "deepseek-tui": "deepseek_pro",
        "intelligence": "deepseek_flash",
        "agent-tars": "tars_gpt54",
        "codex": "codex_cli",
    }
    for agent_id, model_ref in expected.items():
        assert registry.get(agent_id).model_ref == model_ref
    for agent in registry.agents.values():
        assert agent.model_ref in models
    assert registry.get("agent-tars").model_ref != "tars_glm"
    assert models["tars_gpt54"]["model"] == "gpt-5.4"


def test_codegraph_is_scoped_to_code_understanding_agents():
    registry = load_agent_registry(CONFIG_DIR / "agents.yaml")

    assert "mcp-codegraph" in registry.get("hermes-internal").tools
    assert "mcp-codegraph" in registry.get("codex").tools
    assert "mcp-codegraph" in registry.get("claude").tools
    assert "mcp-codegraph" not in registry.get("deepseek-tui").tools


def test_managed_agents_policy_config_loads_and_enforces_priority():
    policy = load_policy_engine(CONFIG_DIR / "policy.yaml")

    decision = policy.evaluate(
        {
            "task_id": "cfg-policy",
            "risk_level": "R4",
            "action_type": "delete_file",
            "user_override": "claude",
        }
    )

    assert decision.outcome == "deny"
    assert decision.winner == "safety"
    assert decision.requires_human_approval is True


def test_managed_agents_router_config_loads_embedded_routes():
    router = load_managed_agent_router(CONFIG_DIR / "agents.yaml")

    decision = router.route(
        {
            "task_id": "cfg-route",
            "task_category": "feature",
            "risk_level": "R2",
        }
    )

    assert decision.agents == ["claude"]
    assert decision.requires_review is True


def test_routes_yaml_references_registered_agents():
    registry = load_agent_registry(CONFIG_DIR / "agents.yaml")
    data = yaml.safe_load((CONFIG_DIR / "routes.yaml").read_text(encoding="utf-8"))

    referenced: set[str] = set()
    for route in data["routes"]:
        for key in ("owner_agent", "fallback_agent"):
            if route.get(key):
                referenced.add(route[key])
        for key in ("planner", "support_agents", "reviewers"):
            value = route.get(key) or []
            if isinstance(value, str):
                referenced.add(value)
            else:
                referenced.update(value)

    assert referenced <= set(registry.agents)
    assert "nesta" not in referenced
    assert "kanban" not in referenced


def test_embedded_routes_do_not_reference_retired_delegate_agents():
    data = yaml.safe_load((CONFIG_DIR / "agents.yaml").read_text(encoding="utf-8"))

    referenced: set[str] = set()
    for route in data["routing"]["rules"]:
        value = route.get("agents") or []
        referenced.update(value if isinstance(value, list) else [value])

    assert "nesta" not in referenced
    assert "kanban" not in referenced
