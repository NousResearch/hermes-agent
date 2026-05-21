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
        "nesta",
        "claude",
        "codex",
        "deepseek-tui",
        "pirlo",
        "intelligence",
        "ambrosini",
        "openclaw",
        "agent-tars",
        "hermes-internal",
        "kanban",
    }
    assert all(agent.can_delegate is False for agent in registry.agents.values())
    assert registry.get("codex").permission is PermissionMode.READ_ONLY
    assert registry.get("ambrosini").permission is PermissionMode.READ_ONLY
    assert not registry.get("deepseek-tui").allows_risk(RiskLevel.R4)


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
