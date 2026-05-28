from __future__ import annotations

from pathlib import Path

import yaml

from agent.managed_agents.policy import load_policy_engine
from agent.managed_agents.registry import PermissionMode, RiskLevel, load_agent_registry
from agent.managed_agents.runtime_mirror import build_runtime_registry
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
        "designer",
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
        assert agent.skills


def test_managed_agents_model_refs_are_declared_in_models_config():
    registry = load_agent_registry(CONFIG_DIR / "agents.yaml")
    models_path = Path("/Users/gu/.hermes/config/models.yaml")
    models = yaml.safe_load(models_path.read_text(encoding="utf-8"))["models"]

    expected = {
        "claude": "claude_opus",
        "deepseek-tui": "opencode_go_deepseek_pro",
        "intelligence": "opencode_go_qwen37_max",
        "pirlo": "opencode_go_kimi26",
        "designer": "opencode_go_kimi26",
        "ambrosini": "opencode_go_glm51",
        "hermes-internal": "opencode_go_glm51",
        "agent-tars": "opencode_go_mimo25_pro",
        "codex": "codex_cli",
    }
    for agent_id, model_ref in expected.items():
        assert registry.get(agent_id).model_ref == model_ref
    for agent in registry.agents.values():
        assert agent.model_ref in models
        strategy = agent.model_strategy or {}
        for chain_ref in strategy.get("chain") or []:
            assert chain_ref in models, (agent.id, chain_ref)
    assert registry.get("agent-tars").model_ref != "tars_glm"
    assert registry.get("agent-tars").model_strategy["chain"][-1] == "tars_gpt54"
    assert models["tars_gpt54"]["model"] == "gpt-5.4"


def test_managed_agent_model_strategies_prefer_subscription_pool_before_api_fallback():
    registry = load_agent_registry(CONFIG_DIR / "agents.yaml")

    subscription_primary = {
        "hermes-internal",
        "deepseek-tui",
        "intelligence",
        "pirlo",
        "designer",
        "agent-tars",
        "ambrosini",
    }
    api_fallbacks = {"deepseek_pro", "deepseek_flash", "tars_gpt54"}

    for agent_id in subscription_primary:
        agent = registry.get(agent_id)
        strategy = agent.model_strategy or {}
        chain = strategy.get("chain") or []
        assert agent.model_ref.startswith("opencode_go_"), agent_id
        assert strategy.get("primary") == agent.model_ref, agent_id
        assert chain[0] == agent.model_ref, agent_id
        assert any(ref in api_fallbacks for ref in chain[1:]), agent_id

    assert registry.get("claude").model_strategy["mode"] == "external"
    assert registry.get("codex").model_strategy["mode"] == "external"


def test_codegraph_is_scoped_to_code_understanding_agents():
    registry = load_agent_registry(CONFIG_DIR / "agents.yaml")

    assert "mcp-codegraph" in registry.get("hermes-internal").tools
    assert "mcp-codegraph" in registry.get("codex").tools
    assert "mcp-codegraph" in registry.get("claude").tools
    assert "mcp-codegraph" in registry.get("ambrosini").tools
    assert "mcp-codegraph" not in registry.get("deepseek-tui").tools
    assert "terminal" not in registry.get("ambrosini").tools


def test_managed_agents_skill_whitelists_are_declared():
    registry = load_agent_registry(CONFIG_DIR / "agents.yaml")

    assert "hermes-subagent-delegation" in registry.get("hermes-internal").skills
    assert "github-pr-workflow" in registry.get("claude").skills
    assert "design-md" in registry.get("claude").skills
    assert "comfyui" in registry.get("claude").skills
    assert "debugging-hermes-tui-commands" in registry.get("deepseek-tui").skills
    assert "codex-superpowers" in registry.get("codex").skills
    assert "playwright-mcp" not in registry.get("codex").skills
    assert registry.get("intelligence").skills == ("competitive-intelligence",)
    assert "claude-design" in registry.get("pirlo").skills
    assert "html-anything" not in registry.get("pirlo").skills
    assert "comfyui" not in registry.get("pirlo").skills
    assert "html-anything" in registry.get("designer").skills
    assert "design-taste-frontend" in registry.get("designer").skills
    assert "gpt-taste" in registry.get("designer").skills
    assert "brandkit" in registry.get("designer").skills
    assert "image_gen" in registry.get("designer").tools
    assert "terminal" in registry.get("designer").tools
    assert "browser-automation-for-blocked-sites" in registry.get("agent-tars").skills
    assert "playwright-mcp" in registry.get("agent-tars").skills
    assert "libreoffice-cli" in registry.get("agent-tars").skills
    assert "github-code-review" in registry.get("ambrosini").skills
    assert "playwright-mcp" not in registry.get("ambrosini").skills
    assert "kanban" not in registry.agents


def test_runtime_agent_registry_profiles_match_managed_agents_config():
    registry = load_agent_registry(CONFIG_DIR / "agents.yaml")
    runtime_path = Path("/Users/gu/.hermes/config/agent-registry.json")
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))

    for agent_id, agent in registry.agents.items():
        profile = runtime["agents"][agent_id]["subagent_profile"]
        assert profile["toolsets"] == list(agent.tools), agent_id
        assert profile["skills"] == list(agent.skills), agent_id
        assert profile.get("model_strategy") == agent.model_strategy, agent_id


def test_runtime_agent_registry_is_generated_from_managed_agents_config():
    runtime_path = Path("/Users/gu/.hermes/config/agent-registry.json")
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    expected = build_runtime_registry(CONFIG_DIR / "agents.yaml")

    assert runtime == expected


def test_capability_routes_are_declared_on_target_agents():
    registry = load_agent_registry(CONFIG_DIR / "agents.yaml")
    data = yaml.safe_load((CONFIG_DIR / "agents.yaml").read_text(encoding="utf-8"))

    capability_routes = data["routing"]["capability_routes"]
    for capability, agent_id in capability_routes.items():
        assert agent_id in registry.agents, capability
        assert capability in registry.get(agent_id).capabilities, (capability, agent_id)


def test_known_tool_bound_skills_are_assigned_to_executable_agents():
    registry = load_agent_registry(CONFIG_DIR / "agents.yaml")

    assert "comfyui" in registry.get("claude").skills
    assert "terminal" in registry.get("claude").tools
    assert "comfyui" not in registry.get("pirlo").skills
    assert "terminal" not in registry.get("pirlo").tools

    assert "html-anything" in registry.get("designer").skills
    assert "terminal" in registry.get("designer").tools
    assert "image_gen" in registry.get("designer").tools

    assert "playwright-mcp" in registry.get("agent-tars").skills
    assert "browser" in registry.get("agent-tars").tools
    assert "playwright-mcp" not in registry.get("codex").skills
    assert "playwright-mcp" not in registry.get("ambrosini").skills

    assert "libreoffice-cli" in registry.get("agent-tars").skills
    assert "terminal" in registry.get("agent-tars").tools
    assert "file" in registry.get("agent-tars").tools


def test_active_skill_frontmatter_uses_canonical_agent_ids():
    valid_agents = {
        "hermes",
        "hermes-internal",
        "claude",
        "deepseek-tui",
        "codex",
        "intelligence",
        "pirlo",
        "designer",
        "agent-tars",
        "ambrosini",
    }
    skills_root = Path("/Users/gu/.hermes/skills")

    for skill_path in skills_root.rglob("SKILL.md"):
        if ".archive" in skill_path.parts:
            continue
        text = skill_path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        end = text.find("\n---", 3)
        assert end != -1, f"{skill_path} is missing closing frontmatter delimiter"
        metadata = yaml.safe_load(text[3:end]) or {}
        agents = metadata.get("agents") or []
        if isinstance(agents, str):
            agents = [agents]
        assert "nesta" not in agents, str(skill_path)
        assert "openclaw" not in agents, str(skill_path)
        assert set(agents) <= valid_agents, f"{skill_path}: {agents}"


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
