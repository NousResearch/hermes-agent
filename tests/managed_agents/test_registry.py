import textwrap
from pathlib import Path

import pytest

from agent.managed_agents.registry import (
    AgentRegistry,
    AgentRegistryError,
    AgentStatus,
    PermissionMode,
    RiskLevel,
    load_agent_registry,
)


def write_agents_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "agents.yaml"
    path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    return path


def test_loads_agent_by_id_and_capability(tmp_path):
    path = write_agents_yaml(
        tmp_path,
        """
        version: "2026-05-21"
        agents:
          - agent_id: claude
            name: Claude Code
            role: lead_implementer
            tools: [file, terminal, git]
            permission: ask
            capabilities: [code_edit, test_run]
            risk_allowed: [R1, R2, R3]
          - agent_id: codex
            name: Codex
            role: principal_engineer
            tools: [file, terminal]
            permission: read_only
            capabilities: [code_review]
            risk_allowed: [R0, R1, R2, R3]
        """,
    )

    registry = load_agent_registry(path)

    claude = registry.get("claude")
    assert claude.agent_id == "claude"
    assert claude.permission is PermissionMode.ASK
    assert registry.find_by_capability("code_review") == [registry.get("codex")]


def test_can_delegate_defaults_to_false(tmp_path):
    path = write_agents_yaml(
        tmp_path,
        """
        version: "2026-05-21"
        agents:
          - agent_id: deepseek-tui
            name: DeepSeek TUI
            role: fast_worker
            tools: [file, terminal]
            permission: ask
            capabilities: [test_generation]
            risk_allowed: [R0, R1, R2]
        """,
    )

    registry = load_agent_registry(path)

    assert registry.get("deepseek-tui").can_delegate is False


def test_risk_allowed_uses_ordered_levels(tmp_path):
    path = write_agents_yaml(
        tmp_path,
        """
        version: "2026-05-21"
        agents:
          - agent_id: deepseek-tui
            name: DeepSeek TUI
            role: fast_worker
            tools: [file, terminal]
            permission: ask
            capabilities: [small_fix]
            risk_allowed: [R0, R1, R2]
        """,
    )

    registry = load_agent_registry(path)
    agent = registry.get("deepseek-tui")

    assert agent.allows_risk(RiskLevel.R2)
    assert not agent.allows_risk(RiskLevel.R4)


def test_read_only_agent_cannot_have_write_tools(tmp_path):
    path = write_agents_yaml(
        tmp_path,
        """
        version: "2026-05-21"
        agents:
          - agent_id: codex
            name: Codex
            role: principal_engineer
            tools: [file, terminal, git]
            permission: read_only
            capabilities: [code_review]
            risk_allowed: [R0, R1]
        """,
    )

    with pytest.raises(AgentRegistryError, match="read_only.*write-capable"):
        load_agent_registry(path)


def test_duplicate_agent_id_fails(tmp_path):
    path = write_agents_yaml(
        tmp_path,
        """
        version: "2026-05-21"
        agents:
          - agent_id: codex
            name: Codex
            role: reviewer
            tools: [file]
            permission: read_only
            capabilities: [code_review]
            risk_allowed: [R0]
          - agent_id: codex
            name: Duplicate Codex
            role: reviewer
            tools: [file]
            permission: read_only
            capabilities: [code_review]
            risk_allowed: [R0]
        """,
    )

    with pytest.raises(AgentRegistryError, match="Duplicate agent_id"):
        load_agent_registry(path)


def test_from_legacy_json_registry_normalizes_existing_config():
    registry = AgentRegistry.from_legacy_json(
        {
            "agents": {
                "codex": {
                    "id": "codex",
                    "display_name": "Codex",
                    "type": "code_reviewer",
                    "capabilities": ["code_review"],
                    "subagent_profile": {
                        "toolsets": ["file", "terminal"],
                        "permission_mode": "read_only",
                    },
                }
            }
        }
    )

    codex = registry.get("codex")
    assert codex.name == "Codex"
    assert codex.role == "code_reviewer"
    assert codex.permission is PermissionMode.READ_ONLY
    assert codex.risk_allowed == frozenset({RiskLevel.R0})
    assert codex.status is AgentStatus.ACTIVE

