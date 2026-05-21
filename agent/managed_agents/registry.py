"""Managed Agents registry for structured agent metadata.

This module is intentionally small. It normalizes the existing legacy
``agent-registry.json`` shape into a typed registry, and it also loads the new
YAML registry format used by the managed-agents rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import json

import yaml


class AgentRegistryError(ValueError):
    """Raised when registry input is malformed or internally inconsistent."""


class RiskLevel(str, Enum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"

    @classmethod
    def from_raw(cls, value: Any) -> "RiskLevel":
        try:
            return cls(str(value))
        except Exception as exc:  # pragma: no cover - defensive
            raise AgentRegistryError(f"Invalid risk level: {value!r}") from exc


class PermissionMode(str, Enum):
    ASK = "ask"
    READ_ONLY = "read_only"

    @classmethod
    def from_raw(cls, value: Any) -> "PermissionMode":
        normalized = str(value or "").strip().lower()
        try:
            return cls(normalized)
        except Exception as exc:  # pragma: no cover - defensive
            raise AgentRegistryError(f"Invalid permission mode: {value!r}") from exc


class AgentStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"

    @classmethod
    def from_raw(cls, value: Any) -> "AgentStatus":
        normalized = str(value or "active").strip().lower()
        try:
            return cls(normalized)
        except Exception:
            return cls.ACTIVE


_WRITE_CAPABLE_TOOLS = {
    "git",
    "patch",
    "write_file",
    "filesystem_write",
    "edit",
    "apply_patch",
}


def _normalize_str_list(values: Iterable[Any] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(item).strip() for item in values if str(item).strip())


def _normalize_risk_levels(values: Iterable[Any] | None) -> frozenset[RiskLevel]:
    if not values:
        return frozenset({RiskLevel.R0})
    return frozenset(RiskLevel.from_raw(item) for item in values)


@dataclass(frozen=True, slots=True)
class AgentSpec:
    agent_id: str
    name: str
    role: str
    tools: tuple[str, ...] = ()
    permission: PermissionMode = PermissionMode.ASK
    can_delegate: bool = False
    capabilities: tuple[str, ...] = ()
    risk_allowed: frozenset[RiskLevel] = field(default_factory=lambda: frozenset({RiskLevel.R0}))
    status: AgentStatus = AgentStatus.ACTIVE
    source: str | None = None

    def allows_risk(self, risk_level: RiskLevel) -> bool:
        return risk_level in self.risk_allowed

    def has_write_tools(self) -> bool:
        return any(tool in _WRITE_CAPABLE_TOOLS for tool in self.tools)


@dataclass(slots=True)
class AgentRegistry:
    version: str
    agents: dict[str, AgentSpec]
    source_path: Path | None = None

    def get(self, agent_id: str) -> AgentSpec:
        try:
            return self.agents[agent_id]
        except KeyError as exc:
            raise AgentRegistryError(f"Unknown agent_id: {agent_id}") from exc

    def find_by_capability(self, capability: str) -> list[AgentSpec]:
        return [agent for agent in self.agents.values() if capability in agent.capabilities]

    @classmethod
    def from_yaml(cls, data: Mapping[str, Any], *, source_path: Path | None = None) -> "AgentRegistry":
        if not isinstance(data, Mapping):
            raise AgentRegistryError("Registry payload must be a mapping")
        version = str(data.get("version", "")).strip()
        if not version:
            raise AgentRegistryError("Registry is missing version")

        raw_agents = data.get("agents")
        if not isinstance(raw_agents, list):
            raise AgentRegistryError("Registry agents must be a list")

        agents: dict[str, AgentSpec] = {}
        for raw in raw_agents:
            if not isinstance(raw, Mapping):
                raise AgentRegistryError("Each agent entry must be a mapping")
            agent = _parse_yaml_agent(raw, source_path=source_path)
            if agent.agent_id in agents:
                raise AgentRegistryError(f"Duplicate agent_id: {agent.agent_id}")
            if agent.permission is PermissionMode.READ_ONLY and agent.has_write_tools():
                raise AgentRegistryError(
                    f"read_only agent {agent.agent_id} cannot declare write-capable tools"
                )
            agents[agent.agent_id] = agent

        return cls(version=version, agents=agents, source_path=source_path)

    @classmethod
    def from_legacy_json(
        cls,
        data: Mapping[str, Any],
        *,
        source_path: Path | None = None,
    ) -> "AgentRegistry":
        if not isinstance(data, Mapping):
            raise AgentRegistryError("Registry payload must be a mapping")

        version = str(data.get("schema_version", "1.0")).strip() or "1.0"
        raw_agents = data.get("agents")
        if not isinstance(raw_agents, Mapping):
            raise AgentRegistryError("Legacy registry agents must be a mapping")

        agents: dict[str, AgentSpec] = {}
        for agent_id, raw in raw_agents.items():
            if not isinstance(raw, Mapping):
                raise AgentRegistryError(f"Agent {agent_id!r} must be a mapping")
            agent = _parse_legacy_agent(agent_id, raw, source_path=source_path)
            if agent.agent_id in agents:
                raise AgentRegistryError(f"Duplicate agent_id: {agent.agent_id}")
            if agent.permission is PermissionMode.READ_ONLY and agent.has_write_tools():
                raise AgentRegistryError(
                    f"read_only agent {agent.agent_id} cannot declare write-capable tools"
                )
            agents[agent.agent_id] = agent

        return cls(version=version, agents=agents, source_path=source_path)


def load_agent_registry(path: str | Path) -> AgentRegistry:
    registry_path = Path(path)
    raw_text = registry_path.read_text(encoding="utf-8")
    if registry_path.suffix.lower() in {".json", ".jsonc"}:
        data = json.loads(raw_text)
        return AgentRegistry.from_legacy_json(data, source_path=registry_path)
    data = yaml.safe_load(raw_text)
    if isinstance(data, Mapping) and "schema_version" in data and "version" not in data:
        return AgentRegistry.from_legacy_json(data, source_path=registry_path)
    return AgentRegistry.from_yaml(data, source_path=registry_path)


def _parse_yaml_agent(raw: Mapping[str, Any], *, source_path: Path | None) -> AgentSpec:
    agent_id = str(raw.get("agent_id") or "").strip()
    if not agent_id:
        raise AgentRegistryError("Agent entry is missing agent_id")
    name = str(raw.get("name") or agent_id).strip()
    role = str(raw.get("role") or "").strip()
    if not role:
        raise AgentRegistryError(f"Agent {agent_id!r} is missing role")
    tools = _normalize_str_list(raw.get("tools"))
    permission = PermissionMode.from_raw(raw.get("permission"))
    can_delegate = bool(raw.get("can_delegate", False))
    capabilities = _normalize_str_list(raw.get("capabilities"))
    risk_allowed = _normalize_risk_levels(raw.get("risk_allowed"))
    status = AgentStatus.from_raw(raw.get("status"))
    return AgentSpec(
        agent_id=agent_id,
        name=name,
        role=role,
        tools=tools,
        permission=permission,
        can_delegate=can_delegate,
        capabilities=capabilities,
        risk_allowed=risk_allowed,
        status=status,
        source=str(source_path) if source_path else None,
    )


def _parse_legacy_agent(agent_id: str, raw: Mapping[str, Any], *, source_path: Path | None) -> AgentSpec:
    profile = raw.get("subagent_profile") or {}
    if not isinstance(profile, Mapping):
        raise AgentRegistryError(f"Agent {agent_id!r} subagent_profile must be a mapping")

    name = str(raw.get("display_name") or raw.get("name") or agent_id).strip()
    role = str(raw.get("type") or raw.get("role") or "").strip()
    if not role:
        raise AgentRegistryError(f"Agent {agent_id!r} is missing type/role")

    tools = _normalize_str_list(profile.get("toolsets"))
    permission = PermissionMode.from_raw(profile.get("permission_mode"))
    can_delegate = bool(profile.get("can_delegate", False))
    capabilities = _normalize_str_list(raw.get("capabilities"))
    risk_allowed = _normalize_risk_levels(raw.get("risk_allowed"))
    status = AgentStatus.from_raw(raw.get("status"))

    return AgentSpec(
        agent_id=agent_id,
        name=name,
        role=role,
        tools=tools,
        permission=permission,
        can_delegate=can_delegate,
        capabilities=capabilities,
        risk_allowed=risk_allowed,
        status=status,
        source=str(source_path) if source_path else None,
    )
