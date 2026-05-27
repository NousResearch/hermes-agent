"""Build the legacy runtime agent registry mirror from managed agents YAML.

``configs/managed_agents/agents.yaml`` is the engineering source of truth.
Some older runtime paths still read ``~/.hermes/config/agent-registry.json``;
this module creates that JSON mirror deterministically so the two files do not
drift by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .registry import AgentSpec, load_agent_registry


SCHEMA_VERSION = "1.0"
READ_ONLY_BLOCKED_TOOLS = ("delegate_task", "send_message", "memory", "write_file", "patch")
ASK_BLOCKED_TOOLS = ("delegate_task", "send_message", "memory", "clarify")


def build_runtime_registry(source_path: str | Path) -> dict[str, Any]:
    """Return the legacy ``agent-registry.json`` payload for ``source_path``."""
    path = Path(source_path)
    registry = load_agent_registry(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    routing = raw.get("routing") if isinstance(raw, Mapping) else {}
    capability_routes = (routing or {}).get("capability_routes") or {}
    _validate_capability_routes(capability_routes, registry.agents)

    return {
        "schema_version": SCHEMA_VERSION,
        "source_of_truth": str(path),
        "generated_from": "configs/managed_agents/agents.yaml",
        "agents": {
            agent_id: _runtime_agent_payload(agent)
            for agent_id, agent in registry.agents.items()
        },
        "routing_rules": dict(capability_routes),
    }


def write_runtime_registry(source_path: str | Path, output_path: str | Path) -> None:
    payload = build_runtime_registry(source_path)
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _runtime_agent_payload(agent: AgentSpec) -> dict[str, Any]:
    profile = {
        "model": "default",
        "toolsets": list(agent.tools),
        "blocked_tools": _blocked_tools_for(agent),
        "permission_mode": agent.permission.value,
        "isolation": "readonly" if agent.permission.value == "read_only" else "shared",
        "allow_background": False,
        "required_mcp_servers": _required_mcp_servers(agent),
        "model_ref": agent.model_ref,
        "model_strategy": dict(agent.model_strategy),
        "skills": list(agent.skills),
    }
    if agent.runtime:
        profile["runtime"] = agent.runtime

    payload = {
        "id": agent.agent_id,
        "type": agent.role,
        "display_name": agent.name,
        "soul": _default_soul(agent),
        "capabilities": list(agent.capabilities),
        "risk_allowed": [risk.value for risk in sorted(agent.risk_allowed, key=lambda item: item.value)],
        "subagent_profile": profile,
        "aliases": list(agent.aliases),
        "role_summary": agent.role_summary,
        "model_ref": agent.model_ref,
        "model_strategy": dict(agent.model_strategy),
    }
    if agent.runtime:
        payload["runtime"] = agent.runtime
    return payload


def _blocked_tools_for(agent: AgentSpec) -> list[str]:
    if agent.permission.value == "read_only":
        return list(READ_ONLY_BLOCKED_TOOLS)
    return list(ASK_BLOCKED_TOOLS)


def _required_mcp_servers(agent: AgentSpec) -> list[str]:
    servers: list[str] = []
    if "mcp-codegraph" in agent.tools:
        servers.append("codegraph")
    return servers


def _default_soul(agent: AgentSpec) -> str:
    summary = (agent.role_summary or agent.name).rstrip("。.")
    return f"你是 {agent.name}，负责{summary}。请遵守该 Agent 的工具权限、风险等级和只读/执行边界。"


def _validate_capability_routes(
    capability_routes: Mapping[str, Any],
    agents: Mapping[str, AgentSpec],
) -> None:
    for capability, agent_id in capability_routes.items():
        if agent_id not in agents:
            raise ValueError(f"capability route {capability!r} references unknown agent {agent_id!r}")
        if capability not in agents[agent_id].capabilities:
            raise ValueError(
                f"capability route {capability!r} points to {agent_id!r}, "
                "but that agent does not declare the capability"
            )
