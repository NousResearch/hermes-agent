"""Permission guard for managed-agent delegation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .registry import AgentSpec, AgentStatus, PermissionMode, RiskLevel

_WRITE_CAPABLE_TOOLS = frozenset(
    {
        "git",
        "patch",
        "write_file",
        "filesystem_write",
        "edit",
        "apply_patch",
        "terminal",
        "process",
    }
)


class DelegationPermissionError(ValueError):
    """Raised when an agent or task violates delegation permissions."""


def _normalize_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    if isinstance(values, Iterable):
        return [str(item).strip() for item in values if str(item).strip()]
    value = str(values).strip()
    return [value] if value else []


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass(frozen=True, slots=True)
class PermissionSnapshot:
    agent_id: str
    risk_level: str
    allowed_files: tuple[str, ...] = ()
    forbidden_actions: tuple[str, ...] = ()
    requested_tools: tuple[str, ...] = ()
    human_approved: bool = False
    agent_permission: str = "ask"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PermissionGuard:
    """Validate that a managed agent can receive a delegated task."""

    def resolve_target_agent_id(self, task_card: Any, routing_decision: Any = None) -> str:
        owner_agent = _get_attr(task_card, "owner_agent", None)
        if owner_agent:
            return str(owner_agent).strip()

        if routing_decision is not None:
            agents = _normalize_list(_get_attr(routing_decision, "agents", []))
            if agents:
                return agents[0]

        execution_plan = _get_attr(task_card, "execution_plan", None)
        agents = _normalize_list(_get_attr(execution_plan, "agents", []))
        if agents:
            return agents[0]

        raise DelegationPermissionError("TaskCard does not identify an owner agent")

    def build_snapshot(
        self,
        agent: AgentSpec,
        *,
        risk_level: RiskLevel,
        allowed_files: Iterable[str] | None = None,
        forbidden_actions: Iterable[str] | None = None,
        requested_tools: Iterable[str] | None = None,
        human_approved: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> PermissionSnapshot:
        self.check_agent_allowed(
            agent,
            risk_level=risk_level,
            requested_tools=requested_tools,
            human_approved=human_approved,
        )
        return PermissionSnapshot(
            agent_id=agent.agent_id,
            risk_level=risk_level.value,
            allowed_files=tuple(_normalize_list(allowed_files)),
            forbidden_actions=tuple(_normalize_list(forbidden_actions)),
            requested_tools=tuple(_normalize_list(requested_tools)),
            human_approved=bool(human_approved),
            agent_permission=agent.permission.value,
            metadata=dict(metadata or {}),
        )

    def check_agent_allowed(
        self,
        agent: AgentSpec,
        *,
        risk_level: RiskLevel,
        requested_tools: Iterable[str] | None = None,
        human_approved: bool = False,
    ) -> None:
        if agent.status is not AgentStatus.ACTIVE:
            raise DelegationPermissionError(
                f"Agent {agent.agent_id} is not active: {agent.status.value}"
            )

        if agent.can_delegate:
            raise DelegationPermissionError(
                f"Agent {agent.agent_id} cannot delegate further"
            )

        if risk_level is RiskLevel.R4 and not human_approved:
            raise DelegationPermissionError(
                f"Risk {risk_level.value} requires human approval"
            )

        if not agent.allows_risk(risk_level):
            raise DelegationPermissionError(
                f"Agent {agent.agent_id} cannot handle risk {risk_level.value}"
            )

        requested = _normalize_list(requested_tools)
        if agent.permission is PermissionMode.READ_ONLY:
            write_tools = [tool for tool in requested if tool in _WRITE_CAPABLE_TOOLS]
            if write_tools:
                raise DelegationPermissionError(
                    f"read_only agent {agent.agent_id} cannot use write-capable tools: {write_tools}"
                )
