"""Agent-native execution policy primitives.

This module centralizes the safety envelope for unattended and child-agent
runs.  It is intentionally dependency-light and pure: callers pass the tool
name/toolset and receive an allow/block decision plus auditable metadata.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


_POLICY_MODES = {"off", "audit", "enforce"}
_HIGH_RISK_TOOLSETS = frozenset({"terminal", "browser", "mcp", "cronjob", "messaging", "code_execution"})
_HIGH_RISK_TOOLS = frozenset({
    "terminal",
    "execute_code",
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_scroll",
    "cronjob",
    "send_message",
    "skill_manage",
    "delegate_task",
})


@dataclass(frozen=True)
class ExecutionPolicy:
    """Normalized execution policy for one agent run."""

    mode: str = "audit"  # off | audit | enforce
    source: str = "interactive"
    policy_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_policy_id: str | None = None
    allow_tools: frozenset[str] | None = None
    deny_tools: frozenset[str] = field(default_factory=frozenset)
    allow_toolsets: frozenset[str] | None = None
    deny_toolsets: frozenset[str] = field(default_factory=frozenset)
    disable_recursive_delegation: bool = False
    sandbox_required: bool = False
    approval_required_for_high_risk: bool = True
    network_policy: str = "default"  # default | disabled | browser_only | open

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source": self.source,
            "policy_id": self.policy_id,
            "parent_policy_id": self.parent_policy_id,
            "allow_tools": sorted(self.allow_tools) if self.allow_tools is not None else None,
            "deny_tools": sorted(self.deny_tools),
            "allow_toolsets": sorted(self.allow_toolsets) if self.allow_toolsets is not None else None,
            "deny_toolsets": sorted(self.deny_toolsets),
            "disable_recursive_delegation": self.disable_recursive_delegation,
            "sandbox_required": self.sandbox_required,
            "approval_required_for_high_risk": self.approval_required_for_high_risk,
            "network_policy": self.network_policy,
        }


@dataclass(frozen=True)
class ExecutionPolicyDecision:
    action: str = "allow"  # allow | audit | block
    code: str = "allowed"
    message: str = ""
    tool_name: str = ""
    toolset: str | None = None
    policy_id: str | None = None
    source: str | None = None

    @property
    def allows_execution(self) -> bool:
        return self.action != "block"

    def to_audit_event(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "code": self.code,
            "message": self.message,
            "tool_name": self.tool_name,
            "toolset": self.toolset,
            "policy_id": self.policy_id,
            "source": self.source,
        }


def _string_set(value: Any) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        return None
    normalized = frozenset(str(item).strip() for item in items if str(item).strip())
    return normalized or frozenset()


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def default_execution_policy(source: str = "interactive") -> ExecutionPolicy:
    """Return the default policy for a run source."""
    source = (source or "interactive").strip().lower()
    if source == "cron":
        return ExecutionPolicy(
            mode="enforce",
            source="cron",
            deny_tools=frozenset({"skill_manage", "delegate_task"}),
            deny_toolsets=frozenset({"messaging", "cronjob"}),
            disable_recursive_delegation=True,
            sandbox_required=True,
            approval_required_for_high_risk=True,
            network_policy="default",
        )
    if source in {"delegate", "subagent"}:
        return ExecutionPolicy(
            mode="enforce",
            source="delegate",
            deny_tools=frozenset({"delegate_task"}),
            disable_recursive_delegation=True,
            sandbox_required=True,
            approval_required_for_high_risk=True,
            network_policy="default",
        )
    return ExecutionPolicy(mode="audit", source=source or "interactive")


def normalize_execution_policy(value: Mapping[str, Any] | ExecutionPolicy | None, *, source: str = "interactive") -> ExecutionPolicy:
    """Normalize a user/job supplied mapping into an ExecutionPolicy."""
    if isinstance(value, ExecutionPolicy):
        return value
    base = default_execution_policy(source)
    if not isinstance(value, Mapping):
        return base
    mode = str(value.get("mode", base.mode)).strip().lower()
    if mode not in _POLICY_MODES:
        mode = base.mode
    return ExecutionPolicy(
        mode=mode,
        source=str(value.get("source", base.source)).strip() or base.source,
        policy_id=str(value.get("policy_id", base.policy_id)).strip() or base.policy_id,
        parent_policy_id=value.get("parent_policy_id", base.parent_policy_id),
        allow_tools=_string_set(value.get("allow_tools", base.allow_tools)),
        deny_tools=_string_set(value.get("deny_tools", base.deny_tools)) or frozenset(),
        allow_toolsets=_string_set(value.get("allow_toolsets", base.allow_toolsets)),
        deny_toolsets=_string_set(value.get("deny_toolsets", base.deny_toolsets)) or frozenset(),
        disable_recursive_delegation=_bool(value.get("disable_recursive_delegation", base.disable_recursive_delegation)),
        sandbox_required=_bool(value.get("sandbox_required", base.sandbox_required)),
        approval_required_for_high_risk=_bool(value.get("approval_required_for_high_risk", base.approval_required_for_high_risk), True),
        network_policy=str(value.get("network_policy", base.network_policy)).strip().lower() or base.network_policy,
    )


def derive_child_policy(parent: ExecutionPolicy | Mapping[str, Any] | None, *, child_toolsets: Sequence[str] | None = None) -> ExecutionPolicy:
    """Derive a narrower policy for a delegated child agent."""
    parent_policy = normalize_execution_policy(parent, source="interactive")
    default_child = default_execution_policy("delegate")
    child_allow_toolsets = _string_set(child_toolsets) if child_toolsets else parent_policy.allow_toolsets
    return ExecutionPolicy(
        mode=parent_policy.mode if parent_policy.mode != "off" else default_child.mode,
        source="delegate",
        policy_id=uuid.uuid4().hex[:12],
        parent_policy_id=parent_policy.policy_id,
        allow_tools=parent_policy.allow_tools,
        deny_tools=frozenset(set(parent_policy.deny_tools) | {"delegate_task"}),
        allow_toolsets=child_allow_toolsets,
        deny_toolsets=parent_policy.deny_toolsets,
        disable_recursive_delegation=True,
        sandbox_required=True,
        approval_required_for_high_risk=parent_policy.approval_required_for_high_risk,
        network_policy=parent_policy.network_policy,
    )


def decide_tool_call(policy: ExecutionPolicy | Mapping[str, Any] | None, tool_name: str, *, toolset: str | None = None) -> ExecutionPolicyDecision:
    policy = normalize_execution_policy(policy)
    name = str(tool_name or "").strip()
    ts = str(toolset or "").strip() or None
    if policy.mode == "off":
        return ExecutionPolicyDecision("allow", "policy_off", tool_name=name, toolset=ts, policy_id=policy.policy_id, source=policy.source)

    reason = None
    if policy.allow_tools is not None and name not in policy.allow_tools:
        reason = "tool_not_allowlisted"
    elif name in policy.deny_tools:
        reason = "tool_denied"
    elif policy.allow_toolsets is not None and ts not in policy.allow_toolsets:
        reason = "toolset_not_allowlisted"
    elif ts in policy.deny_toolsets:
        reason = "toolset_denied"
    elif policy.disable_recursive_delegation and name == "delegate_task":
        reason = "recursive_delegation_disabled"
    elif policy.network_policy == "disabled" and (ts in {"web", "browser"} or (ts or "").startswith("mcp-")):
        reason = "network_disabled"

    if reason:
        action = "block" if policy.mode == "enforce" else "audit"
        return ExecutionPolicyDecision(
            action=action,
            code=reason,
            message=f"Execution policy {policy.policy_id} {action}ed {name}: {reason}",
            tool_name=name,
            toolset=ts,
            policy_id=policy.policy_id,
            source=policy.source,
        )

    if name in _HIGH_RISK_TOOLS or ts in _HIGH_RISK_TOOLSETS or (ts or "").startswith("mcp-"):
        return ExecutionPolicyDecision(
            action="audit",
            code="high_risk_tool_observed",
            message=f"Execution policy {policy.policy_id} audited high-risk tool {name}",
            tool_name=name,
            toolset=ts,
            policy_id=policy.policy_id,
            source=policy.source,
        )
    return ExecutionPolicyDecision("allow", "allowed", tool_name=name, toolset=ts, policy_id=policy.policy_id, source=policy.source)


def block_result(decision: ExecutionPolicyDecision) -> dict[str, Any]:
    return {"error": decision.message or "Tool blocked by execution policy", "policy_audit_event": decision.to_audit_event()}
