"""Policy primitives for cross-profile delegation.

The MVP still uses conservative text heuristics, but callers pass a structured
``DelegationAction`` so future enforcement can authorize concrete tool/action
calls (for example ``mcp:vercel`` tool ``list_projects`` read vs
``deploy_project`` write) without redesigning the API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional


class DelegationRisk(str, Enum):
    READ = "READ"
    PREPARE = "PREPARE"
    CONSEQUENTIAL_WRITE = "CONSEQUENTIAL_WRITE"
    CREDENTIAL_EXPORT = "CREDENTIAL_EXPORT"


class PolicyDecisionStatus(str, Enum):
    ALLOW = "allow"
    BLOCK_APPROVAL = "block_approval"
    DENY = "deny"


@dataclass(frozen=True)
class ToolActionRequest:
    """Future-facing permission atom for a concrete tool action."""

    capability: str
    tool_name: Optional[str] = None
    action_name: Optional[str] = None
    operation: Optional[str] = None
    arguments_schema: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class DelegationAction:
    requester_profile: str
    executor_profile: str
    task: str
    required_capability: str
    requested_risk: DelegationRisk = DelegationRisk.READ
    tool_action: Optional[ToolActionRequest] = None
    approval_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["requested_risk"] = self.requested_risk.value
        return data


@dataclass(frozen=True)
class PolicyDecision:
    status: PolicyDecisionStatus
    risk: DelegationRisk
    reason: str
    approval_required: bool = False

    @property
    def allowed(self) -> bool:
        return self.status == PolicyDecisionStatus.ALLOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "risk": self.risk.value,
            "reason": self.reason,
            "approval_required": self.approval_required,
            "allowed": self.allowed,
        }


_CREDENTIAL_EXPORT_TERMS = (
    "api key", "apikey", "secret", "token", "oauth", "authorization",
    "bearer", "cookie", "credential", "password", "private key",
)
_WRITE_TERMS = (
    "deploy", "delete", "remove", "publish", "send", "transfer", "trade",
    "buy", "sell", "charge", "pay", "spend", "modify", "update", "change",
    "create", "provision", "rotate", "revoke", "grant", "permission",
)
_PREPARE_TERMS = ("draft", "prepare", "plan", "generate", "summarize", "review")
_READ_TERMS = ("read", "inspect", "list", "get", "check", "query", "status", "view", "analyze")


def _coerce_risk(value: str | DelegationRisk | None) -> DelegationRisk:
    if isinstance(value, DelegationRisk):
        return value
    if not value:
        return DelegationRisk.READ
    try:
        return DelegationRisk(str(value).upper())
    except ValueError:
        return DelegationRisk.READ


def classify_delegation_risk(
    *,
    requested_risk: str | DelegationRisk = DelegationRisk.READ,
    task: str = "",
    required_capability: str = "",
    tool_action: Optional[ToolActionRequest] = None,
) -> DelegationRisk:
    """Conservative MVP classifier over a structured policy interface.

    Future versions should primarily use ``tool_action`` metadata emitted by
    tool wrappers/MCP descriptors. Text classification is deliberately kept as
    a fallback, not the permanent authorization model.
    """

    requested = _coerce_risk(requested_risk)
    haystack = " ".join(
        str(x or "") for x in (
            task,
            required_capability,
            getattr(tool_action, "tool_name", None),
            getattr(tool_action, "action_name", None),
            getattr(tool_action, "operation", None),
        )
    ).lower()

    if any(term in haystack for term in _CREDENTIAL_EXPORT_TERMS) and any(
        verb in haystack for verb in ("show", "print", "export", "send", "return", "reveal")
    ):
        return DelegationRisk.CREDENTIAL_EXPORT

    if requested in {DelegationRisk.CONSEQUENTIAL_WRITE, DelegationRisk.CREDENTIAL_EXPORT}:
        return requested

    if any(term in haystack for term in _WRITE_TERMS):
        # A caller may explicitly mark a non-mutating preparation task as PREPARE.
        if requested == DelegationRisk.PREPARE and any(term in haystack for term in _PREPARE_TERMS):
            return DelegationRisk.PREPARE
        return DelegationRisk.CONSEQUENTIAL_WRITE

    if requested == DelegationRisk.PREPARE:
        return DelegationRisk.PREPARE

    if any(term in haystack for term in _READ_TERMS):
        return DelegationRisk.READ

    return requested


def enforce_delegation_policy(action: DelegationAction) -> PolicyDecision:
    risk = classify_delegation_risk(
        requested_risk=action.requested_risk,
        task=action.task,
        required_capability=action.required_capability,
        tool_action=action.tool_action,
    )

    if risk == DelegationRisk.CREDENTIAL_EXPORT:
        return PolicyDecision(
            status=PolicyDecisionStatus.DENY,
            risk=risk,
            reason="Credential export is never allowed; executor must perform the action and return non-secret results only.",
        )
    if risk == DelegationRisk.CONSEQUENTIAL_WRITE and not action.approval_id:
        return PolicyDecision(
            status=PolicyDecisionStatus.BLOCK_APPROVAL,
            risk=risk,
            reason="Consequential external writes require explicit Michael approval before worker execution.",
            approval_required=True,
        )
    return PolicyDecision(
        status=PolicyDecisionStatus.ALLOW,
        risk=risk,
        reason="Allowed by Phase 1 delegation policy.",
    )
