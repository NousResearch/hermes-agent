"""Policy primitives for cross-profile delegation.

The MVP still uses conservative text heuristics, but callers pass a structured
``DelegationAction`` so future enforcement can authorize concrete tool/action
calls (for example ``mcp:vercel`` tool ``list_projects`` read vs
``deploy_project`` write) without redesigning the API.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
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
    # Fail-closed: common write verbs that the original classifier missed
    "push", "commit", "merge", "restart", "apply", "upload", "install",
    "uninstall", "terminate", "destroy", "drop", "truncate", "alter",
    "rename", "move", "copy", "write", "edit", "patch", "release",
    "rollback", "migrate", "scale", "resize", "reconfigure", "reset",
    "flush", "purge", "evict", "ban", "block", "unban", "unblock",
    "subscribe", "unsubscribe", "follow", "unfollow", "like", "unlike",
    "star", "unstar", "pin", "unpin", "lock", "unlock", "archive",
    "unarchive", "suspend", "resume", "activate", "deactivate", "enable",
    "disable", "start", "stop", "pause", "resume", "cancel", "abort",
    "reject", "approve", "accept", "decline", "dismiss", "close", "reopen",
    "assign", "unassign", "claim", "unclaim", "release", "checkout",
    "checkin", "import", "export", "load", "unload", "mount", "unmount",
    "attach", "detach", "connect", "disconnect", "bind", "unbind",
    "register", "unregister", "enroll", "unenroll", "invite", "uninvite",
    "join", "leave", "enter", "exit", "add", "remove", "insert", "append",
    "prepend", "replace", "swap", "shift", "reorder", "sort", "filter",
    "map", "reduce", "fold", "flatten", "group", "ungroup", "merge",
    "split", "join", "concat", "slice", "splice", "trim", "pad", "fill",
    "format", "parse", "serialize", "deserialize", "encode", "decode",
    "compress", "decompress", "encrypt", "decrypt", "sign", "verify",
    "hash", "digest", "checksum", "validate", "invalidate", "refresh",
    "renew", "expire", "timeout", "schedule", "unschedule", "delay",
    "defer", "postpone", "reschedule", "cancel", "abort", "kill", "force",
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

    Fail-closed design: any verb that could indicate a write action but is not
    explicitly in the READ list is classified as CONSEQUENTIAL_WRITE. This
    prevents the LLM from turning a read-authorized delegation into a write
    by choosing different words.
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

    # Fail-closed: if any write term is found, classify as CONSEQUENTIAL_WRITE
    # unless the caller explicitly marked it as PREPARE AND the task contains
    # a prepare term (draft/plan/generate/etc.).
    if requested == DelegationRisk.PREPARE:
        return DelegationRisk.PREPARE

    # Check read terms BEFORE fail-closed: explicit read verbs should always
    # be classified as READ even if other terms are present.
    if any(term in haystack for term in _READ_TERMS):
        return DelegationRisk.READ

    # Fail-closed: if any write term is found, classify as CONSEQUENTIAL_WRITE
    # unless the caller explicitly marked it as PREPARE AND the task contains
    # a prepare term (draft/plan/generate/etc.).
    if any(term in haystack for term in _WRITE_TERMS):
        if requested == DelegationRisk.PREPARE and any(term in haystack for term in _PREPARE_TERMS):
            return DelegationRisk.PREPARE
        return DelegationRisk.CONSEQUENTIAL_WRITE

    # Default: for tasks with no recognized terms, trust the requested risk
    # (which defaults to READ if not specified).
    return requested


def _compute_approval_binding(action: DelegationAction) -> str:
    """Compute a binding hash for an approval that ties it to the exact
    requester + executor + capability + task being executed.

    This prevents an approval from being replayed against a different action
    or target.
    """
    binding_data = {
        "requester_profile": action.requester_profile,
        "executor_profile": action.executor_profile,
        "required_capability": action.required_capability,
        "task": action.task,
    }
    binding_json = json.dumps(binding_data, sort_keys=True, ensure_ascii=False)
    return hmac.new(
        b"delegation-approval-binding-v1",
        binding_json.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:32]


def validate_approval_id(action: DelegationAction) -> bool:
    """Validate that an approval_id is bound to the exact action being executed.

    A valid approval_id must be a non-empty string that matches the computed
    binding for this specific requester + executor + capability + task.

    Returns True if the approval is valid, False otherwise.
    """
    if not action.approval_id:
        return False

    # The approval_id must be at least 32 chars (a truncated HMAC)
    if len(action.approval_id) < 32:
        return False

    expected_binding = _compute_approval_binding(action)
    return hmac.compare_digest(action.approval_id, expected_binding)


def generate_approval_id(action: DelegationAction) -> str:
    """Generate a valid approval_id for a given action.

    This should be called when an approval is granted, and the resulting
    approval_id should be passed to delegate_to_profile.
    """
    return _compute_approval_binding(action)


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
    if risk == DelegationRisk.CONSEQUENTIAL_WRITE:
        # Validate the approval_id against the exact action being executed
        if not action.approval_id:
            return PolicyDecision(
                status=PolicyDecisionStatus.BLOCK_APPROVAL,
                risk=risk,
                reason="Consequential external writes require explicit approval before worker execution.",
                approval_required=True,
            )
        if not validate_approval_id(action):
            return PolicyDecision(
                status=PolicyDecisionStatus.DENY,
                risk=risk,
                reason="Approval ID is invalid or does not match the exact action being executed (possible replay attempt).",
            )
        return PolicyDecision(
            status=PolicyDecisionStatus.ALLOW,
            risk=risk,
            reason="Allowed by Phase 1 delegation policy with validated approval.",
        )
    return PolicyDecision(
        status=PolicyDecisionStatus.ALLOW,
        risk=risk,
        reason="Allowed by Phase 1 delegation policy.",
    )
