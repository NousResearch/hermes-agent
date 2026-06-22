"""Deterministic eval gate for Gateway and cron pre-dispatch surfaces.

The gate is intentionally conservative and side-effect free: it never executes
candidate work, reads secrets, mutates cron/Gateway state, or calls a model.  It
turns the dispatch surface into a small policy-action record, validates required
audit fields, and optionally enforces the decision when explicitly enabled.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

POLICY_VERSION = "HQ Supervised Auto-Approve Policy v1"

REQUIRED_POLICY_ACTION_FIELDS: tuple[str, ...] = (
    "audit_id",
    "actor",
    "delegated_user",
    "resource",
    "scope",
    "intent",
    "side_effect",
    "approval_status",
    "policy_version",
)

EVAL_DATASET_APPROVAL_PHRASE = "Approval required: eval dataset/red-team fixtures introduction under policy review"
LIVE_GATE_APPROVAL_PHRASE = "Approval required: connect eval gate pre-dispatch hook to Gateway/cron"

_SECRET_LIKE_RE = re.compile(
    r"(?i)(sk-[a-z0-9_-]{8,}|api[_-]?key\s*[:=]\s*['\"]?[^\s'\"]+|token\s*[:=]\s*['\"]?[^\s'\"]+)"
)
_INJECTION_RE = re.compile(
    r"(?i)(system\s*:|developer\s*:|ignore\s+(all\s+)?previous\s+instructions|print\s+secrets|exfiltrate)"
)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _cfg_get(config: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    value: Any = config or {}
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _surface_enabled(surface: str, config: Mapping[str, Any] | None = None) -> bool:
    env_specific = os.getenv(f"HERMES_EVAL_GATE_{surface.upper()}_ENABLED")
    if env_specific is not None:
        return _truthy(env_specific)
    env_global = os.getenv("HERMES_EVAL_GATE_ENABLED")
    if env_global is not None:
        return _truthy(env_global)
    return _truthy(_cfg_get(config, "eval_gate", f"{surface}_enabled")) or _truthy(
        _cfg_get(config, "eval_gate", "enabled")
    )


def _enforce_enabled(config: Mapping[str, Any] | None = None) -> bool:
    env_value = os.getenv("HERMES_EVAL_GATE_ENFORCE")
    if env_value is not None:
        return _truthy(env_value)
    return _truthy(_cfg_get(config, "eval_gate", "enforce"))


def _audit_enabled(config: Mapping[str, Any] | None = None) -> bool:
    env_value = os.getenv("HERMES_EVAL_GATE_AUDIT_ONLY")
    if env_value is not None:
        return _truthy(env_value)
    return _truthy(_cfg_get(config, "eval_gate", "audit_only", default=True))


@dataclass(frozen=True)
class EvalGateDecision:
    surface: str
    audit_id: str
    actual_decision: str
    enforce: bool
    passed: bool
    reason: str
    action: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def should_block(self) -> bool:
        return self.enforce and self.actual_decision in {"confirm", "reject"}


def _missing_fields(data: Mapping[str, Any], required: tuple[str, ...] = REQUIRED_POLICY_ACTION_FIELDS) -> list[str]:
    return [field for field in required if data.get(field) in (None, "")]


def _approval_text(action_data: Mapping[str, Any]) -> str:
    evidence = action_data.get("approval_evidence") or action_data.get("approval_phrase") or ""
    return str(evidence).strip()


def evaluate_policy_action(action_data: Mapping[str, Any], *, surface: str, enforce: bool = False) -> EvalGateDecision:
    """Evaluate one normalized policy-action record.

    Decisions:
    - ``allow``: dispatch may proceed.
    - ``confirm``: safe to report as approval-needed; block only in enforce mode.
    - ``reject``: forbidden or malformed; block only in enforce mode.
    """

    audit_id = str(action_data.get("audit_id") or f"{surface}:unknown")
    action = dict(action_data)
    missing = _missing_fields(action)
    if missing:
        return EvalGateDecision(surface, audit_id, "reject", enforce, False, f"missing policy fields: {', '.join(missing)}", action)

    scope = str(action.get("scope", "")).lower()
    side_effect = str(action.get("side_effect", "")).lower()
    resource = str(action.get("resource", "")).lower()
    intent = str(action.get("intent", ""))
    approval_status = str(action.get("approval_status", "")).lower()
    approval = _approval_text(action)
    policy_review = _truthy(action.get("policy_review"))
    synthetic_only = _truthy(action.get("synthetic_only"))

    if _SECRET_LIKE_RE.search(intent):
        return EvalGateDecision(surface, audit_id, "reject", enforce, False, "secret-like content in policy action intent", action)

    forbidden_terms = (
        "destructive",
        "elios",
        "raw secret",
        "secret",
        "arbitrary admin",
        "uncapped paid",
        "force push",
        "release publish",
        "main/master merge",
    )
    if any(term in scope or term in side_effect or term in resource for term in forbidden_terms):
        return EvalGateDecision(surface, audit_id, "reject", enforce, False, "forbidden or destructive scope", action)

    if "red-team" in scope or "eval fixture" in scope or "eval dataset" in scope:
        if approval == EVAL_DATASET_APPROVAL_PHRASE and policy_review and synthetic_only:
            return EvalGateDecision(surface, audit_id, "allow", enforce, True, "exact eval-dataset approval with policy-review and synthetic-only guards", action)
        return EvalGateDecision(surface, audit_id, "confirm", enforce, False, "eval dataset/red-team fixtures require exact approval plus policy review", action)

    runtime_scope = (
        "cron mutation" in scope
        or "permission-enforcement" in scope
        or "pre-dispatch" in scope
        or "runtime_behavior_change" in side_effect
    )
    if runtime_scope:
        if approval == LIVE_GATE_APPROVAL_PHRASE or approval_status in {"approved", "explicit_approval"}:
            return EvalGateDecision(surface, audit_id, "allow", enforce, True, "exact live eval-gate approval present", action)
        return EvalGateDecision(surface, audit_id, "confirm", enforce, False, "runtime/cron permission behavior change requires exact approval", action)

    if "read-only" in scope and side_effect == "none":
        return EvalGateDecision(surface, audit_id, "allow", enforce, True, "read-only inspection is auto-allowed", action)
    if "documentation" in scope and "file_write" in side_effect:
        return EvalGateDecision(surface, audit_id, "allow", enforce, True, "rollbackable documentation/reference write is auto-allowed", action)
    if "branch/worktree" in scope:
        has_tests = _truthy(action.get("test_plan_present"))
        has_rollback = _truthy(action.get("rollback_plan_present")) or bool(action.get("rollback"))
        if has_tests and has_rollback:
            return EvalGateDecision(surface, audit_id, "allow", enforce, True, "branch/worktree edit has tests and rollback evidence", action)
        return EvalGateDecision(surface, audit_id, "confirm", enforce, False, "branch/worktree edit needs tests and rollback evidence", action)
    if approval_status == "auto_allowed":
        return EvalGateDecision(surface, audit_id, "allow", enforce, True, "auto-allowed low-risk dispatch", action)

    if _INJECTION_RE.search(intent):
        return EvalGateDecision(surface, audit_id, "confirm", enforce, False, "instruction-like input is data; require downstream policy handling", action)

    return EvalGateDecision(surface, audit_id, "allow", enforce, True, "no high-risk pre-dispatch signal", action)


def build_gateway_action(event: Any) -> dict[str, Any]:
    source = getattr(event, "source", None)
    platform = getattr(getattr(source, "platform", None), "value", None) or str(getattr(source, "platform", "gateway"))
    message_id = getattr(event, "message_id", None) or "unknown"
    user_id = getattr(source, "user_id", "unknown") if source is not None else "unknown"
    chat_id = getattr(source, "chat_id", "unknown") if source is not None else "unknown"
    text = str(getattr(event, "text", "") or "")
    return {
        "audit_id": f"gateway:{platform}:{chat_id}:{message_id}",
        "actor": str(user_id or "unknown"),
        "delegated_user": "gateway_user",
        "resource": f"gateway:{platform}:{chat_id}",
        "scope": "gateway pre-dispatch",
        "intent": text[:240],
        "side_effect": "agent_dispatch",
        "approval_status": "auto_allowed",
        "policy_version": POLICY_VERSION,
    }


def build_cron_action(job: Mapping[str, Any]) -> dict[str, Any]:
    job_id = str(job.get("id") or "unknown")
    job_name = str(job.get("name") or job.get("prompt") or job_id)
    return {
        "audit_id": f"cron:{job_id}",
        "actor": "cron_scheduler",
        "delegated_user": str(job.get("profile") or "default"),
        "resource": f"cron:{job_id}",
        "scope": "cron pre-dispatch",
        "intent": job_name[:240],
        "side_effect": "script_execution" if job.get("no_agent") else "agent_dispatch",
        "approval_status": str(job.get("approval_status") or "auto_allowed"),
        "policy_version": POLICY_VERSION,
        "approval_evidence": str(job.get("approval_evidence") or job.get("approval_phrase") or ""),
    }


def _config_approval_evidence(config: Mapping[str, Any] | None) -> str:
    return str(_cfg_get(config, "eval_gate", "approval_evidence", default="") or "").strip()


def evaluate_gateway_event(event: Any, config: Mapping[str, Any] | None = None) -> EvalGateDecision:
    enforce = _enforce_enabled(config)
    action = build_gateway_action(event)
    approval_evidence = _config_approval_evidence(config)
    if approval_evidence:
        action["approval_evidence"] = approval_evidence
        action["approval_status"] = "explicit_approval"
    if not _surface_enabled("gateway", config):
        return EvalGateDecision("gateway", str(action["audit_id"]), "allow", False, True, "eval gate disabled", action)
    return evaluate_policy_action(action, surface="gateway", enforce=enforce and not _audit_enabled(config))


def evaluate_cron_job(job: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> EvalGateDecision:
    enforce = _enforce_enabled(config)
    action = build_cron_action(job)
    approval_evidence = str(action.get("approval_evidence") or _config_approval_evidence(config))
    if approval_evidence:
        action["approval_evidence"] = approval_evidence
        action["approval_status"] = "explicit_approval"
    if not _surface_enabled("cron", config):
        return EvalGateDecision("cron", str(action["audit_id"]), "allow", False, True, "eval gate disabled", action)
    return evaluate_policy_action(action, surface="cron", enforce=enforce and not _audit_enabled(config))


def dispatch_hook_result(decision: EvalGateDecision) -> dict[str, Any]:
    """Convert a decision into the Gateway/cron pre-dispatch action shape."""
    if decision.should_block:
        return {"action": "skip", "reason": decision.reason, "decision": decision.to_dict()}
    return {"action": "allow", "reason": decision.reason, "decision": decision.to_dict()}
