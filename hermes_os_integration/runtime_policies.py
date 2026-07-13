"""Runtime policies for Hermes OS delegated execution."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RuntimePolicy:
    max_cost_usd: float = 1.0
    max_retries: int = 1
    approval_required_actions: List[str] = field(default_factory=lambda: ["write", "deploy", "purchase"])
    audit_enabled: bool = True
    retry_backoff_seconds: List[int] = field(default_factory=lambda: [1, 2, 5])


@dataclass(frozen=True)
class RuntimeDecision:
    allowed: bool
    status: str
    reasons: List[str] = field(default_factory=list)
    retry_allowed: bool = False
    approval_required: bool = False
    audit: Dict[str, object] = field(default_factory=dict)


def evaluate_runtime_policy(
    *,
    action: str,
    estimated_cost_usd: float = 0.0,
    retry_count: int = 0,
    approved: bool = False,
    policy: Optional[RuntimePolicy] = None,
):
    policy = policy or RuntimePolicy()
    reasons = []
    approval_required = action in set(policy.approval_required_actions) and not approved
    if estimated_cost_usd > policy.max_cost_usd:
        reasons.append("estimated cost exceeds policy")
    if retry_count > policy.max_retries:
        reasons.append("retry limit exceeded")
    if approval_required:
        reasons.append("human approval required")

    allowed = not reasons
    return RuntimeDecision(
        allowed=allowed,
        status="allowed" if allowed else "blocked",
        reasons=reasons,
        retry_allowed=retry_count < policy.max_retries,
        approval_required=approval_required,
        audit=create_runtime_audit(
            action=action,
            estimated_cost_usd=estimated_cost_usd,
            retry_count=retry_count,
            approved=approved,
            allowed=allowed,
            reasons=reasons,
        ) if policy.audit_enabled else {},
    )


def create_runtime_audit(
    *,
    action: str,
    estimated_cost_usd: float,
    retry_count: int,
    approved: bool,
    allowed: bool,
    reasons: List[str],
):
    return {
        "type": "runtime_policy_decision",
        "action": action,
        "estimated_cost_usd": float(estimated_cost_usd),
        "retry_count": int(retry_count),
        "approved": bool(approved),
        "allowed": bool(allowed),
        "reasons": list(reasons),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def retry_backoff_seconds(retry_count: int, policy: Optional[RuntimePolicy] = None):
    policy = policy or RuntimePolicy()
    if retry_count < 0:
        return 0
    if retry_count < len(policy.retry_backoff_seconds):
        return policy.retry_backoff_seconds[retry_count]
    return policy.retry_backoff_seconds[-1] if policy.retry_backoff_seconds else 0


def aggregate_cost_budget(records: List[Dict[str, object]], project_id: str = "", work_graph_id: str = ""):
    filtered = []
    for record in records:
        if project_id and record.get("project_id") != project_id:
            continue
        if work_graph_id and record.get("work_graph_id") != work_graph_id:
            continue
        filtered.append(record)
    return {
        "project_id": project_id,
        "work_graph_id": work_graph_id,
        "estimated_cost_usd": sum(float(record.get("estimated_cost_usd", 0) or 0) for record in filtered),
        "actual_cost_usd": sum(float(record.get("actual_cost_usd", 0) or 0) for record in filtered),
        "record_count": len(filtered),
    }


def approval_prompt_for_decision(decision: RuntimeDecision):
    return {
        "required": decision.approval_required,
        "status": decision.status,
        "reasons": decision.reasons,
        "prompt": "Approve high-risk Hermes OS runtime action?" if decision.approval_required else "",
    }
