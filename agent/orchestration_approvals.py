"""Human approval interruption primitives for orchestration workflows."""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class ApprovalInterrupt:
    action: str
    reason: str
    payload: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "medium"
    approval_id: str = field(default_factory=lambda: f"appr-{uuid.uuid4().hex[:10]}")
    status: str = "pending_approval"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def approval_required(
    *,
    action: str,
    reason: str,
    payload: Mapping[str, Any] | None = None,
    risk_level: str = "medium",
) -> ApprovalInterrupt:
    return ApprovalInterrupt(
        action=action,
        reason=reason,
        payload=dict(payload or {}),
        risk_level=risk_level,
    )


def resume_after_approval(
    interrupt: ApprovalInterrupt | Mapping[str, Any],
    *,
    approved: bool,
    approver: str = "",
    note: str = "",
) -> dict[str, Any]:
    data = interrupt.to_dict() if isinstance(interrupt, ApprovalInterrupt) else dict(interrupt)
    data.update(
        {
            "status": "approved" if approved else "rejected",
            "approved": bool(approved),
            "approver": approver,
            "approval_note": note,
            "resolved_at": time.time(),
        }
    )
    return data
