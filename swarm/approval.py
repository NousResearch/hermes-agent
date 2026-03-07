from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from swarm.types import SwarmTask


@dataclass
class ApprovalPolicy:
    require_approval_for_tools: list[str] = field(
        default_factory=lambda: ["terminal", "git_push", "deploy"]
    )
    require_approval_for_roles: list[str] = field(
        default_factory=lambda: ["merger"]
    )
    spend_threshold_usd: float = 1.0
    risk_tiers: dict[str, str] = field(default_factory=lambda: {
        "read": "low",
        "glob": "low",
        "grep": "low",
        "web_search": "low",
        "web_fetch": "low",
        "write": "medium",
        "edit": "medium",
        "bash": "high",
        "terminal": "high",
        "git_push": "critical",
        "deploy": "critical",
    })
    auto_approve_low_risk: bool = True


DEFAULT_POLICY = ApprovalPolicy()


@dataclass
class ApprovalRequest:
    id: str
    task_id: str
    action: str
    risk_level: str
    reason: str
    context: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | approved | denied
    decided_by: str | None = None
    decided_at: datetime | None = None


class ApprovalGate:
    def __init__(self, policy: ApprovalPolicy | None = None) -> None:
        self._policy = policy or DEFAULT_POLICY
        self._lock = threading.Lock()
        self._pending: dict[str, ApprovalRequest] = {}
        self._events: dict[str, threading.Event] = {}

    @property
    def policy(self) -> ApprovalPolicy:
        return self._policy

    def check(
        self,
        task: SwarmTask,
        action: str,
        context: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        risk_level = self._policy.risk_tiers.get(action, "medium")
        reasons: list[str] = []

        if action in self._policy.require_approval_for_tools:
            reasons.append(f"tool {action!r} requires approval")

        if task.role in self._policy.require_approval_for_roles:
            reasons.append(f"role {task.role!r} requires approval")

        spend = (context or {}).get("spend_usd", 0.0)
        if spend > self._policy.spend_threshold_usd:
            reasons.append(
                f"spend ${spend:.2f} exceeds threshold "
                f"${self._policy.spend_threshold_usd:.2f}"
            )

        if not reasons and self._policy.auto_approve_low_risk and risk_level == "low":
            return ApprovalRequest(
                id=uuid.uuid4().hex[:12],
                task_id=task.id,
                action=action,
                risk_level=risk_level,
                reason="auto-approved (low risk)",
                context=context or {},
                status="approved",
                decided_by="auto",
                decided_at=datetime.utcnow(),
            )

        if not reasons:
            reasons.append(f"risk level {risk_level!r}")

        req = ApprovalRequest(
            id=uuid.uuid4().hex[:12],
            task_id=task.id,
            action=action,
            risk_level=risk_level,
            reason="; ".join(reasons),
            context=context or {},
        )
        return req

    def request_approval(self, req: ApprovalRequest) -> None:
        with self._lock:
            self._pending[req.id] = req
            self._events[req.id] = threading.Event()

    def approve(self, request_id: str, by: str = "user") -> None:
        with self._lock:
            req = self._pending.get(request_id)
            if req is None:
                raise KeyError(f"No pending request {request_id!r}")
            req.status = "approved"
            req.decided_by = by
            req.decided_at = datetime.utcnow()
            event = self._events.get(request_id)
        if event:
            event.set()

    def deny(
        self, request_id: str, by: str = "user", reason: str = ""
    ) -> None:
        with self._lock:
            req = self._pending.get(request_id)
            if req is None:
                raise KeyError(f"No pending request {request_id!r}")
            req.status = "denied"
            req.decided_by = by
            req.decided_at = datetime.utcnow()
            if reason:
                req.reason = reason
            event = self._events.get(request_id)
        if event:
            event.set()

    def get_pending(self) -> list[ApprovalRequest]:
        with self._lock:
            return [
                r for r in self._pending.values() if r.status == "pending"
            ]

    def wait_for_approval(
        self, request_id: str, timeout: float = 300.0
    ) -> bool:
        with self._lock:
            event = self._events.get(request_id)
            req = self._pending.get(request_id)
        if event is None or req is None:
            raise KeyError(f"No pending request {request_id!r}")
        event.wait(timeout=timeout)
        return req.status == "approved"
