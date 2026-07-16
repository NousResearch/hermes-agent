"""Risk classification and exact-operation approval for Beta."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_HIGH_ACTION = re.compile(
    r"\b(restart|reinici|deploy|terminate|encerr|kill|delete|exclu|drop|firewall|permission|permiss|production|producao)\w*\b",
    re.IGNORECASE,
)
_MEDIUM_ACTION = re.compile(
    r"\b(prepare|prepar|script|configure|configur|edit|write|patch|simulat|dry-run)\w*\b",
    re.IGNORECASE,
)
_HIGH_TOOLS = frozenset({"send_message", "ha_call_service"})
_MEDIUM_TOOLS = frozenset({"write_file", "patch", "skill_manage", "cronjob"})


def classify_risk(action: str, tool_name: str = "") -> RiskLevel:
    """Classify a concrete operation; read-only work remains low risk."""
    if tool_name in _HIGH_TOOLS or _HIGH_ACTION.search(action):
        return RiskLevel.HIGH
    if tool_name in _MEDIUM_TOOLS or _MEDIUM_ACTION.search(action):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


class Operation(BaseModel):
    """Exact approval scope shown to the Chief."""

    model_config = ConfigDict(frozen=True)

    target: str
    action: str
    impact: str
    rollback: str
    risk: RiskLevel

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def approval_message(self) -> str:
        return (
            f"Target: {self.target}\nAction: {self.action}\nImpact: {self.impact}\n"
            f"Rollback: {self.rollback}"
        )


class ApprovalReceipt(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation_fingerprint: str
    expires_at: float


class ApprovalAuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: float
    event: str
    operation_fingerprint: str


def _hermes_request(operation: Operation) -> dict:
    from tools.approval import is_approval_bypass_active, request_tool_approval
    from utils import env_var_enabled

    if is_approval_bypass_active() or env_var_enabled("HERMES_CRON_SESSION"):
        return {"approved": False, "message": "Beta requires explicit approval; bypass mode is ignored."}
    return request_tool_approval(
        "beta_operation",
        operation.approval_message(),
        # A nonce prevents Hermes' session/permanent cache from silently
        # renewing an expired Beta receipt without another human decision.
        rule_key=f"beta:{operation.fingerprint}:{uuid.uuid4().hex}",
    )


class ApprovalGate:
    """Issues short-lived receipts for one exact high-risk operation."""

    def __init__(
        self,
        *,
        requester: Callable[[Operation], dict] = _hermes_request,
        clock: Callable[[], float] = time.time,
    ):
        self._requester = requester
        self._clock = clock
        self._events: list[ApprovalAuditEvent] = []
        self._lock = threading.Lock()

    @property
    def events(self) -> tuple[ApprovalAuditEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def _record(self, event: str, operation: Operation) -> None:
        with self._lock:
            self._events.append(
                ApprovalAuditEvent(
                    timestamp=self._clock(),
                    event=event,
                    operation_fingerprint=operation.fingerprint,
                )
            )

    def request(self, operation: Operation, *, ttl_seconds: int = 300) -> ApprovalReceipt | None:
        if operation.risk != RiskLevel.HIGH:
            return None
        self._record("requested", operation)
        decision = self._requester(operation)
        if decision.get("approved") is not True:
            self._record("denied", operation)
            return None
        receipt = ApprovalReceipt(
            operation_fingerprint=operation.fingerprint,
            expires_at=self._clock() + max(1, ttl_seconds),
        )
        self._record("approved", operation)
        return receipt

    def authorized(self, operation: Operation, receipt: ApprovalReceipt | None = None) -> bool:
        if operation.risk != RiskLevel.HIGH:
            return True
        if receipt is None or receipt.operation_fingerprint != operation.fingerprint:
            self._record("blocked", operation)
            return False
        if receipt.expires_at <= self._clock():
            self._record("expired", operation)
            return False
        self._record("authorized", operation)
        return True
