"""Risk policy and exact-operation approval for Beta."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_HIGH_ACTION = re.compile(
    r"\b(restart|reinici|deploy|terminate|encerr|kill|delete|exclu|drop|truncate|firewall|permission|permiss|production|producao|reboot|shutdown|format|wipe|rotate.?key|create.?user|remove.?user|alter.?table|update\s+\w+\s+set)\w*\b",
    re.IGNORECASE,
)
_MEDIUM_ACTION = re.compile(
    r"\b(prepare|prepar|script|configure|configur|edit|write|patch|simulat|dry-run|generate|gerar|install|instal)\w*\b",
    re.IGNORECASE,
)
_READ_ONLY = re.compile(
    r"\b(read|list|show|describe|inspect|diagnos|query|select|status|metric|log|explain|analy[sz]|verific|consult)\w*\b",
    re.IGNORECASE,
)
_HIGH_TOOLS = frozenset({
    "send_message", "ha_call_service", "terminal", "execute_command", "computer_use",
    "delete_file", "move_file", "write_file", "patch", "cronjob",
})
_MEDIUM_TOOLS = frozenset({"skill_manage", "create_file", "git_commit", "git_push"})


class RiskContext(BaseModel):
    """Structured evidence used before falling back to text heuristics."""

    model_config = ConfigDict(frozen=True)

    tool_name: str = ""
    read_only: bool = False
    changes_state: bool = False
    production: bool = False
    destructive: bool = False
    security_sensitive: bool = False
    financial: bool = False
    externally_visible: bool = False


def classify_risk(
    action: str,
    tool_name: str = "",
    *,
    context: RiskContext | Mapping[str, Any] | None = None,
) -> RiskLevel:
    """Classify using structured metadata first and conservative text fallback."""
    if context is not None and not isinstance(context, RiskContext):
        context = RiskContext.model_validate(context)
    if context is not None:
        effective_tool = context.tool_name or tool_name
        if any((
            context.production,
            context.destructive,
            context.security_sensitive,
            context.financial,
            context.externally_visible,
        )):
            return RiskLevel.HIGH
        if context.changes_state:
            return RiskLevel.HIGH if effective_tool in _HIGH_TOOLS else RiskLevel.MEDIUM
        if context.read_only:
            return RiskLevel.LOW
        tool_name = effective_tool

    if tool_name in _HIGH_TOOLS or _HIGH_ACTION.search(action):
        return RiskLevel.HIGH
    if tool_name in _MEDIUM_TOOLS or _MEDIUM_ACTION.search(action):
        return RiskLevel.MEDIUM
    if _READ_ONLY.search(action):
        return RiskLevel.LOW
    return RiskLevel.LOW


class Operation(BaseModel):
    """Exact approval scope shown to the Chief."""

    model_config = ConfigDict(frozen=True)

    target: str
    action: str
    impact: str
    rollback: str
    risk: RiskLevel
    tool_name: str = ""
    arguments_digest: str = ""

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def approval_message(self) -> str:
        return (
            f"Target: {self.target}\nAction: {self.action}\nImpact: {self.impact}\n"
            f"Rollback: {self.rollback}\nRisk: {self.risk.value}"
        )


class ApprovalReceipt(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation_fingerprint: str
    issued_at: float
    expires_at: float
    approved_by: str = "chief"
    nonce: str


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
            self._events.append(ApprovalAuditEvent(
                timestamp=self._clock(), event=event,
                operation_fingerprint=operation.fingerprint,
            ))

    def request(self, operation: Operation, *, ttl_seconds: int = 300) -> ApprovalReceipt | None:
        if operation.risk != RiskLevel.HIGH:
            return None
        self._record("requested", operation)
        decision = self._requester(operation)
        if decision.get("approved") is not True:
            self._record("denied", operation)
            return None
        now = self._clock()
        receipt = ApprovalReceipt(
            operation_fingerprint=operation.fingerprint,
            issued_at=now,
            expires_at=now + max(1, ttl_seconds),
            approved_by=str(decision.get("approved_by") or "chief"),
            nonce=uuid.uuid4().hex,
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
