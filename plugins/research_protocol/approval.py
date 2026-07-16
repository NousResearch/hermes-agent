"""Exact, per-call workflow approval primitives."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import secrets
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import AwareDatetime, Field, field_validator, model_validator

from .models import BudgetLimits, Capability, ExternalRight
from .models.base import Sha256, StrictContract
from .storage.artifacts import canonical_json_bytes


class ApprovalScope(StrictContract):
    """The immutable capability scope bound to one approval receipt."""

    capability: Capability
    run_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    plan_sha256: Sha256
    input_hashes: tuple[Sha256, ...] = Field(min_length=0)
    budgets: BudgetLimits
    expires_at: AwareDatetime
    max_executions: int = Field(strict=True, gt=0)
    external_rights: tuple[ExternalRight, ...] = ()

    @field_validator("input_hashes", "external_rights", mode="before")
    @classmethod
    def accept_json_arrays(cls, value):
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def canonicalize_collections(self) -> "ApprovalScope":
        object.__setattr__(self, "input_hashes", tuple(sorted(set(self.input_hashes))))
        object.__setattr__(
            self,
            "external_rights",
            tuple(sorted(set(self.external_rights), key=lambda item: item.value)),
        )
        if self.max_executions > self.budgets.max_executions:
            raise ValueError("max_executions cannot exceed budgets.max_executions")
        return self

    def canonical_json(self) -> str:
        """Return the canonical JSON representation used by the database."""
        return canonical_json_bytes(self.model_dump(mode="json")).decode("utf-8")

    def scope_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def scope_hash(self) -> str:
        return self.scope_sha256()


class WorkflowApprovalDecision(str, Enum):
    ACCEPT = "accept"
    DENY = "deny"


class ApprovalVerdict(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"


class ApprovalRecord(StrictContract):
    """Durable approval decision bound to one exact workflow scope."""

    approval_id: str = Field(min_length=22, max_length=256)
    scope_sha256: Sha256
    plan_sha256: Sha256
    scope_json: str = Field(min_length=2)
    verdict: ApprovalVerdict
    surface: str = Field(min_length=1, max_length=128)
    created_at: AwareDatetime
    expires_at: AwareDatetime
    max_executions: int = Field(strict=True, gt=0)
    consumed_count: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def validate_scope_binding(self) -> "ApprovalRecord":
        try:
            scope = ApprovalScope.model_validate_json(self.scope_json, strict=True)
        except Exception as exc:
            raise ValueError("scope_json is not a valid approval scope") from exc
        if scope.canonical_json() != self.scope_json:
            raise ValueError("scope_json must use the canonical representation")
        if scope.scope_sha256() != self.scope_sha256:
            raise ValueError("scope_sha256 does not match scope_json")
        if scope.plan_sha256 != self.plan_sha256:
            raise ValueError("plan_sha256 does not match scope_json")
        if scope.expires_at != self.expires_at:
            raise ValueError("expires_at does not match scope_json")
        if scope.max_executions != self.max_executions:
            raise ValueError("max_executions does not match scope_json")
        if self.consumed_count > self.max_executions:
            raise ValueError("consumed_count exceeds max_executions")
        if (
            self.verdict is ApprovalVerdict.APPROVED
            and self.created_at >= self.expires_at
        ):
            raise ValueError("approved record must expire after creation")
        if self.verdict is ApprovalVerdict.DENIED and self.consumed_count != 0:
            raise ValueError("denied record cannot be consumed")
        return self


class ApprovalService:
    """Orchestrate native consent and durable approval persistence."""

    def __init__(self, store: Any, *, surface: str = "research-protocol"):
        self.store = store
        self.surface = surface

    async def request(
        self, scope: ApprovalScope, summary: str
    ) -> ApprovalRecord | None:
        decision = await asyncio.to_thread(request_workflow_approval, scope, summary)
        verdict = (
            ApprovalVerdict.APPROVED
            if decision is WorkflowApprovalDecision.ACCEPT
            else ApprovalVerdict.DENIED
        )
        record = make_approval_record(scope, surface=self.surface, verdict=verdict)
        await self.store.create_approval(record)
        return record if verdict is ApprovalVerdict.APPROVED else None


def make_approval_id() -> str:
    """Generate an opaque identifier with enough entropy for offline guessing resistance."""
    return secrets.token_urlsafe(32)


def make_approval_record(
    scope: ApprovalScope,
    *,
    surface: str = "research-protocol",
    verdict: ApprovalVerdict = ApprovalVerdict.APPROVED,
) -> ApprovalRecord:
    now = datetime.now(UTC)
    return ApprovalRecord(
        approval_id=make_approval_id(),
        scope_sha256=scope.scope_sha256(),
        plan_sha256=scope.plan_sha256,
        scope_json=scope.canonical_json(),
        verdict=verdict,
        surface=surface,
        created_at=now,
        expires_at=scope.expires_at,
        max_executions=scope.max_executions,
        consumed_count=0,
    )


def _native_consent(message: str, description: str) -> str:
    """Call the Hermes native consent API without importing it at module load."""
    from tools import approval as native_approval

    consent = native_approval.request_elicitation_consent
    try:
        signature = inspect.signature(consent)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("native consent API contract is not inspectable") from exc
    strict_parameter = signature.parameters.get("strict_one_shot")
    if strict_parameter is None or strict_parameter.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        raise RuntimeError("native consent API lacks strict_one_shot support")
    return consent(message, description, strict_one_shot=True)


def request_workflow_approval(
    scope: ApprovalScope, summary: str
) -> WorkflowApprovalDecision:
    """Request exact per-call consent and fail closed on every non-accept path."""
    try:
        validated_scope = ApprovalScope.model_validate(scope)
        if validated_scope.expires_at <= datetime.now(UTC):
            return WorkflowApprovalDecision.DENY
        if not isinstance(summary, str) or not summary.strip():
            return WorkflowApprovalDecision.DENY
        description = (
            "Approve this exact research workflow only once; "
            f"scope_sha256={validated_scope.scope_sha256()}"
        )
        result = _native_consent(summary, description)
    except Exception:
        return WorkflowApprovalDecision.DENY
    return (
        WorkflowApprovalDecision.ACCEPT
        if result == "accept"
        else WorkflowApprovalDecision.DENY
    )


# Convenient aliases for callers that use approval terminology rather than
# workflow terminology.
request_exact_approval = request_workflow_approval


__all__ = [
    "ApprovalRecord",
    "ApprovalScope",
    "ApprovalService",
    "ApprovalVerdict",
    "WorkflowApprovalDecision",
    "make_approval_id",
    "make_approval_record",
    "request_exact_approval",
    "request_workflow_approval",
]
