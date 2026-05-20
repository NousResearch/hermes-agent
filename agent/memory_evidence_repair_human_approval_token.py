"""Read-only human approval token draft for memory evidence repair.

The token layer creates an approval-token draft only after a manual commit
dry-run is ready. It never writes to durable memory stores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from agent.memory_evidence_repair_manual_commit_dry_run import (
    DRY_RUN_STATUS_BLOCKED,
    DRY_RUN_STATUS_NO_ACTION_NEEDED,
    DRY_RUN_STATUS_READY_FOR_MANUAL_COMMIT,
)


APPROVAL_TOKEN_TYPE = "memory_evidence_repair_human_approval"
APPROVAL_TOKEN_SCOPE = "manual_memory_evidence_repair_commit"
TOKEN_STATUS_DRAFT_READY = "draft_ready_for_human_approval"
TOKEN_STATUS_BLOCKED = "blocked"
TOKEN_STATUS_NO_ACTION_NEEDED = "no_action_needed"
DEFAULT_EXPIRES_IN_MINUTES = 30


@dataclass(frozen=True)
class MemoryEvidenceRepairHumanApprovalToken:
    """One read-only human approval token draft."""

    id: str
    token_type: str
    status: str
    scope: str
    approver: str
    approval_reason: str
    dry_run_status: str
    dry_run_digest: str
    token_digest: str
    token_preview: str
    expires_in_minutes: int
    operation_ids: tuple[str, ...]
    ledger_entry_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    patch_digests: tuple[str, ...]
    required_confirmation_text: str
    one_time_constraints: dict[str, Any]
    safety_note: str
    source_dry_run: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "token_type": self.token_type,
            "status": self.status,
            "scope": self.scope,
            "approver": self.approver,
            "approval_reason": self.approval_reason,
            "dry_run_status": self.dry_run_status,
            "dry_run_digest": self.dry_run_digest,
            "token_digest": self.token_digest,
            "token_preview": self.token_preview,
            "expires_in_minutes": self.expires_in_minutes,
            "operation_ids": list(self.operation_ids),
            "ledger_entry_ids": list(self.ledger_entry_ids),
            "candidate_ids": list(self.candidate_ids),
            "patch_digests": list(self.patch_digests),
            "required_confirmation_text": self.required_confirmation_text,
            "one_time_constraints": dict(self.one_time_constraints),
            "safety_note": self.safety_note,
            "source_dry_run": dict(self.source_dry_run),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairHumanApprovalTokenReport:
    """Read-only human approval token draft report."""

    generated_at: str
    status: str
    tokens: tuple[MemoryEvidenceRepairHumanApprovalToken, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "token_count": len(self.tokens),
            "ready_count": 1 if self.status == TOKEN_STATUS_DRAFT_READY else 0,
            "blocked_count": 1 if self.status == TOKEN_STATUS_BLOCKED else 0,
            "no_action_count": 1 if self.status == TOKEN_STATUS_NO_ACTION_NEEDED else 0,
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "human_approval_token_available": bool(self.tokens),
            "requires_followup": bool(self.required_actions),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "tokens": [token.to_dict() for token in self.tokens],
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_human_approval_token(
    *,
    dry_run: Mapping[str, Any] | None = None,
    approver: str = "human",
    approval_reason: str = "",
    expires_in_minutes: int = DEFAULT_EXPIRES_IN_MINUTES,
) -> MemoryEvidenceRepairHumanApprovalTokenReport:
    """Build a read-only human approval token draft from a dry-run."""

    dry_run = dry_run if isinstance(dry_run, Mapping) else {}
    dry_run_status = _str(dry_run.get("status"))
    operations = _operation_rows(dry_run)
    if dry_run_status == DRY_RUN_STATUS_READY_FOR_MANUAL_COMMIT and operations:
        token = _token_from_dry_run(
            dry_run=dry_run,
            approver=approver,
            approval_reason=approval_reason,
            expires_in_minutes=expires_in_minutes,
        )
        return MemoryEvidenceRepairHumanApprovalTokenReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=TOKEN_STATUS_DRAFT_READY,
            tokens=(token,),
            blocking_reasons=(),
            required_actions=(),
        )

    blocking_reasons = tuple(_list_of_str(dry_run.get("blocking_reasons")))
    required_actions = tuple(_list_of_str(dry_run.get("required_actions")))
    if dry_run_status == DRY_RUN_STATUS_BLOCKED:
        status = TOKEN_STATUS_BLOCKED
        if not blocking_reasons:
            blocking_reasons = ("Manual commit dry-run is blocked.",)
        if not required_actions:
            required_actions = ("resolve_manual_commit_dry_run_blocks",)
    elif dry_run_status == DRY_RUN_STATUS_NO_ACTION_NEEDED or not operations:
        status = TOKEN_STATUS_NO_ACTION_NEEDED
    else:
        status = TOKEN_STATUS_BLOCKED
        blocking_reasons = blocking_reasons or ("Manual commit dry-run is not ready.",)
        required_actions = required_actions or ("produce_ready_manual_commit_dry_run",)

    return MemoryEvidenceRepairHumanApprovalTokenReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        tokens=(),
        blocking_reasons=blocking_reasons,
        required_actions=required_actions,
    )


def empty_evidence_repair_human_approval_token() -> MemoryEvidenceRepairHumanApprovalTokenReport:
    """Return an empty read-only human approval token report."""

    return MemoryEvidenceRepairHumanApprovalTokenReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=TOKEN_STATUS_NO_ACTION_NEEDED,
        tokens=(),
        blocking_reasons=(),
        required_actions=(),
    )


def _token_from_dry_run(
    *,
    dry_run: Mapping[str, Any],
    approver: str,
    approval_reason: str,
    expires_in_minutes: int,
) -> MemoryEvidenceRepairHumanApprovalToken:
    operations = _operation_rows(dry_run)
    dry_run_digest = _digest(
        {
            "status": dry_run.get("status"),
            "operations": operations,
            "checks": dry_run.get("checks"),
        }
    )
    operation_ids = tuple(_str(operation.get("id")) for operation in operations)
    ledger_entry_ids = tuple(_str(operation.get("ledger_entry_id")) for operation in operations)
    candidate_ids = tuple(_str(operation.get("candidate_id")) for operation in operations)
    patch_digests = tuple(_str(operation.get("patch_digest")) for operation in operations)
    token_seed = {
        "token_type": APPROVAL_TOKEN_TYPE,
        "scope": APPROVAL_TOKEN_SCOPE,
        "approver": _str(approver) or "human",
        "approval_reason": _str(approval_reason),
        "dry_run_digest": dry_run_digest,
        "operation_ids": operation_ids,
        "ledger_entry_ids": ledger_entry_ids,
        "candidate_ids": candidate_ids,
        "patch_digests": patch_digests,
        "expires_in_minutes": _expires_in_minutes(expires_in_minutes),
    }
    token_digest = _digest(token_seed)
    token_id = f"approval-token-{token_digest[:16]}"
    required_confirmation_text = (
        f"APPROVE {APPROVAL_TOKEN_SCOPE} {token_id}"
    )
    return MemoryEvidenceRepairHumanApprovalToken(
        id=token_id,
        token_type=APPROVAL_TOKEN_TYPE,
        status=TOKEN_STATUS_DRAFT_READY,
        scope=APPROVAL_TOKEN_SCOPE,
        approver=_str(approver) or "human",
        approval_reason=_str(approval_reason),
        dry_run_status=_str(dry_run.get("status")),
        dry_run_digest=dry_run_digest,
        token_digest=token_digest,
        token_preview=f"{token_id}:{token_digest[:12]}",
        expires_in_minutes=_expires_in_minutes(expires_in_minutes),
        operation_ids=operation_ids,
        ledger_entry_ids=ledger_entry_ids,
        candidate_ids=candidate_ids,
        patch_digests=patch_digests,
        required_confirmation_text=required_confirmation_text,
        one_time_constraints={
            "one_time_use": True,
            "requires_exact_dry_run_digest": dry_run_digest,
            "requires_exact_confirmation_text": required_confirmation_text,
            "requires_same_patch_digests": list(patch_digests),
            "expires_in_minutes": _expires_in_minutes(expires_in_minutes),
        },
        safety_note=(
            "Read-only approval token draft. This is not a persisted credential "
            "and does not authorize or mutate durable memory by itself."
        ),
        source_dry_run=dict(dry_run),
    )


def _operation_rows(dry_run: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        operation
        for operation in _list(dry_run.get("operations"))
        if isinstance(operation, Mapping)
    ]


def _expires_in_minutes(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_EXPIRES_IN_MINUTES
    return parsed if parsed > 0 else DEFAULT_EXPIRES_IN_MINUTES


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _list_of_str(value: Any) -> list[str]:
    return [_str(item) for item in value] if isinstance(value, list) else []


def _str(value: Any) -> str:
    return str(value or "").strip()
