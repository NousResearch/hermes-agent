"""Read-only recovery completion receipt draft for memory evidence repair.

The recovery completion receipt layer turns a ready recovery executor preview
into a durable-looking receipt draft and token/lock ledger view. It never writes
to durable memory stores; it only describes what a later manual recovery would
record after successful completion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from agent.memory_evidence_repair_recovery_executor_preview import (
    RECOVERY_EXECUTOR_PREVIEW_STATUS_NO_ACTION_NEEDED,
    RECOVERY_EXECUTOR_PREVIEW_STATUS_READY,
)
from agent.memory_evidence_repair_recovery_write_lock_gate import (
    RECOVERY_WRITE_LOCK_SCOPE,
)


RECOVERY_COMPLETION_RECEIPT_TYPE = "memory_evidence_repair_recovery_completion_receipt"
RECOVERY_COMPLETION_RECEIPT_SCOPE = RECOVERY_WRITE_LOCK_SCOPE
RECOVERY_COMPLETION_RECEIPT_STATUS_READY = "recovery_completion_receipt_ready"
RECOVERY_COMPLETION_RECEIPT_STATUS_BLOCKED = "blocked"
RECOVERY_COMPLETION_RECEIPT_STATUS_NO_ACTION_NEEDED = "no_action_needed"


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryCompletionReceipt:
    """One read-only recovery completion receipt draft."""

    id: str
    receipt_type: str
    status: str
    scope: str
    actor: str
    recovery_reason: str
    token_id: str
    lock_id: str
    decision_id: str
    execution_preview_id: str
    executor_preview_id: str
    route: str
    executor_preview_digest: str
    receipt_digest: str
    receipt_preview: str
    operation_ids: tuple[str, ...]
    recovery_step_ids: tuple[str, ...]
    future_mutation_step_ids: tuple[str, ...]
    executor_step_ids: tuple[str, ...]
    used_recovery_token_id: str
    released_recovery_lock_id: str
    would_record_receipt: bool
    would_mark_recovery_token_used: bool
    would_release_recovery_write_lock: bool
    would_trigger_recovery_audit: bool
    safety_note: str
    source_recovery_executor_preview: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "receipt_type": self.receipt_type,
            "status": self.status,
            "scope": self.scope,
            "actor": self.actor,
            "recovery_reason": self.recovery_reason,
            "token_id": self.token_id,
            "lock_id": self.lock_id,
            "decision_id": self.decision_id,
            "execution_preview_id": self.execution_preview_id,
            "executor_preview_id": self.executor_preview_id,
            "route": self.route,
            "executor_preview_digest": self.executor_preview_digest,
            "receipt_digest": self.receipt_digest,
            "receipt_preview": self.receipt_preview,
            "operation_ids": list(self.operation_ids),
            "recovery_step_ids": list(self.recovery_step_ids),
            "future_mutation_step_ids": list(self.future_mutation_step_ids),
            "executor_step_ids": list(self.executor_step_ids),
            "used_recovery_token_id": self.used_recovery_token_id,
            "released_recovery_lock_id": self.released_recovery_lock_id,
            "would_record_receipt": self.would_record_receipt,
            "would_mark_recovery_token_used": self.would_mark_recovery_token_used,
            "would_release_recovery_write_lock": self.would_release_recovery_write_lock,
            "would_trigger_recovery_audit": self.would_trigger_recovery_audit,
            "safety_note": self.safety_note,
            "source_recovery_executor_preview": dict(self.source_recovery_executor_preview),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryCompletionReceiptReport:
    """Read-only recovery completion receipt report."""

    generated_at: str
    status: str
    receipts: tuple[MemoryEvidenceRepairRecoveryCompletionReceipt, ...]
    used_recovery_token_ids: tuple[str, ...]
    released_recovery_lock_ids: tuple[str, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_recovery_executor_preview_report: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "receipt_count": len(self.receipts),
            "used_recovery_token_count": len(self.used_recovery_token_ids),
            "released_recovery_lock_count": len(self.released_recovery_lock_ids),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "recovery_completion_receipt_available": bool(self.receipts),
            "would_record_receipt_count": sum(
                1 for receipt in self.receipts if receipt.would_record_receipt
            ),
            "would_mark_recovery_token_used_count": sum(
                1 for receipt in self.receipts if receipt.would_mark_recovery_token_used
            ),
            "would_release_recovery_write_lock_count": sum(
                1 for receipt in self.receipts if receipt.would_release_recovery_write_lock
            ),
            "would_trigger_recovery_audit_count": sum(
                1 for receipt in self.receipts if receipt.would_trigger_recovery_audit
            ),
            "has_blocks": self.status == RECOVERY_COMPLETION_RECEIPT_STATUS_BLOCKED,
            "requires_followup": bool(self.required_actions),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "receipts": [receipt.to_dict() for receipt in self.receipts],
            "used_recovery_token_ids": list(self.used_recovery_token_ids),
            "released_recovery_lock_ids": list(self.released_recovery_lock_ids),
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "source_recovery_executor_preview_report": dict(
                self.source_recovery_executor_preview_report
            ),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_completion_receipt(
    *,
    recovery_executor_preview: Mapping[str, Any] | None = None,
    existing_receipts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    used_token_ids: Sequence[str] | None = None,
    released_lock_ids: Sequence[str] | None = None,
    actor: str = "human",
    recovery_reason: str = "",
) -> MemoryEvidenceRepairRecoveryCompletionReceiptReport:
    """Build a read-only recovery completion receipt from an executor preview."""

    recovery_executor_preview = (
        recovery_executor_preview if isinstance(recovery_executor_preview, Mapping) else {}
    )
    report_status = _str(recovery_executor_preview.get("status"))
    preview = _extract_preview(recovery_executor_preview)
    existing_used = _used_recovery_token_ids(existing_receipts, used_token_ids)
    existing_released = _released_recovery_lock_ids(existing_receipts, released_lock_ids)

    if not preview:
        if (
            not recovery_executor_preview
            or report_status == RECOVERY_EXECUTOR_PREVIEW_STATUS_NO_ACTION_NEEDED
        ):
            return _empty_report(
                recovery_executor_preview=recovery_executor_preview,
                used_recovery_token_ids=existing_used,
                released_recovery_lock_ids=existing_released,
            )
        source_blocking = tuple(_list_of_str(recovery_executor_preview.get("blocking_reasons")))
        source_actions = tuple(_list_of_str(recovery_executor_preview.get("required_actions")))
        return _blocked_report(
            recovery_executor_preview=recovery_executor_preview,
            used_recovery_token_ids=existing_used,
            released_recovery_lock_ids=existing_released,
            blocking_reasons=source_blocking
            or ("Recovery executor preview is not ready for completion receipt.",),
            required_actions=source_actions or ("produce_recovery_executor_preview",),
        )

    validation_error = _preview_validation_error(recovery_executor_preview, preview)
    if validation_error is not None:
        return _blocked_report(
            recovery_executor_preview=recovery_executor_preview,
            used_recovery_token_ids=existing_used,
            released_recovery_lock_ids=existing_released,
            blocking_reasons=(validation_error,),
            required_actions=("regenerate_recovery_executor_preview",),
        )

    token_id = _str(preview.get("token_id"))
    lock_id = _str(preview.get("lock_id"))
    if token_id in set(existing_used):
        return _blocked_report(
            recovery_executor_preview=recovery_executor_preview,
            used_recovery_token_ids=existing_used,
            released_recovery_lock_ids=existing_released,
            blocking_reasons=(
                "Recovery approval token is already present in the used-token ledger.",
            ),
            required_actions=("regenerate_recovery_human_approval_token",),
        )
    if lock_id in set(existing_released):
        return _blocked_report(
            recovery_executor_preview=recovery_executor_preview,
            used_recovery_token_ids=existing_used,
            released_recovery_lock_ids=existing_released,
            blocking_reasons=(
                "Recovery write lock is already present in the released-lock ledger.",
            ),
            required_actions=("regenerate_recovery_write_lock_gate",),
        )

    receipt = _receipt_from_preview(
        preview=preview,
        actor=actor,
        recovery_reason=recovery_reason,
    )
    return MemoryEvidenceRepairRecoveryCompletionReceiptReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_COMPLETION_RECEIPT_STATUS_READY,
        receipts=(receipt,),
        used_recovery_token_ids=tuple(
            _dedupe_strings([*existing_used, receipt.used_recovery_token_id])
        ),
        released_recovery_lock_ids=tuple(
            _dedupe_strings([*existing_released, receipt.released_recovery_lock_id])
        ),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_executor_preview_report=dict(recovery_executor_preview),
    )


def empty_evidence_repair_recovery_completion_receipt() -> MemoryEvidenceRepairRecoveryCompletionReceiptReport:
    """Return an empty read-only recovery completion receipt report."""

    return _empty_report(
        recovery_executor_preview={},
        used_recovery_token_ids=(),
        released_recovery_lock_ids=(),
    )


def _receipt_from_preview(
    *,
    preview: Mapping[str, Any],
    actor: str,
    recovery_reason: str,
) -> MemoryEvidenceRepairRecoveryCompletionReceipt:
    operation_ids = tuple(_list_of_str(preview.get("operation_ids")))
    recovery_step_ids = tuple(_list_of_str(preview.get("step_ids")))
    future_mutation_step_ids = tuple(_list_of_str(preview.get("future_mutation_step_ids")))
    executor_step_ids = tuple(_list_of_str(preview.get("executor_step_ids")))
    token_id = _str(preview.get("token_id"))
    lock_id = _str(preview.get("lock_id"))
    receipt_seed = {
        "receipt_type": RECOVERY_COMPLETION_RECEIPT_TYPE,
        "scope": RECOVERY_COMPLETION_RECEIPT_SCOPE,
        "actor": _str(actor) or "human",
        "recovery_reason": _str(recovery_reason),
        "token_id": token_id,
        "lock_id": lock_id,
        "decision_id": _str(preview.get("decision_id")),
        "execution_preview_id": _str(preview.get("execution_preview_id")),
        "executor_preview_id": _str(preview.get("id")),
        "route": _str(preview.get("route")),
        "operation_ids": operation_ids,
        "recovery_step_ids": recovery_step_ids,
        "future_mutation_step_ids": future_mutation_step_ids,
        "executor_step_ids": executor_step_ids,
        "executor_preview_digest": _str(preview.get("preview_digest")),
        "preview_status": _str(preview.get("status")),
    }
    receipt_digest = _digest(receipt_seed)
    receipt_id = f"recovery-completion-receipt-{receipt_digest[:16]}"
    return MemoryEvidenceRepairRecoveryCompletionReceipt(
        id=receipt_id,
        receipt_type=RECOVERY_COMPLETION_RECEIPT_TYPE,
        status=RECOVERY_COMPLETION_RECEIPT_STATUS_READY,
        scope=RECOVERY_COMPLETION_RECEIPT_SCOPE,
        actor=_str(actor) or "human",
        recovery_reason=_str(recovery_reason),
        token_id=token_id,
        lock_id=lock_id,
        decision_id=_str(preview.get("decision_id")),
        execution_preview_id=_str(preview.get("execution_preview_id")),
        executor_preview_id=_str(preview.get("id")),
        route=_str(preview.get("route")),
        executor_preview_digest=_str(preview.get("preview_digest")),
        receipt_digest=receipt_digest,
        receipt_preview=f"{receipt_id}:{receipt_digest[:12]}",
        operation_ids=operation_ids,
        recovery_step_ids=recovery_step_ids,
        future_mutation_step_ids=future_mutation_step_ids,
        executor_step_ids=executor_step_ids,
        used_recovery_token_id=token_id,
        released_recovery_lock_id=lock_id,
        would_record_receipt=True,
        would_mark_recovery_token_used=True,
        would_release_recovery_write_lock=True,
        would_trigger_recovery_audit=True,
        safety_note=(
            "Read-only recovery completion receipt draft. It previews a future "
            "completion receipt, used-token mark, released-lock mark, and audit "
            "trigger, but does not persist any of them."
        ),
        source_recovery_executor_preview=dict(preview),
    )


def _preview_validation_error(
    recovery_executor_preview: Mapping[str, Any],
    preview: Mapping[str, Any],
) -> str | None:
    report_status = _str(recovery_executor_preview.get("status"))
    preview_status = _str(preview.get("status"))
    if report_status and report_status != RECOVERY_EXECUTOR_PREVIEW_STATUS_READY:
        blocking = _list_of_str(recovery_executor_preview.get("blocking_reasons"))
        return (
            "; ".join(blocking)
            if blocking
            else "Recovery executor preview report is not ready."
        )
    if preview_status != RECOVERY_EXECUTOR_PREVIEW_STATUS_READY:
        return "Recovery executor preview entry is not ready."

    missing_fields: list[str] = []
    for field_name in (
        "id",
        "lock_id",
        "token_id",
        "decision_id",
        "execution_preview_id",
        "operation_ids",
        "step_ids",
        "future_mutation_step_ids",
        "executor_step_ids",
        "preview_digest",
    ):
        value = preview.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, list) and not _str(value):
            missing_fields.append(field_name)
    expected_digest = _expected_executor_preview_digest(preview)
    actual_digest = _str(preview.get("preview_digest"))
    expected_id = f"recovery-executor-preview-{expected_digest[:16]}" if expected_digest else ""
    if missing_fields:
        return (
            "Recovery executor preview is missing required fields: "
            + ", ".join(missing_fields)
        )
    if actual_digest != expected_digest or _str(preview.get("id")) != expected_id:
        return "Recovery executor preview digest or id is invalid."
    if not preview.get("would_execute_manual_recovery"):
        return "Recovery executor preview does not indicate future manual recovery execution."
    if not preview.get("would_release_recovery_write_lock"):
        return "Recovery executor preview does not release the future recovery write lock."
    if not preview.get("would_mark_recovery_token_used"):
        return "Recovery executor preview does not mark the future recovery token used."
    return None


def _expected_executor_preview_digest(preview: Mapping[str, Any]) -> str:
    seed = {
        "lock_id": _str(preview.get("lock_id")),
        "token_id": _str(preview.get("token_id")),
        "decision_id": _str(preview.get("decision_id")),
        "execution_preview_id": _str(preview.get("execution_preview_id")),
        "operation_ids": tuple(_list_of_str(preview.get("operation_ids"))),
        "step_ids": tuple(_list_of_str(preview.get("step_ids"))),
        "future_mutation_step_ids": tuple(_list_of_str(preview.get("future_mutation_step_ids"))),
        "executor_step_ids": tuple(_list_of_str(preview.get("executor_step_ids"))),
        "execution_mode": _str(preview.get("execution_mode")),
        "failure_policy": _str(preview.get("failure_policy")),
    }
    return _digest(seed)


def _extract_preview(recovery_executor_preview: Mapping[str, Any]) -> dict[str, Any]:
    previews = recovery_executor_preview.get("previews")
    if isinstance(previews, list):
        for preview in previews:
            if isinstance(preview, Mapping):
                return dict(preview)
    if _str(recovery_executor_preview.get("id")).startswith("recovery-executor-preview-"):
        return dict(recovery_executor_preview)
    return {}


def _empty_report(
    *,
    recovery_executor_preview: Mapping[str, Any],
    used_recovery_token_ids: Sequence[str],
    released_recovery_lock_ids: Sequence[str],
) -> MemoryEvidenceRepairRecoveryCompletionReceiptReport:
    return MemoryEvidenceRepairRecoveryCompletionReceiptReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_COMPLETION_RECEIPT_STATUS_NO_ACTION_NEEDED,
        receipts=(),
        used_recovery_token_ids=tuple(_dedupe_strings(used_recovery_token_ids)),
        released_recovery_lock_ids=tuple(_dedupe_strings(released_recovery_lock_ids)),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_executor_preview_report=dict(recovery_executor_preview),
    )


def _blocked_report(
    *,
    recovery_executor_preview: Mapping[str, Any],
    used_recovery_token_ids: Sequence[str],
    released_recovery_lock_ids: Sequence[str],
    blocking_reasons: Sequence[str],
    required_actions: Sequence[str],
) -> MemoryEvidenceRepairRecoveryCompletionReceiptReport:
    return MemoryEvidenceRepairRecoveryCompletionReceiptReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_COMPLETION_RECEIPT_STATUS_BLOCKED,
        receipts=(),
        used_recovery_token_ids=tuple(_dedupe_strings(used_recovery_token_ids)),
        released_recovery_lock_ids=tuple(_dedupe_strings(released_recovery_lock_ids)),
        blocking_reasons=tuple(_dedupe_strings(blocking_reasons)),
        required_actions=tuple(_dedupe_strings(required_actions)),
        source_recovery_executor_preview_report=dict(recovery_executor_preview),
    )


def _used_recovery_token_ids(
    existing_receipts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    explicit_used_token_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    candidates: list[str] = _list_of_str(explicit_used_token_ids)
    if isinstance(existing_receipts, Mapping):
        candidates.extend(_list_of_str(existing_receipts.get("used_recovery_token_ids")))
        candidates.extend(_list_of_str(existing_receipts.get("used_token_ids")))
        receipt_rows = existing_receipts.get("receipts")
        if isinstance(receipt_rows, list):
            candidates.extend(_token_ids_from_receipts(receipt_rows))
        else:
            candidates.extend(_token_ids_from_receipts([existing_receipts]))
    elif isinstance(existing_receipts, list):
        candidates.extend(_token_ids_from_receipts(existing_receipts))
    return tuple(_dedupe_strings(candidates))


def _released_recovery_lock_ids(
    existing_receipts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    explicit_released_lock_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    candidates: list[str] = _list_of_str(explicit_released_lock_ids)
    if isinstance(existing_receipts, Mapping):
        candidates.extend(_list_of_str(existing_receipts.get("released_recovery_lock_ids")))
        candidates.extend(_list_of_str(existing_receipts.get("released_lock_ids")))
        receipt_rows = existing_receipts.get("receipts")
        if isinstance(receipt_rows, list):
            candidates.extend(_lock_ids_from_receipts(receipt_rows))
        else:
            candidates.extend(_lock_ids_from_receipts([existing_receipts]))
    elif isinstance(existing_receipts, list):
        candidates.extend(_lock_ids_from_receipts(existing_receipts))
    return tuple(_dedupe_strings(candidates))


def _token_ids_from_receipts(receipts: Sequence[Any]) -> list[str]:
    token_ids: list[str] = []
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            continue
        token_id = (
            _str(receipt.get("used_recovery_token_id"))
            or _str(receipt.get("used_token_id"))
            or _str(receipt.get("token_id"))
        )
        if token_id:
            token_ids.append(token_id)
    return token_ids


def _lock_ids_from_receipts(receipts: Sequence[Any]) -> list[str]:
    lock_ids: list[str] = []
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            continue
        lock_id = _str(receipt.get("released_recovery_lock_id")) or _str(receipt.get("lock_id"))
        if lock_id:
            lock_ids.append(lock_id)
    return lock_ids


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _list_of_str(value: Any) -> list[str]:
    return [_str(item) for item in value] if isinstance(value, list) else []


def _dedupe_strings(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _str(value: Any) -> str:
    return str(value or "").strip()
