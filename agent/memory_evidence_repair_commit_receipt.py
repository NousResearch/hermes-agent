"""Read-only manual commit receipt ledger for memory evidence repair.

The receipt layer turns a verified approval-token gate into a durable-looking
receipt draft and a used-token ledger view. It never writes to durable memory
stores; it only describes what a later manual commit would record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from agent.memory_evidence_repair_approval_token_gate import (
    TOKEN_GATE_STATUS_BLOCKED,
    TOKEN_GATE_STATUS_NO_ACTION_NEEDED,
    TOKEN_GATE_STATUS_VERIFIED,
)


RECEIPT_TYPE = "memory_evidence_repair_manual_commit_receipt"
RECEIPT_SCOPE = "manual_memory_evidence_repair_commit"
RECEIPT_STATUS_READY = "receipt_ready_for_manual_commit"
RECEIPT_STATUS_BLOCKED = "blocked"
RECEIPT_STATUS_NO_ACTION_NEEDED = "no_action_needed"


@dataclass(frozen=True)
class MemoryEvidenceRepairCommitReceipt:
    """One read-only manual commit receipt draft."""

    id: str
    receipt_type: str
    status: str
    scope: str
    actor: str
    commit_reason: str
    token_id: str
    token_digest: str
    dry_run_digest: str
    gate_status: str
    receipt_digest: str
    receipt_preview: str
    operation_ids: tuple[str, ...]
    ledger_entry_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    patch_digests: tuple[str, ...]
    used_token_id: str
    would_record_receipt: bool
    would_mark_token_used: bool
    safety_note: str
    source_gate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "receipt_type": self.receipt_type,
            "status": self.status,
            "scope": self.scope,
            "actor": self.actor,
            "commit_reason": self.commit_reason,
            "token_id": self.token_id,
            "token_digest": self.token_digest,
            "dry_run_digest": self.dry_run_digest,
            "gate_status": self.gate_status,
            "receipt_digest": self.receipt_digest,
            "receipt_preview": self.receipt_preview,
            "operation_ids": list(self.operation_ids),
            "ledger_entry_ids": list(self.ledger_entry_ids),
            "candidate_ids": list(self.candidate_ids),
            "patch_digests": list(self.patch_digests),
            "used_token_id": self.used_token_id,
            "would_record_receipt": self.would_record_receipt,
            "would_mark_token_used": self.would_mark_token_used,
            "safety_note": self.safety_note,
            "source_gate": dict(self.source_gate),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairCommitReceiptReport:
    """Read-only manual commit receipt report."""

    generated_at: str
    status: str
    receipts: tuple[MemoryEvidenceRepairCommitReceipt, ...]
    used_token_ids: tuple[str, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_gate: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "receipt_count": len(self.receipts),
            "used_token_count": len(self.used_token_ids),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "commit_receipt_available": bool(self.receipts),
            "would_record_receipt_count": sum(
                1 for receipt in self.receipts if receipt.would_record_receipt
            ),
            "would_mark_token_used_count": sum(
                1 for receipt in self.receipts if receipt.would_mark_token_used
            ),
            "has_blocks": self.status == RECEIPT_STATUS_BLOCKED,
            "requires_followup": bool(self.required_actions),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "receipts": [receipt.to_dict() for receipt in self.receipts],
            "used_token_ids": list(self.used_token_ids),
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "source_gate": dict(self.source_gate),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_commit_receipt(
    *,
    token_gate: Mapping[str, Any] | None = None,
    existing_receipts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    used_token_ids: Sequence[str] | None = None,
    actor: str = "human",
    commit_reason: str = "",
) -> MemoryEvidenceRepairCommitReceiptReport:
    """Build a read-only receipt draft from a verified approval-token gate."""

    token_gate = token_gate if isinstance(token_gate, Mapping) else {}
    gate_status = _str(token_gate.get("status"))
    existing_used = _used_token_ids(existing_receipts, used_token_ids)

    if not token_gate or gate_status == TOKEN_GATE_STATUS_NO_ACTION_NEEDED:
        return _empty_report(
            token_gate=token_gate,
            used_token_ids=existing_used,
        )

    if gate_status != TOKEN_GATE_STATUS_VERIFIED:
        blocking_reasons = tuple(_list_of_str(token_gate.get("blocking_reasons")))
        required_actions = tuple(_list_of_str(token_gate.get("required_actions")))
        if not blocking_reasons:
            blocking_reasons = ("Approval token gate is not verified.",)
        if not required_actions:
            required_actions = ("verify_approval_token_gate",)
        return _blocked_report(
            token_gate=token_gate,
            used_token_ids=existing_used,
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
        )

    token_id = _str(token_gate.get("token_id"))
    operation_ids = tuple(_list_of_str(token_gate.get("verified_operation_ids")))
    if not token_id or not operation_ids:
        return _blocked_report(
            token_gate=token_gate,
            used_token_ids=existing_used,
            blocking_reasons=(
                "Verified approval token gate is missing token id or operations.",
            ),
            required_actions=("regenerate_approval_token_gate",),
        )
    if token_id in set(existing_used):
        return _blocked_report(
            token_gate=token_gate,
            used_token_ids=existing_used,
            blocking_reasons=("Approval token is already present in the used-token ledger.",),
            required_actions=("regenerate_human_approval_token",),
        )

    receipt = _receipt_from_gate(
        token_gate=token_gate,
        actor=actor,
        commit_reason=commit_reason,
    )
    return MemoryEvidenceRepairCommitReceiptReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECEIPT_STATUS_READY,
        receipts=(receipt,),
        used_token_ids=tuple(_dedupe_strings([*existing_used, receipt.used_token_id])),
        blocking_reasons=(),
        required_actions=(),
        source_gate=dict(token_gate),
    )


def empty_evidence_repair_commit_receipt() -> MemoryEvidenceRepairCommitReceiptReport:
    """Return an empty read-only manual commit receipt report."""

    return _empty_report(token_gate={}, used_token_ids=())


def _receipt_from_gate(
    *,
    token_gate: Mapping[str, Any],
    actor: str,
    commit_reason: str,
) -> MemoryEvidenceRepairCommitReceipt:
    source_token = _dict(token_gate.get("source_token"))
    source_dry_run = _dict(token_gate.get("source_dry_run"))
    operations = _operation_rows(source_dry_run)
    token_id = _str(token_gate.get("token_id"))
    operation_ids = tuple(_list_of_str(token_gate.get("verified_operation_ids")))
    candidate_ids = tuple(_list_of_str(token_gate.get("verified_candidate_ids")))
    patch_digests = tuple(_list_of_str(token_gate.get("verified_patch_digests")))
    ledger_entry_ids = tuple(_str(operation.get("ledger_entry_id")) for operation in operations)
    token_digest = _str(source_token.get("token_digest"))
    dry_run_digest = _str(source_token.get("dry_run_digest"))
    receipt_seed = {
        "receipt_type": RECEIPT_TYPE,
        "scope": RECEIPT_SCOPE,
        "actor": _str(actor) or "human",
        "commit_reason": _str(commit_reason),
        "token_id": token_id,
        "token_digest": token_digest,
        "dry_run_digest": dry_run_digest,
        "gate_status": _str(token_gate.get("status")),
        "operation_ids": operation_ids,
        "ledger_entry_ids": ledger_entry_ids,
        "candidate_ids": candidate_ids,
        "patch_digests": patch_digests,
    }
    receipt_digest = _digest(receipt_seed)
    receipt_id = f"commit-receipt-{receipt_digest[:16]}"
    return MemoryEvidenceRepairCommitReceipt(
        id=receipt_id,
        receipt_type=RECEIPT_TYPE,
        status=RECEIPT_STATUS_READY,
        scope=RECEIPT_SCOPE,
        actor=_str(actor) or "human",
        commit_reason=_str(commit_reason),
        token_id=token_id,
        token_digest=token_digest,
        dry_run_digest=dry_run_digest,
        gate_status=_str(token_gate.get("status")),
        receipt_digest=receipt_digest,
        receipt_preview=f"{receipt_id}:{receipt_digest[:12]}",
        operation_ids=operation_ids,
        ledger_entry_ids=ledger_entry_ids,
        candidate_ids=candidate_ids,
        patch_digests=patch_digests,
        used_token_id=token_id,
        would_record_receipt=True,
        would_mark_token_used=True,
        safety_note=(
            "Read-only receipt draft. It previews a future commit receipt and "
            "used-token ledger update, but does not persist either."
        ),
        source_gate=dict(token_gate),
    )


def _empty_report(
    *,
    token_gate: Mapping[str, Any],
    used_token_ids: Sequence[str],
) -> MemoryEvidenceRepairCommitReceiptReport:
    return MemoryEvidenceRepairCommitReceiptReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECEIPT_STATUS_NO_ACTION_NEEDED,
        receipts=(),
        used_token_ids=tuple(_dedupe_strings(used_token_ids)),
        blocking_reasons=(),
        required_actions=(),
        source_gate=dict(token_gate),
    )


def _blocked_report(
    *,
    token_gate: Mapping[str, Any],
    used_token_ids: Sequence[str],
    blocking_reasons: Sequence[str],
    required_actions: Sequence[str],
) -> MemoryEvidenceRepairCommitReceiptReport:
    return MemoryEvidenceRepairCommitReceiptReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECEIPT_STATUS_BLOCKED,
        receipts=(),
        used_token_ids=tuple(_dedupe_strings(used_token_ids)),
        blocking_reasons=tuple(_dedupe_strings(blocking_reasons)),
        required_actions=tuple(_dedupe_strings(required_actions)),
        source_gate=dict(token_gate),
    )


def _used_token_ids(
    existing_receipts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    explicit_used_token_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    candidates: list[str] = _list_of_str(explicit_used_token_ids)
    if isinstance(existing_receipts, Mapping):
        candidates.extend(_list_of_str(existing_receipts.get("used_token_ids")))
        receipt_rows = existing_receipts.get("receipts")
        if isinstance(receipt_rows, list):
            candidates.extend(_token_ids_from_receipts(receipt_rows))
        elif _str(existing_receipts.get("token_id")):
            candidates.append(_str(existing_receipts.get("token_id")))
        elif _str(existing_receipts.get("used_token_id")):
            candidates.append(_str(existing_receipts.get("used_token_id")))
    elif isinstance(existing_receipts, list):
        candidates.extend(_token_ids_from_receipts(existing_receipts))
    return tuple(_dedupe_strings(candidates))


def _token_ids_from_receipts(receipts: Sequence[Any]) -> list[str]:
    token_ids: list[str] = []
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            continue
        token_id = _str(receipt.get("used_token_id")) or _str(receipt.get("token_id"))
        if token_id:
            token_ids.append(token_id)
    return token_ids


def _operation_rows(dry_run: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        operation
        for operation in _list(dry_run.get("operations"))
        if isinstance(operation, Mapping)
    ]


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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
