"""Read-only recovery closure ledger audit preview for memory evidence repair.

The ledger audit preview verifies that a recovery closure ledger entry can be
trusted by later governance layers. It never writes durable memory, records a
real audit, or marks a recovery closed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from agent.memory_evidence_repair_manual_commit_dry_run import (
    CHECK_STATUS_FAIL,
    CHECK_STATUS_PASS,
)
from agent.memory_evidence_repair_recovery_closure_gate import (
    RECOVERY_CLOSURE_DRAFT_STATUS_READY,
)
from agent.memory_evidence_repair_recovery_closure_ledger_preview import (
    RECOVERY_CLOSURE_DECISION,
    RECOVERY_CLOSURE_LEDGER_ENTRY_STATUS_READY,
    RECOVERY_CLOSURE_LEDGER_ENTRY_TYPE,
    RECOVERY_CLOSURE_LEDGER_STATUS_NO_ACTION_NEEDED,
    RECOVERY_CLOSURE_LEDGER_STATUS_READY,
    RECOVERY_CLOSURE_TYPE,
)
from agent.memory_evidence_repair_recovery_completion_audit_preview import (
    RECOVERY_AUDIT_STEP_STATUS_PASS,
    RECOVERY_COMPLETION_AUDIT_STATUS_READY,
)


RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_READY = (
    "recovery_closure_ledger_audit_preview_ready"
)
RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_BLOCKED = "blocked"
RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_NO_ACTION_NEEDED = "no_action_needed"
RECOVERY_CLOSURE_LEDGER_AUDIT_PREVIEW_STATUS_READY = (
    "recovery_closure_ledger_audit_ready"
)

LEDGER_AUDIT_STEP_STATUS_PASS = "pass"
LEDGER_AUDIT_STEP_STATUS_FAIL = "fail"

CHECK_RECOVERY_CLOSURE_LEDGER_READY = "recovery_closure_ledger_ready"
CHECK_RECOVERY_CLOSURE_LEDGER_INTEGRITY = "recovery_closure_ledger_integrity"
CHECK_RECOVERY_CLOSURE_LEDGER_SOURCE_CHAIN = "recovery_closure_ledger_source_chain"
CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_REQUIREMENTS = (
    "recovery_closure_ledger_audit_requirements"
)


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureLedgerAuditCheck:
    """One read-only recovery closure ledger audit check."""

    id: str
    status: str
    reason: str
    required_actions: tuple[str, ...]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "reason": self.reason,
            "required_actions": list(self.required_actions),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureLedgerAuditStep:
    """One structural audit step for a future closure ledger entry."""

    id: str
    sequence: int
    category: str
    assertion: str
    ledger_entry_id: str
    reference_id: str
    expected_signal: dict[str, Any]
    observed_signal: dict[str, Any]
    status: str
    required_action_on_fail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sequence": self.sequence,
            "category": self.category,
            "assertion": self.assertion,
            "ledger_entry_id": self.ledger_entry_id,
            "reference_id": self.reference_id,
            "expected_signal": dict(self.expected_signal),
            "observed_signal": dict(self.observed_signal),
            "status": self.status,
            "required_action_on_fail": self.required_action_on_fail,
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureLedgerAuditPreview:
    """One read-only recovery closure ledger audit preview."""

    id: str
    status: str
    ledger_entry_id: str
    closure_id: str
    audit_preview_id: str
    receipt_id: str
    executor_preview_id: str
    decision_id: str
    execution_preview_id: str
    token_id: str
    lock_id: str
    route: str
    operation_ids: tuple[str, ...]
    recovery_step_ids: tuple[str, ...]
    future_mutation_step_ids: tuple[str, ...]
    completion_audit_step_ids: tuple[str, ...]
    milestone_ids: tuple[str, ...]
    ledger_audit_step_ids: tuple[str, ...]
    ledger_entry_digest: str
    closure_digest: str
    audit_digest: str
    audit_preview: str
    audit_steps: tuple[MemoryEvidenceRepairRecoveryClosureLedgerAuditStep, ...]
    would_verify_ledger_entry_integrity: bool
    would_verify_closure_source_integrity: bool
    would_verify_lifecycle_milestones: bool
    would_verify_completion_audit_evidence: bool
    safety_note: str
    source_recovery_closure_ledger_entry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "ledger_entry_id": self.ledger_entry_id,
            "closure_id": self.closure_id,
            "audit_preview_id": self.audit_preview_id,
            "receipt_id": self.receipt_id,
            "executor_preview_id": self.executor_preview_id,
            "decision_id": self.decision_id,
            "execution_preview_id": self.execution_preview_id,
            "token_id": self.token_id,
            "lock_id": self.lock_id,
            "route": self.route,
            "operation_ids": list(self.operation_ids),
            "recovery_step_ids": list(self.recovery_step_ids),
            "future_mutation_step_ids": list(self.future_mutation_step_ids),
            "completion_audit_step_ids": list(self.completion_audit_step_ids),
            "milestone_ids": list(self.milestone_ids),
            "ledger_audit_step_ids": list(self.ledger_audit_step_ids),
            "ledger_entry_digest": self.ledger_entry_digest,
            "closure_digest": self.closure_digest,
            "audit_digest": self.audit_digest,
            "audit_preview": self.audit_preview,
            "audit_steps": [step.to_dict() for step in self.audit_steps],
            "would_verify_ledger_entry_integrity": (
                self.would_verify_ledger_entry_integrity
            ),
            "would_verify_closure_source_integrity": (
                self.would_verify_closure_source_integrity
            ),
            "would_verify_lifecycle_milestones": self.would_verify_lifecycle_milestones,
            "would_verify_completion_audit_evidence": (
                self.would_verify_completion_audit_evidence
            ),
            "safety_note": self.safety_note,
            "source_recovery_closure_ledger_entry": dict(
                self.source_recovery_closure_ledger_entry
            ),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureLedgerAuditPreviewReport:
    """Read-only recovery closure ledger audit preview report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairRecoveryClosureLedgerAuditCheck, ...]
    previews: tuple[MemoryEvidenceRepairRecoveryClosureLedgerAuditPreview, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_recovery_closure_ledger_report: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        by_check_status: dict[str, int] = {}
        by_audit_step_status: dict[str, int] = {}
        for check in self.checks:
            by_check_status[check.status] = by_check_status.get(check.status, 0) + 1
        for preview in self.previews:
            for step in preview.audit_steps:
                by_audit_step_status[step.status] = (
                    by_audit_step_status.get(step.status, 0) + 1
                )
        return {
            "status": self.status,
            "check_count": len(self.checks),
            "pass_count": by_check_status.get(CHECK_STATUS_PASS, 0),
            "fail_count": by_check_status.get(CHECK_STATUS_FAIL, 0),
            "preview_count": len(self.previews),
            "audit_step_count": sum(len(preview.audit_steps) for preview in self.previews),
            "observed_pass_step_count": by_audit_step_status.get(
                LEDGER_AUDIT_STEP_STATUS_PASS,
                0,
            ),
            "observed_fail_step_count": by_audit_step_status.get(
                LEDGER_AUDIT_STEP_STATUS_FAIL,
                0,
            ),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "recovery_closure_ledger_audit_preview_available": bool(self.previews),
            "recovery_closure_ledger_audit_ready": (
                self.status == RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_READY
            ),
            "has_blocks": self.status == RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_BLOCKED,
            "requires_followup": bool(self.required_actions),
            "by_check_status": by_check_status,
            "by_audit_step_status": by_audit_step_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
            "previews": [preview.to_dict() for preview in self.previews],
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "source_recovery_closure_ledger_report": dict(
                self.source_recovery_closure_ledger_report
            ),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_closure_ledger_audit_preview(
    *,
    recovery_closure_ledger: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairRecoveryClosureLedgerAuditPreviewReport:
    """Build a read-only audit preview from a recovery closure ledger report."""

    recovery_closure_ledger = (
        recovery_closure_ledger if isinstance(recovery_closure_ledger, Mapping) else {}
    )
    report_status = _str(recovery_closure_ledger.get("status"))
    entry = _extract_entry(recovery_closure_ledger)

    if not entry:
        if (
            not recovery_closure_ledger
            or report_status == RECOVERY_CLOSURE_LEDGER_STATUS_NO_ACTION_NEEDED
        ):
            return _empty_report(recovery_closure_ledger=recovery_closure_ledger)
        source_blocking = tuple(
            _list_of_str(recovery_closure_ledger.get("blocking_reasons"))
        )
        source_actions = tuple(_list_of_str(recovery_closure_ledger.get("required_actions")))
        return _blocked_report(
            recovery_closure_ledger=recovery_closure_ledger,
            checks=(
                _fail(
                    CHECK_RECOVERY_CLOSURE_LEDGER_READY,
                    "No recovery closure ledger entry was supplied."
                    if not source_blocking
                    else "; ".join(source_blocking),
                    source_actions
                    or ("produce_recovery_closure_ledger_preview",),
                    {"recovery_closure_ledger_status": report_status},
                ),
            ),
        )

    checks = (
        _ledger_ready_check(recovery_closure_ledger, entry),
        _ledger_integrity_check(entry),
        _source_chain_check(entry),
        _audit_requirements_check(entry),
    )
    blocking_reasons = tuple(
        _dedupe_strings(check.reason for check in checks if check.status == CHECK_STATUS_FAIL)
    )
    required_actions = tuple(
        _dedupe_strings(
            action
            for check in checks
            if check.status == CHECK_STATUS_FAIL
            for action in check.required_actions
        )
    )
    if blocking_reasons:
        return MemoryEvidenceRepairRecoveryClosureLedgerAuditPreviewReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_BLOCKED,
            checks=checks,
            previews=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            source_recovery_closure_ledger_report=dict(recovery_closure_ledger),
        )

    preview = _preview_from_entry(entry)
    return MemoryEvidenceRepairRecoveryClosureLedgerAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_READY,
        checks=checks,
        previews=(preview,),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_closure_ledger_report=dict(recovery_closure_ledger),
    )


def empty_evidence_repair_recovery_closure_ledger_audit_preview() -> MemoryEvidenceRepairRecoveryClosureLedgerAuditPreviewReport:
    """Return an empty read-only recovery closure ledger audit preview."""

    return _empty_report(recovery_closure_ledger={})


def _extract_entry(recovery_closure_ledger: Mapping[str, Any]) -> dict[str, Any]:
    entries = recovery_closure_ledger.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, Mapping):
                return dict(entry)
    if _str(recovery_closure_ledger.get("id")).startswith("recovery-closure-ledger-"):
        return dict(recovery_closure_ledger)
    return {}


def _ledger_ready_check(
    recovery_closure_ledger: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerAuditCheck:
    report_status = _str(recovery_closure_ledger.get("status"))
    entry_status = _str(entry.get("status"))
    if report_status and report_status != RECOVERY_CLOSURE_LEDGER_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_CLOSURE_LEDGER_READY,
            "Recovery closure ledger report is not ready for audit preview.",
            tuple(_list_of_str(recovery_closure_ledger.get("required_actions")))
            or ("produce_ready_recovery_closure_ledger_preview",),
            {"report_status": report_status, "entry_status": entry_status},
        )
    if entry_status != RECOVERY_CLOSURE_LEDGER_ENTRY_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_CLOSURE_LEDGER_READY,
            "Recovery closure ledger entry is not ready for audit preview.",
            ("regenerate_recovery_closure_ledger_preview",),
            {"report_status": report_status, "entry_status": entry_status},
        )
    return _pass(
        CHECK_RECOVERY_CLOSURE_LEDGER_READY,
        "Recovery closure ledger entry is ready for audit preview.",
        {"ledger_entry_id": _str(entry.get("id"))},
    )


def _ledger_integrity_check(
    entry: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerAuditCheck:
    expected_digest = _expected_entry_digest(entry)
    actual_digest = _str(entry.get("entry_digest"))
    expected_id = (
        f"recovery-closure-ledger-{expected_digest[:16]}" if expected_digest else ""
    )
    missing_fields: list[str] = []
    for field_name in (
        "id",
        "entry_type",
        "status",
        "closure_id",
        "closure_decision",
        "audit_preview_id",
        "receipt_id",
        "executor_preview_id",
        "decision_id",
        "execution_preview_id",
        "token_id",
        "lock_id",
        "route",
        "operation_ids",
        "recovery_step_ids",
        "future_mutation_step_ids",
        "audit_step_ids",
        "closure_digest",
        "entry_digest",
        "milestones",
        "source_recovery_closure",
    ):
        value = entry.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif isinstance(value, Mapping) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, (list, Mapping)) and not _str(value):
            missing_fields.append(field_name)
    if (
        missing_fields
        or actual_digest != expected_digest
        or _str(entry.get("id")) != expected_id
        or _str(entry.get("entry_type")) != RECOVERY_CLOSURE_LEDGER_ENTRY_TYPE
        or _str(entry.get("status")) != RECOVERY_CLOSURE_LEDGER_ENTRY_STATUS_READY
    ):
        return _fail(
            CHECK_RECOVERY_CLOSURE_LEDGER_INTEGRITY,
            "Recovery closure ledger entry digest, id, type, status, or required fields are invalid.",
            ("regenerate_recovery_closure_ledger_preview",),
            {
                "expected_entry_digest": expected_digest,
                "actual_entry_digest": actual_digest,
                "expected_entry_id": expected_id,
                "actual_entry_id": _str(entry.get("id")),
                "missing_fields": missing_fields,
            },
        )
    return _pass(
        CHECK_RECOVERY_CLOSURE_LEDGER_INTEGRITY,
        "Recovery closure ledger entry integrity checks pass.",
        {"ledger_entry_id": expected_id, "entry_digest": actual_digest},
    )


def _source_chain_check(
    entry: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerAuditCheck:
    closure = _dict(entry.get("source_recovery_closure"))
    audit = _dict(closure.get("source_recovery_completion_audit_preview"))
    receipt = _dict(audit.get("source_recovery_completion_receipt"))
    executor = _dict(receipt.get("source_recovery_executor_preview"))
    expected_closure_digest = _expected_closure_digest(closure)
    audit_steps = _audit_steps(audit)
    non_pass_audit_steps = [
        _str(step.get("id")) or f"audit-step-{index}"
        for index, step in enumerate(audit_steps, start=1)
        if _str(step.get("status")) != RECOVERY_AUDIT_STEP_STATUS_PASS
    ]
    mismatches: list[dict[str, str]] = []
    for field_name in (
        "closure_id",
        "closure_decision",
        "audit_preview_id",
        "receipt_id",
        "executor_preview_id",
        "decision_id",
        "execution_preview_id",
        "token_id",
        "lock_id",
        "route",
        "closure_digest",
    ):
        closure_key = "id" if field_name == "closure_id" else field_name
        _compare(mismatches, field_name, entry, field_name, closure, closure_key)
    if _str(audit.get("id")) != _str(entry.get("audit_preview_id")):
        mismatches.append(
            {
                "field": "audit_preview_id",
                "left": _str(entry.get("audit_preview_id")),
                "right": _str(audit.get("id")),
                "right_key": "source_recovery_completion_audit_preview.id",
            }
        )
    if _str(receipt.get("id")) != _str(entry.get("receipt_id")):
        mismatches.append(
            {
                "field": "receipt_id",
                "left": _str(entry.get("receipt_id")),
                "right": _str(receipt.get("id")),
                "right_key": "source_recovery_completion_receipt.id",
            }
        )
    if _str(executor.get("id")) != _str(entry.get("executor_preview_id")):
        mismatches.append(
            {
                "field": "executor_preview_id",
                "left": _str(entry.get("executor_preview_id")),
                "right": _str(executor.get("id")),
                "right_key": "source_recovery_executor_preview.id",
            }
        )
    missing_sources = [
        name
        for name, source in (
            ("source_recovery_closure", closure),
            ("source_recovery_completion_audit_preview", audit),
            ("source_recovery_completion_receipt", receipt),
            ("source_recovery_executor_preview", executor),
        )
        if not source
    ]
    if (
        missing_sources
        or mismatches
        or _str(closure.get("status")) != RECOVERY_CLOSURE_DRAFT_STATUS_READY
        or _str(entry.get("closure_digest")) != expected_closure_digest
        or _str(audit.get("status")) != RECOVERY_COMPLETION_AUDIT_STATUS_READY
        or audit.get("observed_state_supplied") is not True
        or non_pass_audit_steps
    ):
        return _fail(
            CHECK_RECOVERY_CLOSURE_LEDGER_SOURCE_CHAIN,
            "Recovery closure ledger source chain is incomplete or inconsistent.",
            ("regenerate_recovery_closure_ledger_preview",),
            {
                "missing_sources": missing_sources,
                "mismatches": mismatches,
                "closure_status": _str(closure.get("status")),
                "expected_closure_digest": expected_closure_digest,
                "actual_closure_digest": _str(entry.get("closure_digest")),
                "audit_status": _str(audit.get("status")),
                "observed_state_supplied": audit.get("observed_state_supplied") is True,
                "non_pass_audit_step_ids": non_pass_audit_steps,
            },
        )
    return _pass(
        CHECK_RECOVERY_CLOSURE_LEDGER_SOURCE_CHAIN,
        "Recovery closure ledger source chain is internally consistent.",
        {
            "closure_id": _str(entry.get("closure_id")),
            "audit_preview_id": _str(entry.get("audit_preview_id")),
            "receipt_id": _str(entry.get("receipt_id")),
            "executor_preview_id": _str(entry.get("executor_preview_id")),
            "audit_step_count": len(audit_steps),
        },
    )


def _audit_requirements_check(
    entry: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerAuditCheck:
    milestones = _milestones(entry)
    missing_lists = [
        field_name
        for field_name in (
            "operation_ids",
            "recovery_step_ids",
            "future_mutation_step_ids",
            "audit_step_ids",
        )
        if not _list_of_str(entry.get(field_name))
    ]
    missing_flags = [
        field_name
        for field_name in (
            "would_record_ledger_entry",
            "would_preserve_recovery_evidence",
        )
        if entry.get(field_name) is not True
    ]
    milestone_errors = _milestone_errors(entry)
    if missing_lists or missing_flags or milestone_errors:
        return _fail(
            CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_REQUIREMENTS,
            "Recovery closure ledger audit requirements are incomplete.",
            ("regenerate_recovery_closure_ledger_preview",),
            {
                "missing_lists": missing_lists,
                "missing_flags": missing_flags,
                "milestone_errors": milestone_errors,
                "milestone_count": len(milestones),
            },
        )
    return _pass(
        CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_REQUIREMENTS,
        "Recovery closure ledger audit has all structural evidence needed.",
        {
            "milestone_count": len(milestones),
            "operation_count": len(_list_of_str(entry.get("operation_ids"))),
            "recovery_step_count": len(_list_of_str(entry.get("recovery_step_ids"))),
            "completion_audit_step_count": len(_list_of_str(entry.get("audit_step_ids"))),
        },
    )


def _preview_from_entry(
    entry: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerAuditPreview:
    audit_steps = tuple(_audit_steps_from_entry(entry))
    ledger_audit_step_ids = tuple(step.id for step in audit_steps)
    operation_ids = tuple(_list_of_str(entry.get("operation_ids")))
    recovery_step_ids = tuple(_list_of_str(entry.get("recovery_step_ids")))
    future_mutation_step_ids = tuple(_list_of_str(entry.get("future_mutation_step_ids")))
    completion_audit_step_ids = tuple(_list_of_str(entry.get("audit_step_ids")))
    milestone_ids = tuple(_str(milestone.get("id")) for milestone in _milestones(entry))
    audit_seed = {
        "ledger_entry_id": _str(entry.get("id")),
        "closure_id": _str(entry.get("closure_id")),
        "audit_preview_id": _str(entry.get("audit_preview_id")),
        "receipt_id": _str(entry.get("receipt_id")),
        "executor_preview_id": _str(entry.get("executor_preview_id")),
        "decision_id": _str(entry.get("decision_id")),
        "execution_preview_id": _str(entry.get("execution_preview_id")),
        "token_id": _str(entry.get("token_id")),
        "lock_id": _str(entry.get("lock_id")),
        "route": _str(entry.get("route")),
        "operation_ids": operation_ids,
        "recovery_step_ids": recovery_step_ids,
        "future_mutation_step_ids": future_mutation_step_ids,
        "completion_audit_step_ids": completion_audit_step_ids,
        "milestone_ids": milestone_ids,
        "ledger_audit_step_ids": ledger_audit_step_ids,
        "entry_digest": _str(entry.get("entry_digest")),
        "closure_digest": _str(entry.get("closure_digest")),
    }
    audit_digest = _digest(audit_seed)
    audit_id = f"recovery-closure-ledger-audit-{audit_digest[:16]}"
    return MemoryEvidenceRepairRecoveryClosureLedgerAuditPreview(
        id=audit_id,
        status=RECOVERY_CLOSURE_LEDGER_AUDIT_PREVIEW_STATUS_READY,
        ledger_entry_id=_str(entry.get("id")),
        closure_id=_str(entry.get("closure_id")),
        audit_preview_id=_str(entry.get("audit_preview_id")),
        receipt_id=_str(entry.get("receipt_id")),
        executor_preview_id=_str(entry.get("executor_preview_id")),
        decision_id=_str(entry.get("decision_id")),
        execution_preview_id=_str(entry.get("execution_preview_id")),
        token_id=_str(entry.get("token_id")),
        lock_id=_str(entry.get("lock_id")),
        route=_str(entry.get("route")),
        operation_ids=operation_ids,
        recovery_step_ids=recovery_step_ids,
        future_mutation_step_ids=future_mutation_step_ids,
        completion_audit_step_ids=completion_audit_step_ids,
        milestone_ids=milestone_ids,
        ledger_audit_step_ids=ledger_audit_step_ids,
        ledger_entry_digest=_str(entry.get("entry_digest")),
        closure_digest=_str(entry.get("closure_digest")),
        audit_digest=audit_digest,
        audit_preview=f"{audit_id}:{audit_digest[:12]}",
        audit_steps=audit_steps,
        would_verify_ledger_entry_integrity=True,
        would_verify_closure_source_integrity=True,
        would_verify_lifecycle_milestones=True,
        would_verify_completion_audit_evidence=True,
        safety_note=(
            "Read-only recovery closure ledger audit preview. It checks whether "
            "a future closure ledger entry is structurally trustworthy, but does "
            "not record the audit or mutate durable memory."
        ),
        source_recovery_closure_ledger_entry=dict(entry),
    )


def _audit_steps_from_entry(
    entry: Mapping[str, Any],
) -> list[MemoryEvidenceRepairRecoveryClosureLedgerAuditStep]:
    rows = (
        (
            "ledger_entry",
            "ledger_entry_digest_valid",
            _str(entry.get("id")),
            {"expected_entry_digest": _expected_entry_digest(entry)},
            {"actual_entry_digest": _str(entry.get("entry_digest"))},
        ),
        (
            "closure_source",
            "closure_source_digest_valid",
            _str(entry.get("closure_id")),
            {
                "expected_closure_digest": _expected_closure_digest(
                    _dict(entry.get("source_recovery_closure"))
                )
            },
            {"actual_closure_digest": _str(entry.get("closure_digest"))},
        ),
        (
            "milestones",
            "lifecycle_milestones_ordered",
            _str(entry.get("id")),
            {"required_stages": list(_required_milestone_stages())},
            {"stages": [_str(milestone.get("stage")) for milestone in _milestones(entry)]},
        ),
        (
            "source_chain",
            "source_chain_consistent",
            _str(entry.get("closure_id")),
            {"closure_id": _str(entry.get("closure_id"))},
            {"source_closure_id": _str(_dict(entry.get("source_recovery_closure")).get("id"))},
        ),
        (
            "completion_audit",
            "completion_audit_observed_pass",
            _str(entry.get("audit_preview_id")),
            {"observed_state_supplied": True},
            {
                "observed_state_supplied": _dict(
                    _dict(entry.get("source_recovery_closure")).get(
                        "source_recovery_completion_audit_preview"
                    )
                ).get("observed_state_supplied")
                is True
            },
        ),
        (
            "read_only",
            "ledger_entry_is_preview_only",
            _str(entry.get("id")),
            {"would_record_ledger_entry": True},
            {"would_record_ledger_entry": entry.get("would_record_ledger_entry") is True},
        ),
    )
    return [
        _audit_step(
            sequence=index,
            category=category,
            assertion=assertion,
            entry=entry,
            reference_id=reference_id,
            expected_signal=expected_signal,
            observed_signal=observed_signal,
        )
        for index, (
            category,
            assertion,
            reference_id,
            expected_signal,
            observed_signal,
        ) in enumerate(rows, start=1)
    ]


def _audit_step(
    *,
    sequence: int,
    category: str,
    assertion: str,
    entry: Mapping[str, Any],
    reference_id: str,
    expected_signal: Mapping[str, Any],
    observed_signal: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerAuditStep:
    step_seed = {
        "sequence": sequence,
        "category": category,
        "assertion": assertion,
        "ledger_entry_id": _str(entry.get("id")),
        "reference_id": _str(reference_id),
    }
    step_id = f"recovery-closure-ledger-audit-step-{_digest(step_seed)[:16]}"
    return MemoryEvidenceRepairRecoveryClosureLedgerAuditStep(
        id=step_id,
        sequence=sequence,
        category=category,
        assertion=assertion,
        ledger_entry_id=_str(entry.get("id")),
        reference_id=_str(reference_id),
        expected_signal=dict(expected_signal),
        observed_signal=dict(observed_signal),
        status=LEDGER_AUDIT_STEP_STATUS_PASS,
        required_action_on_fail="regenerate_recovery_closure_ledger_preview",
    )


def _expected_entry_digest(entry: Mapping[str, Any]) -> str:
    seed = {
        "entry_type": RECOVERY_CLOSURE_LEDGER_ENTRY_TYPE,
        "closure_id": _str(entry.get("closure_id")),
        "closure_digest": _str(entry.get("closure_digest")),
        "audit_preview_id": _str(entry.get("audit_preview_id")),
        "receipt_id": _str(entry.get("receipt_id")),
        "executor_preview_id": _str(entry.get("executor_preview_id")),
        "decision_id": _str(entry.get("decision_id")),
        "execution_preview_id": _str(entry.get("execution_preview_id")),
        "token_id": _str(entry.get("token_id")),
        "lock_id": _str(entry.get("lock_id")),
        "route": _str(entry.get("route")),
        "operation_ids": tuple(_list_of_str(entry.get("operation_ids"))),
        "recovery_step_ids": tuple(_list_of_str(entry.get("recovery_step_ids"))),
        "future_mutation_step_ids": tuple(
            _list_of_str(entry.get("future_mutation_step_ids"))
        ),
        "audit_step_ids": tuple(_list_of_str(entry.get("audit_step_ids"))),
        "milestone_ids": tuple(_str(milestone.get("id")) for milestone in _milestones(entry)),
    }
    return _digest(seed)


def _expected_closure_digest(closure: Mapping[str, Any]) -> str:
    seed = {
        "closure_type": RECOVERY_CLOSURE_TYPE,
        "closure_decision": RECOVERY_CLOSURE_DECISION,
        "audit_preview_id": _str(closure.get("audit_preview_id")),
        "receipt_id": _str(closure.get("receipt_id")),
        "executor_preview_id": _str(closure.get("executor_preview_id")),
        "decision_id": _str(closure.get("decision_id")),
        "execution_preview_id": _str(closure.get("execution_preview_id")),
        "token_id": _str(closure.get("token_id")),
        "lock_id": _str(closure.get("lock_id")),
        "route": _str(closure.get("route")),
        "operation_ids": tuple(_list_of_str(closure.get("operation_ids"))),
        "recovery_step_ids": tuple(_list_of_str(closure.get("recovery_step_ids"))),
        "future_mutation_step_ids": tuple(
            _list_of_str(closure.get("future_mutation_step_ids"))
        ),
        "audit_step_ids": tuple(_list_of_str(closure.get("audit_step_ids"))),
    }
    return _digest(seed)


def _milestone_errors(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    milestones = _milestones(entry)
    errors: list[dict[str, Any]] = []
    expected_reference_ids = {
        "recovery_decision": _str(entry.get("decision_id")),
        "recovery_execution_preview": _str(entry.get("execution_preview_id")),
        "recovery_approval_token": _str(entry.get("token_id")),
        "recovery_write_lock": _str(entry.get("lock_id")),
        "recovery_executor_preview": _str(entry.get("executor_preview_id")),
        "recovery_completion_receipt": _str(entry.get("receipt_id")),
        "recovery_completion_audit": _str(entry.get("audit_preview_id")),
        "recovery_closure": _str(entry.get("closure_id")),
    }
    stages = [_str(milestone.get("stage")) for milestone in milestones]
    if tuple(stages) != _required_milestone_stages():
        errors.append(
            {
                "type": "stage_order",
                "expected": list(_required_milestone_stages()),
                "actual": stages,
            }
        )
    for expected_sequence, milestone in enumerate(milestones, start=1):
        stage = _str(milestone.get("stage"))
        reference_id = _str(milestone.get("reference_id"))
        expected_reference_id = expected_reference_ids.get(stage, "")
        expected_id = _milestone_id(stage, expected_reference_id, expected_sequence)
        if int(milestone.get("sequence") or 0) != expected_sequence:
            errors.append(
                {
                    "type": "sequence",
                    "stage": stage,
                    "expected": expected_sequence,
                    "actual": milestone.get("sequence"),
                }
            )
        if reference_id != expected_reference_id:
            errors.append(
                {
                    "type": "reference_id",
                    "stage": stage,
                    "expected": expected_reference_id,
                    "actual": reference_id,
                }
            )
        if _str(milestone.get("id")) != expected_id:
            errors.append(
                {
                    "type": "milestone_id",
                    "stage": stage,
                    "expected": expected_id,
                    "actual": _str(milestone.get("id")),
                }
            )
    return errors


def _milestones(entry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        milestone
        for milestone in _list(entry.get("milestones"))
        if isinstance(milestone, Mapping)
    ]


def _required_milestone_stages() -> tuple[str, ...]:
    return (
        "recovery_decision",
        "recovery_execution_preview",
        "recovery_approval_token",
        "recovery_write_lock",
        "recovery_executor_preview",
        "recovery_completion_receipt",
        "recovery_completion_audit",
        "recovery_closure",
    )


def _milestone_id(stage: str, reference_id: str, sequence: int) -> str:
    digest = _digest({"stage": stage, "reference_id": reference_id, "sequence": sequence})
    return f"recovery-closure-milestone-{digest[:16]}"


def _audit_steps(audit: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        step
        for step in _list(audit.get("audit_steps"))
        if isinstance(step, Mapping)
    ]


def _compare(
    mismatches: list[dict[str, str]],
    label: str,
    left: Mapping[str, Any],
    left_key: str,
    right: Mapping[str, Any],
    right_key: str,
) -> None:
    left_value = _str(left.get(left_key))
    right_value = _str(right.get(right_key))
    if left_value != right_value:
        mismatches.append(
            {
                "field": label,
                "left": left_value,
                "right": right_value,
                "right_key": right_key,
            }
        )


def _empty_report(
    *,
    recovery_closure_ledger: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerAuditPreviewReport:
    return MemoryEvidenceRepairRecoveryClosureLedgerAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_NO_ACTION_NEEDED,
        checks=(),
        previews=(),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_closure_ledger_report=dict(recovery_closure_ledger),
    )


def _blocked_report(
    *,
    recovery_closure_ledger: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairRecoveryClosureLedgerAuditCheck, ...],
) -> MemoryEvidenceRepairRecoveryClosureLedgerAuditPreviewReport:
    return MemoryEvidenceRepairRecoveryClosureLedgerAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_BLOCKED,
        checks=checks,
        previews=(),
        blocking_reasons=tuple(
            _dedupe_strings(check.reason for check in checks if check.status == CHECK_STATUS_FAIL)
        ),
        required_actions=tuple(
            _dedupe_strings(
                action
                for check in checks
                if check.status == CHECK_STATUS_FAIL
                for action in check.required_actions
            )
        ),
        source_recovery_closure_ledger_report=dict(recovery_closure_ledger),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerAuditCheck:
    return MemoryEvidenceRepairRecoveryClosureLedgerAuditCheck(
        id=check_id,
        status=CHECK_STATUS_PASS,
        reason=reason,
        required_actions=(),
        details=dict(details),
    )


def _fail(
    check_id: str,
    reason: str,
    required_actions: Sequence[str],
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerAuditCheck:
    return MemoryEvidenceRepairRecoveryClosureLedgerAuditCheck(
        id=check_id,
        status=CHECK_STATUS_FAIL,
        reason=reason,
        required_actions=tuple(required_actions),
        details=dict(details),
    )


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
