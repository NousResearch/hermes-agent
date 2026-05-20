"""Read-only recovery closure ledger preview for memory evidence repair.

This layer turns a ready recovery closure draft into an audit-style lifecycle
ledger entry. It never writes durable memory and never marks a recovery closed.
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
    RECOVERY_CLOSURE_STATUS_NO_ACTION_NEEDED,
    RECOVERY_CLOSURE_STATUS_READY,
)
from agent.memory_evidence_repair_recovery_completion_audit_preview import (
    RECOVERY_AUDIT_STEP_STATUS_PASS,
    RECOVERY_COMPLETION_AUDIT_STATUS_READY,
)


RECOVERY_CLOSURE_LEDGER_STATUS_READY = "recovery_closure_ledger_preview_ready"
RECOVERY_CLOSURE_LEDGER_STATUS_BLOCKED = "blocked"
RECOVERY_CLOSURE_LEDGER_STATUS_NO_ACTION_NEEDED = "no_action_needed"
RECOVERY_CLOSURE_LEDGER_ENTRY_STATUS_READY = "recovery_closure_ledger_entry_ready"
RECOVERY_CLOSURE_LEDGER_ENTRY_TYPE = (
    "memory_evidence_repair_recovery_closure_ledger_entry"
)

CHECK_RECOVERY_CLOSURE_READY = "recovery_closure_ready"
CHECK_RECOVERY_CLOSURE_INTEGRITY = "recovery_closure_integrity"
CHECK_RECOVERY_LIFECYCLE_CHAIN = "recovery_lifecycle_chain"
CHECK_RECOVERY_LEDGER_REQUIREMENTS = "recovery_ledger_requirements"

RECOVERY_CLOSURE_TYPE = "memory_evidence_repair_recovery_closure"
RECOVERY_CLOSURE_DECISION = "close_recovery_loop"


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureLedgerCheck:
    """One read-only recovery closure ledger check."""

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
class MemoryEvidenceRepairRecoveryClosureLedgerMilestone:
    """One lifecycle milestone inside a future recovery closure ledger entry."""

    id: str
    sequence: int
    stage: str
    reference_id: str
    status: str
    summary: str
    source: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sequence": self.sequence,
            "stage": self.stage,
            "reference_id": self.reference_id,
            "status": self.status,
            "summary": self.summary,
            "source": dict(self.source),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureLedgerEntry:
    """One read-only future recovery closure ledger entry preview."""

    id: str
    entry_type: str
    status: str
    closure_id: str
    closure_decision: str
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
    audit_step_ids: tuple[str, ...]
    closure_digest: str
    entry_digest: str
    entry_preview: str
    milestones: tuple[MemoryEvidenceRepairRecoveryClosureLedgerMilestone, ...]
    would_record_ledger_entry: bool
    would_preserve_recovery_evidence: bool
    safety_note: str
    source_recovery_closure: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entry_type": self.entry_type,
            "status": self.status,
            "closure_id": self.closure_id,
            "closure_decision": self.closure_decision,
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
            "audit_step_ids": list(self.audit_step_ids),
            "closure_digest": self.closure_digest,
            "entry_digest": self.entry_digest,
            "entry_preview": self.entry_preview,
            "milestones": [milestone.to_dict() for milestone in self.milestones],
            "would_record_ledger_entry": self.would_record_ledger_entry,
            "would_preserve_recovery_evidence": self.would_preserve_recovery_evidence,
            "safety_note": self.safety_note,
            "source_recovery_closure": dict(self.source_recovery_closure),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureLedgerPreviewReport:
    """Read-only recovery closure ledger preview report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairRecoveryClosureLedgerCheck, ...]
    entries: tuple[MemoryEvidenceRepairRecoveryClosureLedgerEntry, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_recovery_closure_gate: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        by_check_status: dict[str, int] = {}
        milestone_count = 0
        for check in self.checks:
            by_check_status[check.status] = by_check_status.get(check.status, 0) + 1
        for entry in self.entries:
            milestone_count += len(entry.milestones)
        return {
            "status": self.status,
            "check_count": len(self.checks),
            "pass_count": by_check_status.get(CHECK_STATUS_PASS, 0),
            "fail_count": by_check_status.get(CHECK_STATUS_FAIL, 0),
            "entry_count": len(self.entries),
            "milestone_count": milestone_count,
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "recovery_closure_ledger_preview_available": bool(self.entries),
            "recovery_closure_ledger_ready": (
                self.status == RECOVERY_CLOSURE_LEDGER_STATUS_READY
            ),
            "would_record_ledger_entry_count": sum(
                1 for entry in self.entries if entry.would_record_ledger_entry
            ),
            "would_preserve_recovery_evidence_count": sum(
                1 for entry in self.entries if entry.would_preserve_recovery_evidence
            ),
            "has_blocks": self.status == RECOVERY_CLOSURE_LEDGER_STATUS_BLOCKED,
            "requires_followup": bool(self.required_actions),
            "by_check_status": by_check_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
            "entries": [entry.to_dict() for entry in self.entries],
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "source_recovery_closure_gate": dict(self.source_recovery_closure_gate),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_closure_ledger_preview(
    *,
    recovery_closure_gate: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairRecoveryClosureLedgerPreviewReport:
    """Build a read-only lifecycle ledger preview from a recovery closure gate."""

    recovery_closure_gate = (
        recovery_closure_gate if isinstance(recovery_closure_gate, Mapping) else {}
    )
    report_status = _str(recovery_closure_gate.get("status"))
    closure = _extract_closure(recovery_closure_gate)

    if not closure:
        if (
            not recovery_closure_gate
            or report_status == RECOVERY_CLOSURE_STATUS_NO_ACTION_NEEDED
        ):
            return _empty_report(recovery_closure_gate=recovery_closure_gate)
        source_blocking = tuple(_list_of_str(recovery_closure_gate.get("blocking_reasons")))
        source_actions = tuple(_list_of_str(recovery_closure_gate.get("required_actions")))
        return _blocked_report(
            recovery_closure_gate=recovery_closure_gate,
            checks=(
                _fail(
                    CHECK_RECOVERY_CLOSURE_READY,
                    "No recovery closure draft was supplied."
                    if not source_blocking
                    else "; ".join(source_blocking),
                    source_actions or ("produce_ready_recovery_closure_gate",),
                    {"recovery_closure_gate_status": report_status},
                ),
            ),
        )

    checks = (
        _closure_ready_check(recovery_closure_gate, closure),
        _closure_integrity_check(closure),
        _lifecycle_chain_check(closure),
        _ledger_requirements_check(closure),
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
        return MemoryEvidenceRepairRecoveryClosureLedgerPreviewReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_CLOSURE_LEDGER_STATUS_BLOCKED,
            checks=checks,
            entries=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            source_recovery_closure_gate=dict(recovery_closure_gate),
        )

    entry = _entry_from_closure(closure)
    return MemoryEvidenceRepairRecoveryClosureLedgerPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_LEDGER_STATUS_READY,
        checks=checks,
        entries=(entry,),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_closure_gate=dict(recovery_closure_gate),
    )


def empty_evidence_repair_recovery_closure_ledger_preview() -> MemoryEvidenceRepairRecoveryClosureLedgerPreviewReport:
    """Return an empty read-only recovery closure ledger preview."""

    return _empty_report(recovery_closure_gate={})


def _extract_closure(recovery_closure_gate: Mapping[str, Any]) -> dict[str, Any]:
    closures = recovery_closure_gate.get("closures")
    if isinstance(closures, list):
        for closure in closures:
            if isinstance(closure, Mapping):
                return dict(closure)
    if _str(recovery_closure_gate.get("id")).startswith("recovery-closure-"):
        return dict(recovery_closure_gate)
    return {}


def _closure_ready_check(
    recovery_closure_gate: Mapping[str, Any],
    closure: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerCheck:
    report_status = _str(recovery_closure_gate.get("status"))
    closure_status = _str(closure.get("status"))
    if report_status and report_status != RECOVERY_CLOSURE_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_CLOSURE_READY,
            "Recovery closure gate report is not ready for ledger preview.",
            tuple(_list_of_str(recovery_closure_gate.get("required_actions")))
            or ("produce_ready_recovery_closure_gate",),
            {"report_status": report_status, "closure_status": closure_status},
        )
    if closure_status != RECOVERY_CLOSURE_DRAFT_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_CLOSURE_READY,
            "Recovery closure draft is not ready for ledger preview.",
            ("regenerate_recovery_closure_gate",),
            {"report_status": report_status, "closure_status": closure_status},
        )
    return _pass(
        CHECK_RECOVERY_CLOSURE_READY,
        "Recovery closure draft is ready for ledger preview.",
        {"closure_id": _str(closure.get("id"))},
    )


def _closure_integrity_check(
    closure: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerCheck:
    expected_digest = _expected_closure_digest(closure)
    actual_digest = _str(closure.get("closure_digest"))
    expected_id = f"recovery-closure-{expected_digest[:16]}" if expected_digest else ""
    missing_fields: list[str] = []
    for field_name in (
        "id",
        "status",
        "closure_type",
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
        "source_recovery_completion_audit_preview",
    ):
        value = closure.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif isinstance(value, Mapping) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, (list, Mapping)) and not _str(value):
            missing_fields.append(field_name)
    if (
        missing_fields
        or actual_digest != expected_digest
        or _str(closure.get("id")) != expected_id
        or _str(closure.get("closure_type")) != RECOVERY_CLOSURE_TYPE
        or _str(closure.get("closure_decision")) != RECOVERY_CLOSURE_DECISION
    ):
        return _fail(
            CHECK_RECOVERY_CLOSURE_INTEGRITY,
            "Recovery closure digest, id, type, decision, or required fields are invalid.",
            ("regenerate_recovery_closure_gate",),
            {
                "expected_closure_digest": expected_digest,
                "actual_closure_digest": actual_digest,
                "expected_closure_id": expected_id,
                "actual_closure_id": _str(closure.get("id")),
                "missing_fields": missing_fields,
            },
        )
    return _pass(
        CHECK_RECOVERY_CLOSURE_INTEGRITY,
        "Recovery closure integrity checks pass.",
        {"closure_id": expected_id, "closure_digest": actual_digest},
    )


def _lifecycle_chain_check(
    closure: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerCheck:
    audit = _dict(closure.get("source_recovery_completion_audit_preview"))
    receipt = _dict(audit.get("source_recovery_completion_receipt"))
    executor = _dict(receipt.get("source_recovery_executor_preview"))
    mismatches: list[dict[str, str]] = []

    _compare(mismatches, "audit_preview_id", closure, "audit_preview_id", audit, "id")
    _compare(mismatches, "receipt_id", closure, "receipt_id", audit, "receipt_id")
    _compare(mismatches, "receipt_id", closure, "receipt_id", receipt, "id")
    _compare(
        mismatches,
        "executor_preview_id",
        closure,
        "executor_preview_id",
        audit,
        "executor_preview_id",
    )
    _compare(
        mismatches,
        "executor_preview_id",
        closure,
        "executor_preview_id",
        receipt,
        "executor_preview_id",
    )
    _compare(mismatches, "executor_preview_id", closure, "executor_preview_id", executor, "id")
    for field_name in ("decision_id", "execution_preview_id", "token_id", "lock_id", "route"):
        _compare(mismatches, field_name, closure, field_name, audit, field_name)
        _compare(mismatches, field_name, closure, field_name, receipt, field_name)
        _compare(mismatches, field_name, closure, field_name, executor, field_name)

    audit_steps = _audit_steps(audit)
    non_pass_steps = [
        _str(step.get("id")) or f"step-{index}"
        for index, step in enumerate(audit_steps, start=1)
        if _str(step.get("status")) != RECOVERY_AUDIT_STEP_STATUS_PASS
    ]
    missing_sources = [
        name
        for name, source in (
            ("source_recovery_completion_audit_preview", audit),
            ("source_recovery_completion_receipt", receipt),
            ("source_recovery_executor_preview", executor),
        )
        if not source
    ]
    if (
        missing_sources
        or mismatches
        or _str(audit.get("status")) != RECOVERY_COMPLETION_AUDIT_STATUS_READY
        or audit.get("observed_state_supplied") is not True
        or non_pass_steps
    ):
        return _fail(
            CHECK_RECOVERY_LIFECYCLE_CHAIN,
            "Recovery closure lifecycle chain is incomplete or inconsistent.",
            ("regenerate_recovery_closure_gate",),
            {
                "missing_sources": missing_sources,
                "mismatches": mismatches,
                "audit_status": _str(audit.get("status")),
                "observed_state_supplied": audit.get("observed_state_supplied") is True,
                "non_pass_audit_step_ids": non_pass_steps,
            },
        )
    return _pass(
        CHECK_RECOVERY_LIFECYCLE_CHAIN,
        "Recovery closure lifecycle chain is internally consistent.",
        {
            "closure_id": _str(closure.get("id")),
            "audit_preview_id": _str(audit.get("id")),
            "receipt_id": _str(receipt.get("id")),
            "executor_preview_id": _str(executor.get("id")),
            "audit_step_count": len(audit_steps),
        },
    )


def _ledger_requirements_check(
    closure: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerCheck:
    milestones = _milestones(closure)
    missing_stages = [
        stage
        for stage in _required_milestone_stages()
        if not any(
            milestone.stage == stage and milestone.reference_id for milestone in milestones
        )
    ]
    missing_lists = [
        field_name
        for field_name in (
            "operation_ids",
            "recovery_step_ids",
            "future_mutation_step_ids",
            "audit_step_ids",
        )
        if not _list_of_str(closure.get(field_name))
    ]
    missing_flags = [
        field_name
        for field_name in (
            "would_close_recovery_loop",
            "would_mark_recovery_resolved",
            "would_preserve_audit_evidence",
        )
        if closure.get(field_name) is not True
    ]
    if missing_stages or missing_lists or missing_flags:
        return _fail(
            CHECK_RECOVERY_LEDGER_REQUIREMENTS,
            "Recovery closure ledger requirements are incomplete.",
            ("regenerate_recovery_closure_gate",),
            {
                "missing_stages": missing_stages,
                "missing_lists": missing_lists,
                "missing_flags": missing_flags,
                "milestone_count": len(milestones),
            },
        )
    return _pass(
        CHECK_RECOVERY_LEDGER_REQUIREMENTS,
        "Recovery closure ledger has all lifecycle milestones and evidence ids.",
        {
            "milestone_count": len(milestones),
            "operation_count": len(_list_of_str(closure.get("operation_ids"))),
            "recovery_step_count": len(_list_of_str(closure.get("recovery_step_ids"))),
            "audit_step_count": len(_list_of_str(closure.get("audit_step_ids"))),
        },
    )


def _entry_from_closure(
    closure: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerEntry:
    operation_ids = tuple(_list_of_str(closure.get("operation_ids")))
    recovery_step_ids = tuple(_list_of_str(closure.get("recovery_step_ids")))
    future_mutation_step_ids = tuple(_list_of_str(closure.get("future_mutation_step_ids")))
    audit_step_ids = tuple(_list_of_str(closure.get("audit_step_ids")))
    milestones = tuple(_milestones(closure))
    entry_seed = {
        "entry_type": RECOVERY_CLOSURE_LEDGER_ENTRY_TYPE,
        "closure_id": _str(closure.get("id")),
        "closure_digest": _str(closure.get("closure_digest")),
        "audit_preview_id": _str(closure.get("audit_preview_id")),
        "receipt_id": _str(closure.get("receipt_id")),
        "executor_preview_id": _str(closure.get("executor_preview_id")),
        "decision_id": _str(closure.get("decision_id")),
        "execution_preview_id": _str(closure.get("execution_preview_id")),
        "token_id": _str(closure.get("token_id")),
        "lock_id": _str(closure.get("lock_id")),
        "route": _str(closure.get("route")),
        "operation_ids": operation_ids,
        "recovery_step_ids": recovery_step_ids,
        "future_mutation_step_ids": future_mutation_step_ids,
        "audit_step_ids": audit_step_ids,
        "milestone_ids": tuple(milestone.id for milestone in milestones),
    }
    entry_digest = _digest(entry_seed)
    entry_id = f"recovery-closure-ledger-{entry_digest[:16]}"
    return MemoryEvidenceRepairRecoveryClosureLedgerEntry(
        id=entry_id,
        entry_type=RECOVERY_CLOSURE_LEDGER_ENTRY_TYPE,
        status=RECOVERY_CLOSURE_LEDGER_ENTRY_STATUS_READY,
        closure_id=_str(closure.get("id")),
        closure_decision=_str(closure.get("closure_decision")),
        audit_preview_id=_str(closure.get("audit_preview_id")),
        receipt_id=_str(closure.get("receipt_id")),
        executor_preview_id=_str(closure.get("executor_preview_id")),
        decision_id=_str(closure.get("decision_id")),
        execution_preview_id=_str(closure.get("execution_preview_id")),
        token_id=_str(closure.get("token_id")),
        lock_id=_str(closure.get("lock_id")),
        route=_str(closure.get("route")),
        operation_ids=operation_ids,
        recovery_step_ids=recovery_step_ids,
        future_mutation_step_ids=future_mutation_step_ids,
        audit_step_ids=audit_step_ids,
        closure_digest=_str(closure.get("closure_digest")),
        entry_digest=entry_digest,
        entry_preview=f"{entry_id}:{entry_digest[:12]}",
        milestones=milestones,
        would_record_ledger_entry=True,
        would_preserve_recovery_evidence=True,
        safety_note=(
            "Read-only recovery closure ledger preview. It records the intended "
            "audit trail for a future closure, but does not persist a ledger "
            "entry or mutate durable memory."
        ),
        source_recovery_closure=dict(closure),
    )


def _milestones(
    closure: Mapping[str, Any],
) -> list[MemoryEvidenceRepairRecoveryClosureLedgerMilestone]:
    audit = _dict(closure.get("source_recovery_completion_audit_preview"))
    receipt = _dict(audit.get("source_recovery_completion_receipt"))
    executor = _dict(receipt.get("source_recovery_executor_preview"))
    lock = _dict(executor.get("source_lock"))
    token_gate = _dict(lock.get("source_recovery_token_gate"))
    execution = _dict(token_gate.get("source_recovery_execution_preview"))
    decision = _dict(execution.get("source_recovery_decision"))
    sources = {
        "recovery_decision": decision,
        "recovery_execution_preview": execution,
        "recovery_approval_token": _dict(token_gate.get("source_token")),
        "recovery_write_lock": lock,
        "recovery_executor_preview": executor,
        "recovery_completion_receipt": receipt,
        "recovery_completion_audit": audit,
        "recovery_closure": dict(closure),
    }
    fallback_ids = {
        "recovery_decision": _str(closure.get("decision_id")),
        "recovery_execution_preview": _str(closure.get("execution_preview_id")),
        "recovery_approval_token": _str(closure.get("token_id")),
        "recovery_write_lock": _str(closure.get("lock_id")),
        "recovery_executor_preview": _str(closure.get("executor_preview_id")),
        "recovery_completion_receipt": _str(closure.get("receipt_id")),
        "recovery_completion_audit": _str(closure.get("audit_preview_id")),
        "recovery_closure": _str(closure.get("id")),
    }
    summaries = {
        "recovery_decision": "Recovery route decision selected.",
        "recovery_execution_preview": "Manual recovery execution preview prepared.",
        "recovery_approval_token": "Recovery human approval token verified.",
        "recovery_write_lock": "Recovery write lock drafted for the manual window.",
        "recovery_executor_preview": "Recovery executor preview completed.",
        "recovery_completion_receipt": "Recovery completion receipt drafted.",
        "recovery_completion_audit": "Observed recovery completion audit passed.",
        "recovery_closure": "Recovery closure draft is ready to close the loop.",
    }
    milestones: list[MemoryEvidenceRepairRecoveryClosureLedgerMilestone] = []
    for sequence, stage in enumerate(_required_milestone_stages(), start=1):
        source = sources.get(stage, {})
        reference_id = _str(source.get("id")) or fallback_ids.get(stage, "")
        status = _str(source.get("status")) or _str(closure.get("status"))
        milestone_id = _milestone_id(stage, reference_id, sequence)
        milestones.append(
            MemoryEvidenceRepairRecoveryClosureLedgerMilestone(
                id=milestone_id,
                sequence=sequence,
                stage=stage,
                reference_id=reference_id,
                status=status,
                summary=summaries[stage],
                source=_source_summary(source, fallback_id=reference_id),
            )
        )
    return milestones


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


def _source_summary(source: Mapping[str, Any], *, fallback_id: str) -> dict[str, Any]:
    return {
        "id": _str(source.get("id")) or fallback_id,
        "status": _str(source.get("status")),
        "route": _str(source.get("route")),
        "type": _str(source.get("closure_type"))
        or _str(source.get("receipt_type"))
        or _str(source.get("lock_type")),
    }


def _milestone_id(stage: str, reference_id: str, sequence: int) -> str:
    digest = _digest({"stage": stage, "reference_id": reference_id, "sequence": sequence})
    return f"recovery-closure-milestone-{digest[:16]}"


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
    recovery_closure_gate: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerPreviewReport:
    return MemoryEvidenceRepairRecoveryClosureLedgerPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_LEDGER_STATUS_NO_ACTION_NEEDED,
        checks=(),
        entries=(),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_closure_gate=dict(recovery_closure_gate),
    )


def _blocked_report(
    *,
    recovery_closure_gate: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairRecoveryClosureLedgerCheck, ...],
) -> MemoryEvidenceRepairRecoveryClosureLedgerPreviewReport:
    return MemoryEvidenceRepairRecoveryClosureLedgerPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_LEDGER_STATUS_BLOCKED,
        checks=checks,
        entries=(),
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
        source_recovery_closure_gate=dict(recovery_closure_gate),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureLedgerCheck:
    return MemoryEvidenceRepairRecoveryClosureLedgerCheck(
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
) -> MemoryEvidenceRepairRecoveryClosureLedgerCheck:
    return MemoryEvidenceRepairRecoveryClosureLedgerCheck(
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
