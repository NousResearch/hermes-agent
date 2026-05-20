"""Read-only recovery closure governance seal audit preview.

This layer audits a recovery closure governance seal draft so later governance
layers can verify the seal itself before any future handoff or freeze. It never
writes durable memory, records a real audit, or mutates a real seal.
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
from agent.memory_evidence_repair_recovery_closure_governance_seal_preview import (
    RECOVERY_CLOSURE_GOVERNANCE_DECISION,
    RECOVERY_CLOSURE_GOVERNANCE_SCOPE,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_DRAFT_STATUS_READY,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_NO_ACTION_NEEDED,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_READY,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_TYPE,
)
from agent.memory_evidence_repair_recovery_closure_ledger_audit_preview import (
    LEDGER_AUDIT_STEP_STATUS_PASS,
    RECOVERY_CLOSURE_LEDGER_AUDIT_PREVIEW_STATUS_READY,
)
from agent.memory_evidence_repair_recovery_closure_ledger_preview import (
    RECOVERY_CLOSURE_LEDGER_ENTRY_STATUS_READY,
)


RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_READY = (
    "recovery_closure_governance_seal_audit_preview_ready"
)
RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_BLOCKED = "blocked"
RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_NO_ACTION_NEEDED = "no_action_needed"
RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_PREVIEW_STATUS_READY = (
    "recovery_closure_governance_seal_audit_ready"
)

GOVERNANCE_SEAL_AUDIT_STEP_STATUS_PASS = "pass"
GOVERNANCE_SEAL_AUDIT_STEP_STATUS_FAIL = "fail"

CHECK_RECOVERY_GOVERNANCE_SEAL_READY = "recovery_governance_seal_ready"
CHECK_RECOVERY_GOVERNANCE_SEAL_INTEGRITY = "recovery_governance_seal_integrity"
CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_SOURCE_CHAIN = (
    "recovery_governance_seal_audit_source_chain"
)
CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_REQUIREMENTS = (
    "recovery_governance_seal_audit_requirements"
)


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditCheck:
    """One read-only governance seal audit readiness check."""

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
class MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditStep:
    """One structural audit step for a future governance seal."""

    id: str
    sequence: int
    category: str
    assertion: str
    seal_id: str
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
            "seal_id": self.seal_id,
            "reference_id": self.reference_id,
            "expected_signal": dict(self.expected_signal),
            "observed_signal": dict(self.observed_signal),
            "status": self.status,
            "required_action_on_fail": self.required_action_on_fail,
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditPreview:
    """One read-only recovery closure governance seal audit preview."""

    id: str
    status: str
    seal_id: str
    ledger_audit_preview_id: str
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
    seal_audit_step_ids: tuple[str, ...]
    seal_digest: str
    ledger_audit_digest: str
    ledger_entry_digest: str
    closure_digest: str
    audit_digest: str
    audit_preview: str
    audit_steps: tuple[MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditStep, ...]
    would_verify_seal_integrity: bool
    would_verify_governance_handoff: bool
    would_verify_source_audit_evidence: bool
    would_verify_read_only_constraints: bool
    safety_note: str
    source_recovery_closure_governance_seal: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "seal_id": self.seal_id,
            "ledger_audit_preview_id": self.ledger_audit_preview_id,
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
            "seal_audit_step_ids": list(self.seal_audit_step_ids),
            "seal_digest": self.seal_digest,
            "ledger_audit_digest": self.ledger_audit_digest,
            "ledger_entry_digest": self.ledger_entry_digest,
            "closure_digest": self.closure_digest,
            "audit_digest": self.audit_digest,
            "audit_preview": self.audit_preview,
            "audit_steps": [step.to_dict() for step in self.audit_steps],
            "would_verify_seal_integrity": self.would_verify_seal_integrity,
            "would_verify_governance_handoff": self.would_verify_governance_handoff,
            "would_verify_source_audit_evidence": self.would_verify_source_audit_evidence,
            "would_verify_read_only_constraints": self.would_verify_read_only_constraints,
            "safety_note": self.safety_note,
            "source_recovery_closure_governance_seal": dict(
                self.source_recovery_closure_governance_seal
            ),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditPreviewReport:
    """Read-only recovery closure governance seal audit preview report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditCheck, ...]
    previews: tuple[MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditPreview, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_recovery_closure_governance_seal_report: dict[str, Any]

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
                GOVERNANCE_SEAL_AUDIT_STEP_STATUS_PASS,
                0,
            ),
            "observed_fail_step_count": by_audit_step_status.get(
                GOVERNANCE_SEAL_AUDIT_STEP_STATUS_FAIL,
                0,
            ),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "recovery_closure_governance_seal_audit_preview_available": bool(
                self.previews
            ),
            "recovery_closure_governance_seal_audit_ready": (
                self.status == RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_READY
            ),
            "has_blocks": (
                self.status == RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_BLOCKED
            ),
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
            "source_recovery_closure_governance_seal_report": dict(
                self.source_recovery_closure_governance_seal_report
            ),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_closure_governance_seal_audit_preview(
    *,
    recovery_closure_governance_seal: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditPreviewReport:
    """Build a read-only audit preview from a recovery governance seal report."""

    recovery_closure_governance_seal = (
        recovery_closure_governance_seal
        if isinstance(recovery_closure_governance_seal, Mapping)
        else {}
    )
    report_status = _str(recovery_closure_governance_seal.get("status"))
    seal = _extract_seal(recovery_closure_governance_seal)

    if not seal:
        if (
            not recovery_closure_governance_seal
            or report_status == RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_NO_ACTION_NEEDED
        ):
            return _empty_report(
                recovery_closure_governance_seal=recovery_closure_governance_seal
            )
        source_blocking = tuple(
            _list_of_str(recovery_closure_governance_seal.get("blocking_reasons"))
        )
        source_actions = tuple(
            _list_of_str(recovery_closure_governance_seal.get("required_actions"))
        )
        return _blocked_report(
            recovery_closure_governance_seal=recovery_closure_governance_seal,
            checks=(
                _fail(
                    CHECK_RECOVERY_GOVERNANCE_SEAL_READY,
                    "No recovery closure governance seal draft was supplied."
                    if not source_blocking
                    else "; ".join(source_blocking),
                    source_actions
                    or ("produce_recovery_closure_governance_seal_preview",),
                    {"recovery_closure_governance_seal_status": report_status},
                ),
            ),
        )

    checks = (
        _seal_ready_check(recovery_closure_governance_seal, seal),
        _seal_integrity_check(seal),
        _seal_source_chain_check(seal),
        _audit_requirements_check(seal),
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
        return MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditPreviewReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_BLOCKED,
            checks=checks,
            previews=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            source_recovery_closure_governance_seal_report=dict(
                recovery_closure_governance_seal
            ),
        )

    preview = _preview_from_seal(seal)
    return MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_READY,
        checks=checks,
        previews=(preview,),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_closure_governance_seal_report=dict(
            recovery_closure_governance_seal
        ),
    )


def empty_evidence_repair_recovery_closure_governance_seal_audit_preview() -> MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditPreviewReport:
    """Return an empty read-only recovery closure governance seal audit preview."""

    return _empty_report(recovery_closure_governance_seal={})


def _extract_seal(recovery_closure_governance_seal: Mapping[str, Any]) -> dict[str, Any]:
    seals = recovery_closure_governance_seal.get("seals")
    if isinstance(seals, list):
        for seal in seals:
            if isinstance(seal, Mapping):
                return dict(seal)
    if _str(recovery_closure_governance_seal.get("id")).startswith(
        "recovery-closure-governance-seal-"
    ):
        return dict(recovery_closure_governance_seal)
    return {}


def _seal_ready_check(
    recovery_closure_governance_seal: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditCheck:
    report_status = _str(recovery_closure_governance_seal.get("status"))
    seal_status = _str(seal.get("status"))
    if report_status and report_status != RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_GOVERNANCE_SEAL_READY,
            "Recovery closure governance seal report is not ready for audit preview.",
            tuple(_list_of_str(recovery_closure_governance_seal.get("required_actions")))
            or ("produce_ready_recovery_closure_governance_seal_preview",),
            {"report_status": report_status, "seal_status": seal_status},
        )
    if seal_status != RECOVERY_CLOSURE_GOVERNANCE_SEAL_DRAFT_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_GOVERNANCE_SEAL_READY,
            "Recovery closure governance seal draft is not ready for audit preview.",
            ("regenerate_recovery_closure_governance_seal_preview",),
            {"report_status": report_status, "seal_status": seal_status},
        )
    return _pass(
        CHECK_RECOVERY_GOVERNANCE_SEAL_READY,
        "Recovery closure governance seal draft is ready for audit preview.",
        {"seal_id": _str(seal.get("id"))},
    )


def _seal_integrity_check(
    seal: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditCheck:
    expected_digest = _expected_seal_digest(seal)
    actual_digest = _str(seal.get("seal_digest"))
    expected_id = (
        f"recovery-closure-governance-seal-{expected_digest[:16]}"
        if expected_digest
        else ""
    )
    missing_fields: list[str] = []
    for field_name in (
        "id",
        "seal_type",
        "status",
        "governance_decision",
        "governance_scope",
        "ledger_audit_preview_id",
        "ledger_entry_id",
        "closure_id",
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
        "completion_audit_step_ids",
        "milestone_ids",
        "ledger_audit_step_ids",
        "ledger_entry_digest",
        "closure_digest",
        "ledger_audit_digest",
        "seal_digest",
        "source_recovery_closure_ledger_audit_preview",
    ):
        value = seal.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif isinstance(value, Mapping) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, (list, Mapping)) and not _str(value):
            missing_fields.append(field_name)
    if (
        missing_fields
        or actual_digest != expected_digest
        or _str(seal.get("id")) != expected_id
        or _str(seal.get("seal_type")) != RECOVERY_CLOSURE_GOVERNANCE_SEAL_TYPE
        or _str(seal.get("governance_decision"))
        != RECOVERY_CLOSURE_GOVERNANCE_DECISION
        or _str(seal.get("governance_scope")) != RECOVERY_CLOSURE_GOVERNANCE_SCOPE
    ):
        return _fail(
            CHECK_RECOVERY_GOVERNANCE_SEAL_INTEGRITY,
            "Recovery closure governance seal digest, id, type, scope, or required fields are invalid.",
            ("regenerate_recovery_closure_governance_seal_preview",),
            {
                "expected_seal_digest": expected_digest,
                "actual_seal_digest": actual_digest,
                "expected_seal_id": expected_id,
                "actual_seal_id": _str(seal.get("id")),
                "missing_fields": missing_fields,
            },
        )
    return _pass(
        CHECK_RECOVERY_GOVERNANCE_SEAL_INTEGRITY,
        "Recovery closure governance seal integrity checks pass.",
        {"seal_id": expected_id, "seal_digest": actual_digest},
    )


def _seal_source_chain_check(
    seal: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditCheck:
    ledger_audit = _dict(seal.get("source_recovery_closure_ledger_audit_preview"))
    entry = _dict(ledger_audit.get("source_recovery_closure_ledger_entry"))
    mismatches: list[dict[str, str]] = []
    for field_name in (
        "ledger_audit_preview_id",
        "ledger_entry_id",
        "closure_id",
        "audit_preview_id",
        "receipt_id",
        "executor_preview_id",
        "decision_id",
        "execution_preview_id",
        "token_id",
        "lock_id",
        "route",
        "ledger_entry_digest",
        "closure_digest",
    ):
        audit_key = "id" if field_name == "ledger_audit_preview_id" else field_name
        if field_name == "ledger_audit_digest":
            audit_key = "audit_digest"
        _compare(mismatches, field_name, seal, field_name, ledger_audit, audit_key)
    if _str(seal.get("ledger_audit_digest")) != _str(ledger_audit.get("audit_digest")):
        mismatches.append(
            {
                "field": "ledger_audit_digest",
                "left": _str(seal.get("ledger_audit_digest")),
                "right": _str(ledger_audit.get("audit_digest")),
                "right_key": "source_recovery_closure_ledger_audit_preview.audit_digest",
            }
        )
    audit_steps = _audit_steps(ledger_audit)
    non_pass_steps = [
        _str(step.get("id")) or f"ledger-audit-step-{index}"
        for index, step in enumerate(audit_steps, start=1)
        if _str(step.get("status")) != LEDGER_AUDIT_STEP_STATUS_PASS
    ]
    missing_sources = [
        name
        for name, source in (
            ("source_recovery_closure_ledger_audit_preview", ledger_audit),
            ("source_recovery_closure_ledger_entry", entry),
        )
        if not source
    ]
    if (
        missing_sources
        or mismatches
        or _str(ledger_audit.get("status"))
        != RECOVERY_CLOSURE_LEDGER_AUDIT_PREVIEW_STATUS_READY
        or _str(entry.get("status")) != RECOVERY_CLOSURE_LEDGER_ENTRY_STATUS_READY
        or non_pass_steps
    ):
        return _fail(
            CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_SOURCE_CHAIN,
            "Recovery closure governance seal source chain is incomplete or inconsistent.",
            ("regenerate_recovery_closure_governance_seal_preview",),
            {
                "missing_sources": missing_sources,
                "mismatches": mismatches,
                "ledger_audit_status": _str(ledger_audit.get("status")),
                "entry_status": _str(entry.get("status")),
                "non_pass_ledger_audit_step_ids": non_pass_steps,
            },
        )
    return _pass(
        CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_SOURCE_CHAIN,
        "Recovery closure governance seal source chain is internally consistent.",
        {
            "seal_id": _str(seal.get("id")),
            "ledger_audit_preview_id": _str(seal.get("ledger_audit_preview_id")),
            "ledger_entry_id": _str(seal.get("ledger_entry_id")),
            "ledger_audit_step_count": len(audit_steps),
        },
    )


def _audit_requirements_check(
    seal: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditCheck:
    required_flags = (
        "would_freeze_recovery_closure",
        "would_preserve_audit_evidence",
        "would_enable_governed_handoff",
        "would_block_unreviewed_mutation",
    )
    missing_flags = [field_name for field_name in required_flags if seal.get(field_name) is not True]
    missing_lists = [
        field_name
        for field_name in (
            "operation_ids",
            "recovery_step_ids",
            "future_mutation_step_ids",
            "completion_audit_step_ids",
            "milestone_ids",
            "ledger_audit_step_ids",
        )
        if not _list_of_str(seal.get(field_name))
    ]
    ledger_audit = _dict(seal.get("source_recovery_closure_ledger_audit_preview"))
    source_flags = (
        "would_verify_ledger_entry_integrity",
        "would_verify_closure_source_integrity",
        "would_verify_lifecycle_milestones",
        "would_verify_completion_audit_evidence",
    )
    missing_source_flags = [
        field_name for field_name in source_flags if ledger_audit.get(field_name) is not True
    ]
    if missing_flags or missing_lists or missing_source_flags:
        return _fail(
            CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_REQUIREMENTS,
            "Recovery closure governance seal audit requirements are incomplete.",
            ("regenerate_recovery_closure_governance_seal_preview",),
            {
                "missing_flags": missing_flags,
                "missing_lists": missing_lists,
                "missing_source_flags": missing_source_flags,
            },
        )
    return _pass(
        CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_REQUIREMENTS,
        "Recovery closure governance seal audit can verify integrity, handoff, source evidence, and read-only constraints.",
        {
            "ledger_audit_step_count": len(_audit_steps(ledger_audit)),
            "milestone_count": len(_list_of_str(seal.get("milestone_ids"))),
        },
    )


def _preview_from_seal(
    seal: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditPreview:
    audit_steps = tuple(_audit_steps_from_seal(seal))
    seal_audit_step_ids = tuple(step.id for step in audit_steps)
    operation_ids = tuple(_list_of_str(seal.get("operation_ids")))
    recovery_step_ids = tuple(_list_of_str(seal.get("recovery_step_ids")))
    future_mutation_step_ids = tuple(_list_of_str(seal.get("future_mutation_step_ids")))
    completion_audit_step_ids = tuple(_list_of_str(seal.get("completion_audit_step_ids")))
    milestone_ids = tuple(_list_of_str(seal.get("milestone_ids")))
    ledger_audit_step_ids = tuple(_list_of_str(seal.get("ledger_audit_step_ids")))
    audit_seed = {
        "seal_id": _str(seal.get("id")),
        "ledger_audit_preview_id": _str(seal.get("ledger_audit_preview_id")),
        "ledger_entry_id": _str(seal.get("ledger_entry_id")),
        "closure_id": _str(seal.get("closure_id")),
        "audit_preview_id": _str(seal.get("audit_preview_id")),
        "receipt_id": _str(seal.get("receipt_id")),
        "executor_preview_id": _str(seal.get("executor_preview_id")),
        "decision_id": _str(seal.get("decision_id")),
        "execution_preview_id": _str(seal.get("execution_preview_id")),
        "token_id": _str(seal.get("token_id")),
        "lock_id": _str(seal.get("lock_id")),
        "route": _str(seal.get("route")),
        "operation_ids": operation_ids,
        "recovery_step_ids": recovery_step_ids,
        "future_mutation_step_ids": future_mutation_step_ids,
        "completion_audit_step_ids": completion_audit_step_ids,
        "milestone_ids": milestone_ids,
        "ledger_audit_step_ids": ledger_audit_step_ids,
        "seal_audit_step_ids": seal_audit_step_ids,
        "seal_digest": _str(seal.get("seal_digest")),
        "ledger_audit_digest": _str(seal.get("ledger_audit_digest")),
        "ledger_entry_digest": _str(seal.get("ledger_entry_digest")),
        "closure_digest": _str(seal.get("closure_digest")),
    }
    audit_digest = _digest(audit_seed)
    audit_id = f"recovery-closure-governance-seal-audit-{audit_digest[:16]}"
    return MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditPreview(
        id=audit_id,
        status=RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_PREVIEW_STATUS_READY,
        seal_id=_str(seal.get("id")),
        ledger_audit_preview_id=_str(seal.get("ledger_audit_preview_id")),
        ledger_entry_id=_str(seal.get("ledger_entry_id")),
        closure_id=_str(seal.get("closure_id")),
        audit_preview_id=_str(seal.get("audit_preview_id")),
        receipt_id=_str(seal.get("receipt_id")),
        executor_preview_id=_str(seal.get("executor_preview_id")),
        decision_id=_str(seal.get("decision_id")),
        execution_preview_id=_str(seal.get("execution_preview_id")),
        token_id=_str(seal.get("token_id")),
        lock_id=_str(seal.get("lock_id")),
        route=_str(seal.get("route")),
        operation_ids=operation_ids,
        recovery_step_ids=recovery_step_ids,
        future_mutation_step_ids=future_mutation_step_ids,
        completion_audit_step_ids=completion_audit_step_ids,
        milestone_ids=milestone_ids,
        ledger_audit_step_ids=ledger_audit_step_ids,
        seal_audit_step_ids=seal_audit_step_ids,
        seal_digest=_str(seal.get("seal_digest")),
        ledger_audit_digest=_str(seal.get("ledger_audit_digest")),
        ledger_entry_digest=_str(seal.get("ledger_entry_digest")),
        closure_digest=_str(seal.get("closure_digest")),
        audit_digest=audit_digest,
        audit_preview=f"{audit_id}:{audit_digest[:12]}",
        audit_steps=audit_steps,
        would_verify_seal_integrity=True,
        would_verify_governance_handoff=True,
        would_verify_source_audit_evidence=True,
        would_verify_read_only_constraints=True,
        safety_note=(
            "Read-only recovery closure governance seal audit preview. It checks "
            "a future governance seal's traceability and handoff controls, but "
            "does not record an audit or mutate durable memory."
        ),
        source_recovery_closure_governance_seal=dict(seal),
    )


def _audit_steps_from_seal(
    seal: Mapping[str, Any],
) -> list[MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditStep]:
    ledger_audit = _dict(seal.get("source_recovery_closure_ledger_audit_preview"))
    rows = (
        (
            "seal_integrity",
            "seal_digest_valid",
            _str(seal.get("id")),
            {"expected_seal_digest": _expected_seal_digest(seal)},
            {"actual_seal_digest": _str(seal.get("seal_digest"))},
        ),
        (
            "ledger_audit_source",
            "ledger_audit_digest_valid",
            _str(seal.get("ledger_audit_preview_id")),
            {"expected_ledger_audit_digest": _expected_ledger_audit_digest(ledger_audit)},
            {"actual_ledger_audit_digest": _str(seal.get("ledger_audit_digest"))},
        ),
        (
            "ledger_entry_source",
            "ledger_entry_reference_consistent",
            _str(seal.get("ledger_entry_id")),
            {"ledger_entry_id": _str(seal.get("ledger_entry_id"))},
            {"source_ledger_entry_id": _str(ledger_audit.get("ledger_entry_id"))},
        ),
        (
            "closure_trace",
            "closure_reference_consistent",
            _str(seal.get("closure_id")),
            {"closure_id": _str(seal.get("closure_id"))},
            {"source_closure_id": _str(ledger_audit.get("closure_id"))},
        ),
        (
            "governance_controls",
            "governance_handoff_controls_present",
            _str(seal.get("id")),
            {
                "would_freeze_recovery_closure": True,
                "would_enable_governed_handoff": True,
            },
            {
                "would_freeze_recovery_closure": seal.get("would_freeze_recovery_closure")
                is True,
                "would_enable_governed_handoff": seal.get("would_enable_governed_handoff")
                is True,
            },
        ),
        (
            "read_only",
            "seal_is_preview_only",
            _str(seal.get("id")),
            {"would_block_unreviewed_mutation": True},
            {
                "would_block_unreviewed_mutation": seal.get(
                    "would_block_unreviewed_mutation"
                )
                is True
            },
        ),
    )
    return [
        _audit_step(
            sequence=index,
            category=category,
            assertion=assertion,
            seal=seal,
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
    seal: Mapping[str, Any],
    reference_id: str,
    expected_signal: Mapping[str, Any],
    observed_signal: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditStep:
    step_seed = {
        "sequence": sequence,
        "category": category,
        "assertion": assertion,
        "seal_id": _str(seal.get("id")),
        "reference_id": _str(reference_id),
    }
    step_id = f"recovery-closure-governance-seal-audit-step-{_digest(step_seed)[:16]}"
    return MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditStep(
        id=step_id,
        sequence=sequence,
        category=category,
        assertion=assertion,
        seal_id=_str(seal.get("id")),
        reference_id=_str(reference_id),
        expected_signal=dict(expected_signal),
        observed_signal=dict(observed_signal),
        status=GOVERNANCE_SEAL_AUDIT_STEP_STATUS_PASS,
        required_action_on_fail="regenerate_recovery_closure_governance_seal_preview",
    )


def _expected_seal_digest(seal: Mapping[str, Any]) -> str:
    seed = {
        "seal_type": RECOVERY_CLOSURE_GOVERNANCE_SEAL_TYPE,
        "governance_decision": RECOVERY_CLOSURE_GOVERNANCE_DECISION,
        "governance_scope": RECOVERY_CLOSURE_GOVERNANCE_SCOPE,
        "ledger_audit_preview_id": _str(seal.get("ledger_audit_preview_id")),
        "ledger_entry_id": _str(seal.get("ledger_entry_id")),
        "closure_id": _str(seal.get("closure_id")),
        "audit_preview_id": _str(seal.get("audit_preview_id")),
        "receipt_id": _str(seal.get("receipt_id")),
        "executor_preview_id": _str(seal.get("executor_preview_id")),
        "decision_id": _str(seal.get("decision_id")),
        "execution_preview_id": _str(seal.get("execution_preview_id")),
        "token_id": _str(seal.get("token_id")),
        "lock_id": _str(seal.get("lock_id")),
        "route": _str(seal.get("route")),
        "operation_ids": tuple(_list_of_str(seal.get("operation_ids"))),
        "recovery_step_ids": tuple(_list_of_str(seal.get("recovery_step_ids"))),
        "future_mutation_step_ids": tuple(
            _list_of_str(seal.get("future_mutation_step_ids"))
        ),
        "completion_audit_step_ids": tuple(
            _list_of_str(seal.get("completion_audit_step_ids"))
        ),
        "milestone_ids": tuple(_list_of_str(seal.get("milestone_ids"))),
        "ledger_audit_step_ids": tuple(
            _list_of_str(seal.get("ledger_audit_step_ids"))
        ),
        "ledger_entry_digest": _str(seal.get("ledger_entry_digest")),
        "closure_digest": _str(seal.get("closure_digest")),
        "ledger_audit_digest": _str(seal.get("ledger_audit_digest")),
    }
    return _digest(seed)


def _expected_ledger_audit_digest(preview: Mapping[str, Any]) -> str:
    seed = {
        "ledger_entry_id": _str(preview.get("ledger_entry_id")),
        "closure_id": _str(preview.get("closure_id")),
        "audit_preview_id": _str(preview.get("audit_preview_id")),
        "receipt_id": _str(preview.get("receipt_id")),
        "executor_preview_id": _str(preview.get("executor_preview_id")),
        "decision_id": _str(preview.get("decision_id")),
        "execution_preview_id": _str(preview.get("execution_preview_id")),
        "token_id": _str(preview.get("token_id")),
        "lock_id": _str(preview.get("lock_id")),
        "route": _str(preview.get("route")),
        "operation_ids": tuple(_list_of_str(preview.get("operation_ids"))),
        "recovery_step_ids": tuple(_list_of_str(preview.get("recovery_step_ids"))),
        "future_mutation_step_ids": tuple(
            _list_of_str(preview.get("future_mutation_step_ids"))
        ),
        "completion_audit_step_ids": tuple(
            _list_of_str(preview.get("completion_audit_step_ids"))
        ),
        "milestone_ids": tuple(_list_of_str(preview.get("milestone_ids"))),
        "ledger_audit_step_ids": tuple(
            _list_of_str(preview.get("ledger_audit_step_ids"))
        ),
        "entry_digest": _str(preview.get("ledger_entry_digest")),
        "closure_digest": _str(preview.get("closure_digest")),
    }
    return _digest(seed)


def _audit_steps(preview: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        step
        for step in _list(preview.get("audit_steps"))
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
    recovery_closure_governance_seal: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditPreviewReport:
    return MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_NO_ACTION_NEEDED,
        checks=(),
        previews=(),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_closure_governance_seal_report=dict(
            recovery_closure_governance_seal
        ),
    )


def _blocked_report(
    *,
    recovery_closure_governance_seal: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditCheck, ...],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditPreviewReport:
    return MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_BLOCKED,
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
        source_recovery_closure_governance_seal_report=dict(
            recovery_closure_governance_seal
        ),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditCheck:
    return MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditCheck(
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
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditCheck:
    return MemoryEvidenceRepairRecoveryClosureGovernanceSealAuditCheck(
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
