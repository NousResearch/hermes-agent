"""Read-only recovery closure finalization readiness audit preview.

This layer audits a recovery closure finalization readiness draft so a later
manual finalization path can trust the gate that would allow closure. It never
writes durable memory, records a real audit, or finalizes a recovery.
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
from agent.memory_evidence_repair_recovery_closure_finalization_readiness_preview import (
    RECOVERY_CLOSURE_FINALIZATION_DECISION,
    RECOVERY_CLOSURE_FINALIZATION_DRAFT_STATUS_READY,
    RECOVERY_CLOSURE_FINALIZATION_SCOPE,
    RECOVERY_CLOSURE_FINALIZATION_STATUS_NO_ACTION_NEEDED,
    RECOVERY_CLOSURE_FINALIZATION_STATUS_READY,
    RECOVERY_CLOSURE_FINALIZATION_TYPE,
)
from agent.memory_evidence_repair_recovery_closure_governance_seal_audit_preview import (
    GOVERNANCE_SEAL_AUDIT_STEP_STATUS_PASS,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_PREVIEW_STATUS_READY,
)
from agent.memory_evidence_repair_recovery_closure_governance_seal_preview import (
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_DRAFT_STATUS_READY,
)
from agent.memory_evidence_repair_recovery_closure_ledger_audit_preview import (
    LEDGER_AUDIT_STEP_STATUS_PASS,
    RECOVERY_CLOSURE_LEDGER_AUDIT_PREVIEW_STATUS_READY,
)
from agent.memory_evidence_repair_recovery_closure_ledger_preview import (
    RECOVERY_CLOSURE_LEDGER_ENTRY_STATUS_READY,
)


RECOVERY_CLOSURE_FINALIZATION_READINESS_AUDIT_STATUS_READY = (
    "recovery_closure_finalization_readiness_audit_preview_ready"
)
RECOVERY_CLOSURE_FINALIZATION_READINESS_AUDIT_STATUS_BLOCKED = "blocked"
RECOVERY_CLOSURE_FINALIZATION_READINESS_AUDIT_STATUS_NO_ACTION_NEEDED = (
    "no_action_needed"
)
RECOVERY_CLOSURE_FINALIZATION_READINESS_AUDIT_PREVIEW_STATUS_READY = (
    "recovery_closure_finalization_readiness_audit_ready"
)

FINALIZATION_READINESS_AUDIT_STEP_STATUS_PASS = "pass"
FINALIZATION_READINESS_AUDIT_STEP_STATUS_FAIL = "fail"

CHECK_RECOVERY_FINALIZATION_READINESS_READY = "recovery_finalization_readiness_ready"
CHECK_RECOVERY_FINALIZATION_READINESS_INTEGRITY = (
    "recovery_finalization_readiness_integrity"
)
CHECK_RECOVERY_FINALIZATION_READINESS_SOURCE_CHAIN = (
    "recovery_finalization_readiness_source_chain"
)
CHECK_RECOVERY_FINALIZATION_READINESS_AUDIT_REQUIREMENTS = (
    "recovery_finalization_readiness_audit_requirements"
)


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditCheck:
    """One read-only finalization readiness audit check."""

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
class MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditStep:
    """One structural audit step for a future finalization readiness draft."""

    id: str
    sequence: int
    category: str
    assertion: str
    finalization_readiness_id: str
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
            "finalization_readiness_id": self.finalization_readiness_id,
            "reference_id": self.reference_id,
            "expected_signal": dict(self.expected_signal),
            "observed_signal": dict(self.observed_signal),
            "status": self.status,
            "required_action_on_fail": self.required_action_on_fail,
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditPreview:
    """One read-only recovery closure finalization readiness audit preview."""

    id: str
    status: str
    finalization_readiness_id: str
    seal_audit_preview_id: str
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
    readiness_audit_step_ids: tuple[str, ...]
    finalization_digest: str
    seal_audit_digest: str
    seal_digest: str
    ledger_audit_digest: str
    ledger_entry_digest: str
    closure_digest: str
    audit_digest: str
    audit_preview: str
    audit_steps: tuple[
        MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditStep, ...
    ]
    would_verify_finalization_integrity: bool
    would_verify_manual_handoff: bool
    would_verify_source_governance_chain: bool
    would_verify_read_only_constraints: bool
    safety_note: str
    source_recovery_closure_finalization_readiness: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "finalization_readiness_id": self.finalization_readiness_id,
            "seal_audit_preview_id": self.seal_audit_preview_id,
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
            "readiness_audit_step_ids": list(self.readiness_audit_step_ids),
            "finalization_digest": self.finalization_digest,
            "seal_audit_digest": self.seal_audit_digest,
            "seal_digest": self.seal_digest,
            "ledger_audit_digest": self.ledger_audit_digest,
            "ledger_entry_digest": self.ledger_entry_digest,
            "closure_digest": self.closure_digest,
            "audit_digest": self.audit_digest,
            "audit_preview": self.audit_preview,
            "audit_steps": [step.to_dict() for step in self.audit_steps],
            "would_verify_finalization_integrity": (
                self.would_verify_finalization_integrity
            ),
            "would_verify_manual_handoff": self.would_verify_manual_handoff,
            "would_verify_source_governance_chain": (
                self.would_verify_source_governance_chain
            ),
            "would_verify_read_only_constraints": (
                self.would_verify_read_only_constraints
            ),
            "safety_note": self.safety_note,
            "source_recovery_closure_finalization_readiness": dict(
                self.source_recovery_closure_finalization_readiness
            ),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditPreviewReport:
    """Read-only finalization readiness audit preview report."""

    generated_at: str
    status: str
    checks: tuple[
        MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditCheck, ...
    ]
    previews: tuple[
        MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditPreview, ...
    ]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_recovery_closure_finalization_readiness_report: dict[str, Any]

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
                FINALIZATION_READINESS_AUDIT_STEP_STATUS_PASS,
                0,
            ),
            "observed_fail_step_count": by_audit_step_status.get(
                FINALIZATION_READINESS_AUDIT_STEP_STATUS_FAIL,
                0,
            ),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "recovery_closure_finalization_readiness_audit_preview_available": bool(
                self.previews
            ),
            "recovery_closure_finalization_readiness_audit_ready": (
                self.status
                == RECOVERY_CLOSURE_FINALIZATION_READINESS_AUDIT_STATUS_READY
            ),
            "has_blocks": (
                self.status
                == RECOVERY_CLOSURE_FINALIZATION_READINESS_AUDIT_STATUS_BLOCKED
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
            "source_recovery_closure_finalization_readiness_report": dict(
                self.source_recovery_closure_finalization_readiness_report
            ),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_closure_finalization_readiness_audit_preview(
    *,
    recovery_closure_finalization_readiness: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditPreviewReport:
    """Build a read-only audit preview from a finalization readiness report."""

    recovery_closure_finalization_readiness = (
        recovery_closure_finalization_readiness
        if isinstance(recovery_closure_finalization_readiness, Mapping)
        else {}
    )
    report_status = _str(recovery_closure_finalization_readiness.get("status"))
    readiness = _extract_readiness(recovery_closure_finalization_readiness)

    if not readiness:
        if (
            not recovery_closure_finalization_readiness
            or report_status == RECOVERY_CLOSURE_FINALIZATION_STATUS_NO_ACTION_NEEDED
        ):
            return _empty_report(
                recovery_closure_finalization_readiness=(
                    recovery_closure_finalization_readiness
                )
            )
        source_blocking = tuple(
            _list_of_str(
                recovery_closure_finalization_readiness.get("blocking_reasons")
            )
        )
        source_actions = tuple(
            _list_of_str(
                recovery_closure_finalization_readiness.get("required_actions")
            )
        )
        return _blocked_report(
            recovery_closure_finalization_readiness=(
                recovery_closure_finalization_readiness
            ),
            checks=(
                _fail(
                    CHECK_RECOVERY_FINALIZATION_READINESS_READY,
                    "No recovery closure finalization readiness draft was supplied."
                    if not source_blocking
                    else "; ".join(source_blocking),
                    source_actions
                    or ("produce_recovery_closure_finalization_readiness_preview",),
                    {
                        "recovery_closure_finalization_readiness_status": (
                            report_status
                        )
                    },
                ),
            ),
        )

    checks = (
        _readiness_ready_check(recovery_closure_finalization_readiness, readiness),
        _readiness_integrity_check(readiness),
        _readiness_source_chain_check(readiness),
        _audit_requirements_check(readiness),
    )
    blocking_reasons = tuple(
        _dedupe_strings(
            check.reason for check in checks if check.status == CHECK_STATUS_FAIL
        )
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
        return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditPreviewReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_CLOSURE_FINALIZATION_READINESS_AUDIT_STATUS_BLOCKED,
            checks=checks,
            previews=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            source_recovery_closure_finalization_readiness_report=dict(
                recovery_closure_finalization_readiness
            ),
        )

    preview = _preview_from_readiness(readiness)
    return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_FINALIZATION_READINESS_AUDIT_STATUS_READY,
        checks=checks,
        previews=(preview,),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_closure_finalization_readiness_report=dict(
            recovery_closure_finalization_readiness
        ),
    )


def empty_evidence_repair_recovery_closure_finalization_readiness_audit_preview() -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditPreviewReport:
    """Return an empty read-only finalization readiness audit preview."""

    return _empty_report(recovery_closure_finalization_readiness={})


def _extract_readiness(
    recovery_closure_finalization_readiness: Mapping[str, Any],
) -> dict[str, Any]:
    readiness = recovery_closure_finalization_readiness.get("readiness")
    if isinstance(readiness, list):
        for item in readiness:
            if isinstance(item, Mapping):
                return dict(item)
    if _str(recovery_closure_finalization_readiness.get("id")).startswith(
        "recovery-closure-finalization-readiness-"
    ):
        return dict(recovery_closure_finalization_readiness)
    return {}


def _readiness_ready_check(
    recovery_closure_finalization_readiness: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditCheck:
    report_status = _str(recovery_closure_finalization_readiness.get("status"))
    readiness_status = _str(readiness.get("status"))
    if report_status and report_status != RECOVERY_CLOSURE_FINALIZATION_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_FINALIZATION_READINESS_READY,
            "Recovery closure finalization readiness report is not ready for audit preview.",
            tuple(
                _list_of_str(
                    recovery_closure_finalization_readiness.get("required_actions")
                )
            )
            or ("produce_ready_recovery_closure_finalization_readiness_preview",),
            {"report_status": report_status, "readiness_status": readiness_status},
        )
    if readiness_status != RECOVERY_CLOSURE_FINALIZATION_DRAFT_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_FINALIZATION_READINESS_READY,
            "Recovery closure finalization readiness draft is not ready for audit preview.",
            ("regenerate_recovery_closure_finalization_readiness_preview",),
            {"report_status": report_status, "readiness_status": readiness_status},
        )
    return _pass(
        CHECK_RECOVERY_FINALIZATION_READINESS_READY,
        "Recovery closure finalization readiness draft is ready for audit preview.",
        {"finalization_readiness_id": _str(readiness.get("id"))},
    )


def _readiness_integrity_check(
    readiness: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditCheck:
    expected_digest = _expected_finalization_digest(readiness)
    actual_digest = _str(readiness.get("finalization_digest"))
    expected_id = (
        f"recovery-closure-finalization-readiness-{expected_digest[:16]}"
        if expected_digest
        else ""
    )
    missing_fields: list[str] = []
    for field_name in (
        "id",
        "finalization_type",
        "status",
        "finalization_decision",
        "finalization_scope",
        "seal_audit_preview_id",
        "seal_id",
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
        "seal_audit_step_ids",
        "seal_digest",
        "seal_audit_digest",
        "ledger_audit_digest",
        "ledger_entry_digest",
        "closure_digest",
        "finalization_digest",
        "source_recovery_closure_governance_seal_audit_preview",
    ):
        value = readiness.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif isinstance(value, Mapping) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, (list, Mapping)) and not _str(value):
            missing_fields.append(field_name)
    if (
        missing_fields
        or actual_digest != expected_digest
        or _str(readiness.get("id")) != expected_id
        or _str(readiness.get("finalization_type"))
        != RECOVERY_CLOSURE_FINALIZATION_TYPE
        or _str(readiness.get("finalization_decision"))
        != RECOVERY_CLOSURE_FINALIZATION_DECISION
        or _str(readiness.get("finalization_scope"))
        != RECOVERY_CLOSURE_FINALIZATION_SCOPE
    ):
        return _fail(
            CHECK_RECOVERY_FINALIZATION_READINESS_INTEGRITY,
            "Recovery closure finalization readiness digest, id, type, scope, or required fields are invalid.",
            ("regenerate_recovery_closure_finalization_readiness_preview",),
            {
                "expected_finalization_digest": expected_digest,
                "actual_finalization_digest": actual_digest,
                "expected_finalization_readiness_id": expected_id,
                "actual_finalization_readiness_id": _str(readiness.get("id")),
                "missing_fields": missing_fields,
            },
        )
    return _pass(
        CHECK_RECOVERY_FINALIZATION_READINESS_INTEGRITY,
        "Recovery closure finalization readiness integrity checks pass.",
        {
            "finalization_readiness_id": expected_id,
            "finalization_digest": actual_digest,
        },
    )


def _readiness_source_chain_check(
    readiness: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditCheck:
    seal_audit = _dict(
        readiness.get("source_recovery_closure_governance_seal_audit_preview")
    )
    seal = _dict(seal_audit.get("source_recovery_closure_governance_seal"))
    ledger_audit = _dict(seal.get("source_recovery_closure_ledger_audit_preview"))
    entry = _dict(ledger_audit.get("source_recovery_closure_ledger_entry"))
    missing_sources = [
        name
        for name, source in (
            ("source_recovery_closure_governance_seal_audit_preview", seal_audit),
            ("source_recovery_closure_governance_seal", seal),
            ("source_recovery_closure_ledger_audit_preview", ledger_audit),
            ("source_recovery_closure_ledger_entry", entry),
        )
        if not source
    ]
    mismatches: list[dict[str, str]] = []
    for field_name in (
        "seal_audit_preview_id",
        "seal_id",
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
        "seal_digest",
        "seal_audit_digest",
        "ledger_audit_digest",
        "ledger_entry_digest",
        "closure_digest",
    ):
        source_key = "id" if field_name == "seal_audit_preview_id" else field_name
        if field_name == "seal_audit_digest":
            source_key = "audit_digest"
        _compare(mismatches, field_name, readiness, field_name, seal_audit, source_key)
    if _str(readiness.get("seal_id")) != _str(seal.get("id")):
        mismatches.append(
            {
                "field": "seal_id",
                "left": _str(readiness.get("seal_id")),
                "right": _str(seal.get("id")),
                "right_key": "source_recovery_closure_governance_seal.id",
            }
        )
    if _str(readiness.get("ledger_audit_preview_id")) != _str(ledger_audit.get("id")):
        mismatches.append(
            {
                "field": "ledger_audit_preview_id",
                "left": _str(readiness.get("ledger_audit_preview_id")),
                "right": _str(ledger_audit.get("id")),
                "right_key": "source_recovery_closure_ledger_audit_preview.id",
            }
        )
    if _str(readiness.get("ledger_entry_id")) != _str(entry.get("id")):
        mismatches.append(
            {
                "field": "ledger_entry_id",
                "left": _str(readiness.get("ledger_entry_id")),
                "right": _str(entry.get("id")),
                "right_key": "source_recovery_closure_ledger_entry.id",
            }
        )
    seal_audit_steps = _audit_steps(seal_audit)
    ledger_audit_steps = _audit_steps(ledger_audit)
    non_pass_seal_audit_steps = [
        _str(step.get("id")) or f"seal-audit-step-{index}"
        for index, step in enumerate(seal_audit_steps, start=1)
        if _str(step.get("status")) != GOVERNANCE_SEAL_AUDIT_STEP_STATUS_PASS
    ]
    non_pass_ledger_audit_steps = [
        _str(step.get("id")) or f"ledger-audit-step-{index}"
        for index, step in enumerate(ledger_audit_steps, start=1)
        if _str(step.get("status")) != LEDGER_AUDIT_STEP_STATUS_PASS
    ]
    if (
        missing_sources
        or mismatches
        or _str(seal_audit.get("status"))
        != RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_PREVIEW_STATUS_READY
        or _str(seal.get("status")) != RECOVERY_CLOSURE_GOVERNANCE_SEAL_DRAFT_STATUS_READY
        or _str(ledger_audit.get("status"))
        != RECOVERY_CLOSURE_LEDGER_AUDIT_PREVIEW_STATUS_READY
        or _str(entry.get("status")) != RECOVERY_CLOSURE_LEDGER_ENTRY_STATUS_READY
        or non_pass_seal_audit_steps
        or non_pass_ledger_audit_steps
    ):
        return _fail(
            CHECK_RECOVERY_FINALIZATION_READINESS_SOURCE_CHAIN,
            "Recovery closure finalization readiness source chain is incomplete or inconsistent.",
            ("regenerate_recovery_closure_finalization_readiness_preview",),
            {
                "missing_sources": missing_sources,
                "mismatches": mismatches,
                "seal_audit_status": _str(seal_audit.get("status")),
                "seal_status": _str(seal.get("status")),
                "ledger_audit_status": _str(ledger_audit.get("status")),
                "entry_status": _str(entry.get("status")),
                "non_pass_seal_audit_step_ids": non_pass_seal_audit_steps,
                "non_pass_ledger_audit_step_ids": non_pass_ledger_audit_steps,
            },
        )
    return _pass(
        CHECK_RECOVERY_FINALIZATION_READINESS_SOURCE_CHAIN,
        "Recovery closure finalization readiness source chain preserves finalization, seal audit, seal, ledger audit, and ledger evidence.",
        {
            "finalization_readiness_id": _str(readiness.get("id")),
            "seal_audit_preview_id": _str(readiness.get("seal_audit_preview_id")),
            "seal_id": _str(readiness.get("seal_id")),
            "ledger_audit_preview_id": _str(readiness.get("ledger_audit_preview_id")),
            "seal_audit_step_count": len(seal_audit_steps),
            "ledger_audit_step_count": len(ledger_audit_steps),
        },
    )


def _audit_requirements_check(
    readiness: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditCheck:
    required_readiness_flags = (
        "would_allow_final_closure",
        "would_preserve_full_governance_chain",
        "would_require_human_finalization",
        "would_keep_read_only_until_manual_commit",
    )
    missing_readiness_flags = [
        field_name
        for field_name in required_readiness_flags
        if readiness.get(field_name) is not True
    ]
    missing_lists = [
        field_name
        for field_name in (
            "operation_ids",
            "recovery_step_ids",
            "future_mutation_step_ids",
            "completion_audit_step_ids",
            "milestone_ids",
            "ledger_audit_step_ids",
            "seal_audit_step_ids",
        )
        if not _list_of_str(readiness.get(field_name))
    ]
    seal_audit = _dict(
        readiness.get("source_recovery_closure_governance_seal_audit_preview")
    )
    source_flags = (
        "would_verify_seal_integrity",
        "would_verify_governance_handoff",
        "would_verify_source_audit_evidence",
        "would_verify_read_only_constraints",
    )
    missing_source_flags = [
        field_name
        for field_name in source_flags
        if seal_audit.get(field_name) is not True
    ]
    categories = {_str(step.get("category")) for step in _audit_steps(seal_audit)}
    required_categories = {
        "seal_integrity",
        "ledger_audit_source",
        "ledger_entry_source",
        "closure_trace",
        "governance_controls",
        "read_only",
    }
    missing_categories = sorted(required_categories - categories)
    if (
        missing_readiness_flags
        or missing_lists
        or missing_source_flags
        or missing_categories
    ):
        return _fail(
            CHECK_RECOVERY_FINALIZATION_READINESS_AUDIT_REQUIREMENTS,
            "Recovery closure finalization readiness audit requirements are incomplete.",
            ("regenerate_recovery_closure_finalization_readiness_preview",),
            {
                "missing_readiness_flags": missing_readiness_flags,
                "missing_lists": missing_lists,
                "missing_source_flags": missing_source_flags,
                "missing_categories": missing_categories,
            },
        )
    return _pass(
        CHECK_RECOVERY_FINALIZATION_READINESS_AUDIT_REQUIREMENTS,
        "Recovery closure finalization readiness audit can verify integrity, manual handoff, source governance chain, and read-only constraints.",
        {
            "seal_audit_step_count": len(_audit_steps(seal_audit)),
            "milestone_count": len(_list_of_str(readiness.get("milestone_ids"))),
        },
    )


def _preview_from_readiness(
    readiness: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditPreview:
    audit_steps = tuple(_audit_steps_from_readiness(readiness))
    readiness_audit_step_ids = tuple(step.id for step in audit_steps)
    operation_ids = tuple(_list_of_str(readiness.get("operation_ids")))
    recovery_step_ids = tuple(_list_of_str(readiness.get("recovery_step_ids")))
    future_mutation_step_ids = tuple(
        _list_of_str(readiness.get("future_mutation_step_ids"))
    )
    completion_audit_step_ids = tuple(
        _list_of_str(readiness.get("completion_audit_step_ids"))
    )
    milestone_ids = tuple(_list_of_str(readiness.get("milestone_ids")))
    ledger_audit_step_ids = tuple(_list_of_str(readiness.get("ledger_audit_step_ids")))
    seal_audit_step_ids = tuple(_list_of_str(readiness.get("seal_audit_step_ids")))
    audit_seed = {
        "finalization_readiness_id": _str(readiness.get("id")),
        "seal_audit_preview_id": _str(readiness.get("seal_audit_preview_id")),
        "seal_id": _str(readiness.get("seal_id")),
        "ledger_audit_preview_id": _str(readiness.get("ledger_audit_preview_id")),
        "ledger_entry_id": _str(readiness.get("ledger_entry_id")),
        "closure_id": _str(readiness.get("closure_id")),
        "audit_preview_id": _str(readiness.get("audit_preview_id")),
        "receipt_id": _str(readiness.get("receipt_id")),
        "executor_preview_id": _str(readiness.get("executor_preview_id")),
        "decision_id": _str(readiness.get("decision_id")),
        "execution_preview_id": _str(readiness.get("execution_preview_id")),
        "token_id": _str(readiness.get("token_id")),
        "lock_id": _str(readiness.get("lock_id")),
        "route": _str(readiness.get("route")),
        "operation_ids": operation_ids,
        "recovery_step_ids": recovery_step_ids,
        "future_mutation_step_ids": future_mutation_step_ids,
        "completion_audit_step_ids": completion_audit_step_ids,
        "milestone_ids": milestone_ids,
        "ledger_audit_step_ids": ledger_audit_step_ids,
        "seal_audit_step_ids": seal_audit_step_ids,
        "readiness_audit_step_ids": readiness_audit_step_ids,
        "finalization_digest": _str(readiness.get("finalization_digest")),
        "seal_audit_digest": _str(readiness.get("seal_audit_digest")),
        "seal_digest": _str(readiness.get("seal_digest")),
        "ledger_audit_digest": _str(readiness.get("ledger_audit_digest")),
        "ledger_entry_digest": _str(readiness.get("ledger_entry_digest")),
        "closure_digest": _str(readiness.get("closure_digest")),
    }
    audit_digest = _digest(audit_seed)
    audit_id = f"recovery-closure-finalization-readiness-audit-{audit_digest[:16]}"
    return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditPreview(
        id=audit_id,
        status=RECOVERY_CLOSURE_FINALIZATION_READINESS_AUDIT_PREVIEW_STATUS_READY,
        finalization_readiness_id=_str(readiness.get("id")),
        seal_audit_preview_id=_str(readiness.get("seal_audit_preview_id")),
        seal_id=_str(readiness.get("seal_id")),
        ledger_audit_preview_id=_str(readiness.get("ledger_audit_preview_id")),
        ledger_entry_id=_str(readiness.get("ledger_entry_id")),
        closure_id=_str(readiness.get("closure_id")),
        audit_preview_id=_str(readiness.get("audit_preview_id")),
        receipt_id=_str(readiness.get("receipt_id")),
        executor_preview_id=_str(readiness.get("executor_preview_id")),
        decision_id=_str(readiness.get("decision_id")),
        execution_preview_id=_str(readiness.get("execution_preview_id")),
        token_id=_str(readiness.get("token_id")),
        lock_id=_str(readiness.get("lock_id")),
        route=_str(readiness.get("route")),
        operation_ids=operation_ids,
        recovery_step_ids=recovery_step_ids,
        future_mutation_step_ids=future_mutation_step_ids,
        completion_audit_step_ids=completion_audit_step_ids,
        milestone_ids=milestone_ids,
        ledger_audit_step_ids=ledger_audit_step_ids,
        seal_audit_step_ids=seal_audit_step_ids,
        readiness_audit_step_ids=readiness_audit_step_ids,
        finalization_digest=_str(readiness.get("finalization_digest")),
        seal_audit_digest=_str(readiness.get("seal_audit_digest")),
        seal_digest=_str(readiness.get("seal_digest")),
        ledger_audit_digest=_str(readiness.get("ledger_audit_digest")),
        ledger_entry_digest=_str(readiness.get("ledger_entry_digest")),
        closure_digest=_str(readiness.get("closure_digest")),
        audit_digest=audit_digest,
        audit_preview=f"{audit_id}:{audit_digest[:12]}",
        audit_steps=audit_steps,
        would_verify_finalization_integrity=True,
        would_verify_manual_handoff=True,
        would_verify_source_governance_chain=True,
        would_verify_read_only_constraints=True,
        safety_note=(
            "Read-only recovery closure finalization readiness audit preview. "
            "It verifies that a future manual finalization gate is traceable "
            "and guarded, but does not record an audit or mutate durable memory."
        ),
        source_recovery_closure_finalization_readiness=dict(readiness),
    )


def _audit_steps_from_readiness(
    readiness: Mapping[str, Any],
) -> list[MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditStep]:
    seal_audit = _dict(
        readiness.get("source_recovery_closure_governance_seal_audit_preview")
    )
    rows = (
        (
            "finalization_integrity",
            "finalization_digest_valid",
            _str(readiness.get("id")),
            {"expected_finalization_digest": _expected_finalization_digest(readiness)},
            {"actual_finalization_digest": _str(readiness.get("finalization_digest"))},
        ),
        (
            "seal_audit_source",
            "seal_audit_digest_consistent",
            _str(readiness.get("seal_audit_preview_id")),
            {"seal_audit_digest": _str(readiness.get("seal_audit_digest"))},
            {"source_seal_audit_digest": _str(seal_audit.get("audit_digest"))},
        ),
        (
            "governance_seal_source",
            "governance_seal_digest_consistent",
            _str(readiness.get("seal_id")),
            {"seal_digest": _str(readiness.get("seal_digest"))},
            {"source_seal_digest": _str(seal_audit.get("seal_digest"))},
        ),
        (
            "ledger_audit_source",
            "ledger_audit_digest_consistent",
            _str(readiness.get("ledger_audit_preview_id")),
            {"ledger_audit_digest": _str(readiness.get("ledger_audit_digest"))},
            {"source_ledger_audit_digest": _str(seal_audit.get("ledger_audit_digest"))},
        ),
        (
            "manual_handoff",
            "human_finalization_required",
            _str(readiness.get("id")),
            {"would_require_human_finalization": True},
            {
                "would_require_human_finalization": readiness.get(
                    "would_require_human_finalization"
                )
                is True
            },
        ),
        (
            "read_only",
            "finalization_readiness_is_preview_only",
            _str(readiness.get("id")),
            {"would_keep_read_only_until_manual_commit": True},
            {
                "would_keep_read_only_until_manual_commit": readiness.get(
                    "would_keep_read_only_until_manual_commit"
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
            readiness=readiness,
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
    readiness: Mapping[str, Any],
    reference_id: str,
    expected_signal: Mapping[str, Any],
    observed_signal: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditStep:
    step_seed = {
        "sequence": sequence,
        "category": category,
        "assertion": assertion,
        "finalization_readiness_id": _str(readiness.get("id")),
        "reference_id": _str(reference_id),
    }
    step_id = f"recovery-closure-finalization-readiness-audit-step-{_digest(step_seed)[:16]}"
    return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditStep(
        id=step_id,
        sequence=sequence,
        category=category,
        assertion=assertion,
        finalization_readiness_id=_str(readiness.get("id")),
        reference_id=_str(reference_id),
        expected_signal=dict(expected_signal),
        observed_signal=dict(observed_signal),
        status=FINALIZATION_READINESS_AUDIT_STEP_STATUS_PASS,
        required_action_on_fail=(
            "regenerate_recovery_closure_finalization_readiness_preview"
        ),
    )


def _expected_finalization_digest(readiness: Mapping[str, Any]) -> str:
    seed = {
        "finalization_type": RECOVERY_CLOSURE_FINALIZATION_TYPE,
        "finalization_decision": RECOVERY_CLOSURE_FINALIZATION_DECISION,
        "finalization_scope": RECOVERY_CLOSURE_FINALIZATION_SCOPE,
        "seal_audit_preview_id": _str(readiness.get("seal_audit_preview_id")),
        "seal_id": _str(readiness.get("seal_id")),
        "ledger_audit_preview_id": _str(readiness.get("ledger_audit_preview_id")),
        "ledger_entry_id": _str(readiness.get("ledger_entry_id")),
        "closure_id": _str(readiness.get("closure_id")),
        "audit_preview_id": _str(readiness.get("audit_preview_id")),
        "receipt_id": _str(readiness.get("receipt_id")),
        "executor_preview_id": _str(readiness.get("executor_preview_id")),
        "decision_id": _str(readiness.get("decision_id")),
        "execution_preview_id": _str(readiness.get("execution_preview_id")),
        "token_id": _str(readiness.get("token_id")),
        "lock_id": _str(readiness.get("lock_id")),
        "route": _str(readiness.get("route")),
        "operation_ids": tuple(_list_of_str(readiness.get("operation_ids"))),
        "recovery_step_ids": tuple(_list_of_str(readiness.get("recovery_step_ids"))),
        "future_mutation_step_ids": tuple(
            _list_of_str(readiness.get("future_mutation_step_ids"))
        ),
        "completion_audit_step_ids": tuple(
            _list_of_str(readiness.get("completion_audit_step_ids"))
        ),
        "milestone_ids": tuple(_list_of_str(readiness.get("milestone_ids"))),
        "ledger_audit_step_ids": tuple(
            _list_of_str(readiness.get("ledger_audit_step_ids"))
        ),
        "seal_audit_step_ids": tuple(
            _list_of_str(readiness.get("seal_audit_step_ids"))
        ),
        "seal_digest": _str(readiness.get("seal_digest")),
        "seal_audit_digest": _str(readiness.get("seal_audit_digest")),
        "ledger_audit_digest": _str(readiness.get("ledger_audit_digest")),
        "ledger_entry_digest": _str(readiness.get("ledger_entry_digest")),
        "closure_digest": _str(readiness.get("closure_digest")),
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
    recovery_closure_finalization_readiness: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditPreviewReport:
    return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_FINALIZATION_READINESS_AUDIT_STATUS_NO_ACTION_NEEDED,
        checks=(),
        previews=(),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_closure_finalization_readiness_report=dict(
            recovery_closure_finalization_readiness
        ),
    )


def _blocked_report(
    *,
    recovery_closure_finalization_readiness: Mapping[str, Any],
    checks: tuple[
        MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditCheck, ...
    ],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditPreviewReport:
    return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_FINALIZATION_READINESS_AUDIT_STATUS_BLOCKED,
        checks=checks,
        previews=(),
        blocking_reasons=tuple(
            _dedupe_strings(
                check.reason for check in checks if check.status == CHECK_STATUS_FAIL
            )
        ),
        required_actions=tuple(
            _dedupe_strings(
                action
                for check in checks
                if check.status == CHECK_STATUS_FAIL
                for action in check.required_actions
            )
        ),
        source_recovery_closure_finalization_readiness_report=dict(
            recovery_closure_finalization_readiness
        ),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditCheck:
    return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditCheck(
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
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditCheck:
    return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessAuditCheck(
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
