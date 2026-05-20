"""Read-only recovery closure governance seal preview for memory evidence repair.

The governance seal preview turns a ready recovery closure ledger audit into a
final governance seal draft. It never writes durable memory, records a real
seal, or freezes a real recovery closure.
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
from agent.memory_evidence_repair_recovery_closure_ledger_audit_preview import (
    LEDGER_AUDIT_STEP_STATUS_PASS,
    RECOVERY_CLOSURE_LEDGER_AUDIT_PREVIEW_STATUS_READY,
    RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_NO_ACTION_NEEDED,
    RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_READY,
)
from agent.memory_evidence_repair_recovery_closure_ledger_preview import (
    RECOVERY_CLOSURE_LEDGER_ENTRY_STATUS_READY,
)


RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_READY = (
    "recovery_closure_governance_seal_preview_ready"
)
RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_BLOCKED = "blocked"
RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_NO_ACTION_NEEDED = "no_action_needed"
RECOVERY_CLOSURE_GOVERNANCE_SEAL_DRAFT_STATUS_READY = (
    "recovery_closure_governance_seal_draft_ready"
)
RECOVERY_CLOSURE_GOVERNANCE_SEAL_TYPE = (
    "memory_evidence_repair_recovery_closure_governance_seal"
)
RECOVERY_CLOSURE_GOVERNANCE_DECISION = "seal_recovery_closure"
RECOVERY_CLOSURE_GOVERNANCE_SCOPE = "manual_memory_evidence_repair_recovery_closure"

CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_READY = "recovery_closure_ledger_audit_ready"
CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_INTEGRITY = (
    "recovery_closure_ledger_audit_integrity"
)
CHECK_RECOVERY_GOVERNANCE_SEAL_SOURCE_CHAIN = (
    "recovery_governance_seal_source_chain"
)
CHECK_RECOVERY_GOVERNANCE_SEAL_REQUIREMENTS = (
    "recovery_governance_seal_requirements"
)


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureGovernanceSealCheck:
    """One read-only recovery closure governance seal check."""

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
class MemoryEvidenceRepairRecoveryClosureGovernanceSealDraft:
    """One read-only future recovery closure governance seal draft."""

    id: str
    seal_type: str
    status: str
    governance_decision: str
    governance_scope: str
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
    ledger_entry_digest: str
    closure_digest: str
    ledger_audit_digest: str
    seal_digest: str
    seal_preview: str
    would_freeze_recovery_closure: bool
    would_preserve_audit_evidence: bool
    would_enable_governed_handoff: bool
    would_block_unreviewed_mutation: bool
    safety_note: str
    source_recovery_closure_ledger_audit_preview: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seal_type": self.seal_type,
            "status": self.status,
            "governance_decision": self.governance_decision,
            "governance_scope": self.governance_scope,
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
            "ledger_entry_digest": self.ledger_entry_digest,
            "closure_digest": self.closure_digest,
            "ledger_audit_digest": self.ledger_audit_digest,
            "seal_digest": self.seal_digest,
            "seal_preview": self.seal_preview,
            "would_freeze_recovery_closure": self.would_freeze_recovery_closure,
            "would_preserve_audit_evidence": self.would_preserve_audit_evidence,
            "would_enable_governed_handoff": self.would_enable_governed_handoff,
            "would_block_unreviewed_mutation": self.would_block_unreviewed_mutation,
            "safety_note": self.safety_note,
            "source_recovery_closure_ledger_audit_preview": dict(
                self.source_recovery_closure_ledger_audit_preview
            ),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureGovernanceSealPreviewReport:
    """Read-only recovery closure governance seal preview report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairRecoveryClosureGovernanceSealCheck, ...]
    seals: tuple[MemoryEvidenceRepairRecoveryClosureGovernanceSealDraft, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_recovery_closure_ledger_audit_report: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        by_check_status: dict[str, int] = {}
        for check in self.checks:
            by_check_status[check.status] = by_check_status.get(check.status, 0) + 1
        return {
            "status": self.status,
            "check_count": len(self.checks),
            "pass_count": by_check_status.get(CHECK_STATUS_PASS, 0),
            "fail_count": by_check_status.get(CHECK_STATUS_FAIL, 0),
            "seal_count": len(self.seals),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "recovery_closure_governance_seal_available": bool(self.seals),
            "recovery_closure_governance_seal_ready": (
                self.status == RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_READY
            ),
            "would_freeze_recovery_closure_count": sum(
                1 for seal in self.seals if seal.would_freeze_recovery_closure
            ),
            "would_enable_governed_handoff_count": sum(
                1 for seal in self.seals if seal.would_enable_governed_handoff
            ),
            "has_blocks": self.status == RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_BLOCKED,
            "requires_followup": bool(self.required_actions),
            "by_check_status": by_check_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
            "seals": [seal.to_dict() for seal in self.seals],
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "source_recovery_closure_ledger_audit_report": dict(
                self.source_recovery_closure_ledger_audit_report
            ),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_closure_governance_seal_preview(
    *,
    recovery_closure_ledger_audit: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealPreviewReport:
    """Build a read-only governance seal from a ready closure ledger audit."""

    recovery_closure_ledger_audit = (
        recovery_closure_ledger_audit
        if isinstance(recovery_closure_ledger_audit, Mapping)
        else {}
    )
    report_status = _str(recovery_closure_ledger_audit.get("status"))
    preview = _extract_preview(recovery_closure_ledger_audit)

    if not preview:
        if (
            not recovery_closure_ledger_audit
            or report_status == RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_NO_ACTION_NEEDED
        ):
            return _empty_report(
                recovery_closure_ledger_audit=recovery_closure_ledger_audit
            )
        source_blocking = tuple(
            _list_of_str(recovery_closure_ledger_audit.get("blocking_reasons"))
        )
        source_actions = tuple(
            _list_of_str(recovery_closure_ledger_audit.get("required_actions"))
        )
        return _blocked_report(
            recovery_closure_ledger_audit=recovery_closure_ledger_audit,
            checks=(
                _fail(
                    CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_READY,
                    "No recovery closure ledger audit preview was supplied."
                    if not source_blocking
                    else "; ".join(source_blocking),
                    source_actions
                    or ("produce_recovery_closure_ledger_audit_preview",),
                    {"recovery_closure_ledger_audit_status": report_status},
                ),
            ),
        )

    checks = (
        _ledger_audit_ready_check(recovery_closure_ledger_audit, preview),
        _ledger_audit_integrity_check(preview),
        _seal_source_chain_check(preview),
        _seal_requirements_check(preview),
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
        return MemoryEvidenceRepairRecoveryClosureGovernanceSealPreviewReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_BLOCKED,
            checks=checks,
            seals=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            source_recovery_closure_ledger_audit_report=dict(
                recovery_closure_ledger_audit
            ),
        )

    seal = _seal_from_ledger_audit(preview)
    return MemoryEvidenceRepairRecoveryClosureGovernanceSealPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_READY,
        checks=checks,
        seals=(seal,),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_closure_ledger_audit_report=dict(recovery_closure_ledger_audit),
    )


def empty_evidence_repair_recovery_closure_governance_seal_preview() -> MemoryEvidenceRepairRecoveryClosureGovernanceSealPreviewReport:
    """Return an empty read-only recovery closure governance seal preview."""

    return _empty_report(recovery_closure_ledger_audit={})


def _extract_preview(recovery_closure_ledger_audit: Mapping[str, Any]) -> dict[str, Any]:
    previews = recovery_closure_ledger_audit.get("previews")
    if isinstance(previews, list):
        for preview in previews:
            if isinstance(preview, Mapping):
                return dict(preview)
    if _str(recovery_closure_ledger_audit.get("id")).startswith(
        "recovery-closure-ledger-audit-"
    ):
        return dict(recovery_closure_ledger_audit)
    return {}


def _ledger_audit_ready_check(
    recovery_closure_ledger_audit: Mapping[str, Any],
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealCheck:
    report_status = _str(recovery_closure_ledger_audit.get("status"))
    preview_status = _str(preview.get("status"))
    if report_status and report_status != RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_READY,
            "Recovery closure ledger audit report is not ready for governance sealing.",
            tuple(_list_of_str(recovery_closure_ledger_audit.get("required_actions")))
            or ("produce_ready_recovery_closure_ledger_audit_preview",),
            {"report_status": report_status, "preview_status": preview_status},
        )
    if preview_status != RECOVERY_CLOSURE_LEDGER_AUDIT_PREVIEW_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_READY,
            "Recovery closure ledger audit preview is not ready for governance sealing.",
            ("regenerate_recovery_closure_ledger_audit_preview",),
            {"report_status": report_status, "preview_status": preview_status},
        )
    return _pass(
        CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_READY,
        "Recovery closure ledger audit preview is ready for governance sealing.",
        {"ledger_audit_preview_id": _str(preview.get("id"))},
    )


def _ledger_audit_integrity_check(
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealCheck:
    expected_digest = _expected_ledger_audit_digest(preview)
    actual_digest = _str(preview.get("audit_digest"))
    expected_id = (
        f"recovery-closure-ledger-audit-{expected_digest[:16]}"
        if expected_digest
        else ""
    )
    missing_fields: list[str] = []
    for field_name in (
        "id",
        "status",
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
        "audit_digest",
        "audit_steps",
        "source_recovery_closure_ledger_entry",
    ):
        value = preview.get(field_name)
        if isinstance(value, list) and not value:
            missing_fields.append(field_name)
        elif isinstance(value, Mapping) and not value:
            missing_fields.append(field_name)
        elif not isinstance(value, (list, Mapping)) and not _str(value):
            missing_fields.append(field_name)
    if (
        missing_fields
        or actual_digest != expected_digest
        or _str(preview.get("id")) != expected_id
    ):
        return _fail(
            CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_INTEGRITY,
            "Recovery closure ledger audit digest, id, or required fields are invalid.",
            ("regenerate_recovery_closure_ledger_audit_preview",),
            {
                "expected_audit_digest": expected_digest,
                "actual_audit_digest": actual_digest,
                "expected_audit_id": expected_id,
                "actual_audit_id": _str(preview.get("id")),
                "missing_fields": missing_fields,
            },
        )
    return _pass(
        CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_INTEGRITY,
        "Recovery closure ledger audit integrity checks pass.",
        {"ledger_audit_preview_id": expected_id, "audit_digest": actual_digest},
    )


def _seal_source_chain_check(
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealCheck:
    entry = _dict(preview.get("source_recovery_closure_ledger_entry"))
    mismatches: list[dict[str, str]] = []
    for field_name in (
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
        "closure_digest",
    ):
        entry_key = "id" if field_name == "ledger_entry_id" else field_name
        _compare(mismatches, field_name, preview, field_name, entry, entry_key)
    if _str(entry.get("entry_digest")) != _str(preview.get("ledger_entry_digest")):
        mismatches.append(
            {
                "field": "ledger_entry_digest",
                "left": _str(preview.get("ledger_entry_digest")),
                "right": _str(entry.get("entry_digest")),
                "right_key": "source_recovery_closure_ledger_entry.entry_digest",
            }
        )
    audit_steps = _audit_steps(preview)
    non_pass_steps = [
        _str(step.get("id")) or f"ledger-audit-step-{index}"
        for index, step in enumerate(audit_steps, start=1)
        if _str(step.get("status")) != LEDGER_AUDIT_STEP_STATUS_PASS
    ]
    if (
        not entry
        or mismatches
        or _str(entry.get("status")) != RECOVERY_CLOSURE_LEDGER_ENTRY_STATUS_READY
        or non_pass_steps
    ):
        return _fail(
            CHECK_RECOVERY_GOVERNANCE_SEAL_SOURCE_CHAIN,
            "Recovery closure governance seal source chain is incomplete or inconsistent.",
            ("regenerate_recovery_closure_ledger_audit_preview",),
            {
                "missing_sources": [
                    "source_recovery_closure_ledger_entry"
                ]
                if not entry
                else [],
                "mismatches": mismatches,
                "entry_status": _str(entry.get("status")),
                "non_pass_ledger_audit_step_ids": non_pass_steps,
            },
        )
    return _pass(
        CHECK_RECOVERY_GOVERNANCE_SEAL_SOURCE_CHAIN,
        "Recovery closure governance seal source chain is internally consistent.",
        {
            "ledger_entry_id": _str(preview.get("ledger_entry_id")),
            "closure_id": _str(preview.get("closure_id")),
            "ledger_audit_step_count": len(audit_steps),
        },
    )


def _seal_requirements_check(
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealCheck:
    required_flags = (
        "would_verify_ledger_entry_integrity",
        "would_verify_closure_source_integrity",
        "would_verify_lifecycle_milestones",
        "would_verify_completion_audit_evidence",
    )
    missing_flags = [field_name for field_name in required_flags if preview.get(field_name) is not True]
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
        if not _list_of_str(preview.get(field_name))
    ]
    categories = {_str(step.get("category")) for step in _audit_steps(preview)}
    required_categories = {
        "ledger_entry",
        "closure_source",
        "milestones",
        "source_chain",
        "completion_audit",
        "read_only",
    }
    missing_categories = sorted(required_categories - categories)
    if missing_flags or missing_lists or missing_categories:
        return _fail(
            CHECK_RECOVERY_GOVERNANCE_SEAL_REQUIREMENTS,
            "Recovery closure governance seal requirements are incomplete.",
            ("regenerate_recovery_closure_ledger_audit_preview",),
            {
                "missing_flags": missing_flags,
                "missing_lists": missing_lists,
                "missing_categories": missing_categories,
            },
        )
    return _pass(
        CHECK_RECOVERY_GOVERNANCE_SEAL_REQUIREMENTS,
        "Recovery closure governance seal requirements cover integrity, lifecycle, audit, and read-only guarantees.",
        {
            "ledger_audit_step_count": len(_audit_steps(preview)),
            "milestone_count": len(_list_of_str(preview.get("milestone_ids"))),
        },
    )


def _seal_from_ledger_audit(
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealDraft:
    operation_ids = tuple(_list_of_str(preview.get("operation_ids")))
    recovery_step_ids = tuple(_list_of_str(preview.get("recovery_step_ids")))
    future_mutation_step_ids = tuple(_list_of_str(preview.get("future_mutation_step_ids")))
    completion_audit_step_ids = tuple(_list_of_str(preview.get("completion_audit_step_ids")))
    milestone_ids = tuple(_list_of_str(preview.get("milestone_ids")))
    ledger_audit_step_ids = tuple(_list_of_str(preview.get("ledger_audit_step_ids")))
    seal_seed = {
        "seal_type": RECOVERY_CLOSURE_GOVERNANCE_SEAL_TYPE,
        "governance_decision": RECOVERY_CLOSURE_GOVERNANCE_DECISION,
        "governance_scope": RECOVERY_CLOSURE_GOVERNANCE_SCOPE,
        "ledger_audit_preview_id": _str(preview.get("id")),
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
        "operation_ids": operation_ids,
        "recovery_step_ids": recovery_step_ids,
        "future_mutation_step_ids": future_mutation_step_ids,
        "completion_audit_step_ids": completion_audit_step_ids,
        "milestone_ids": milestone_ids,
        "ledger_audit_step_ids": ledger_audit_step_ids,
        "ledger_entry_digest": _str(preview.get("ledger_entry_digest")),
        "closure_digest": _str(preview.get("closure_digest")),
        "ledger_audit_digest": _str(preview.get("audit_digest")),
    }
    seal_digest = _digest(seal_seed)
    seal_id = f"recovery-closure-governance-seal-{seal_digest[:16]}"
    return MemoryEvidenceRepairRecoveryClosureGovernanceSealDraft(
        id=seal_id,
        seal_type=RECOVERY_CLOSURE_GOVERNANCE_SEAL_TYPE,
        status=RECOVERY_CLOSURE_GOVERNANCE_SEAL_DRAFT_STATUS_READY,
        governance_decision=RECOVERY_CLOSURE_GOVERNANCE_DECISION,
        governance_scope=RECOVERY_CLOSURE_GOVERNANCE_SCOPE,
        ledger_audit_preview_id=_str(preview.get("id")),
        ledger_entry_id=_str(preview.get("ledger_entry_id")),
        closure_id=_str(preview.get("closure_id")),
        audit_preview_id=_str(preview.get("audit_preview_id")),
        receipt_id=_str(preview.get("receipt_id")),
        executor_preview_id=_str(preview.get("executor_preview_id")),
        decision_id=_str(preview.get("decision_id")),
        execution_preview_id=_str(preview.get("execution_preview_id")),
        token_id=_str(preview.get("token_id")),
        lock_id=_str(preview.get("lock_id")),
        route=_str(preview.get("route")),
        operation_ids=operation_ids,
        recovery_step_ids=recovery_step_ids,
        future_mutation_step_ids=future_mutation_step_ids,
        completion_audit_step_ids=completion_audit_step_ids,
        milestone_ids=milestone_ids,
        ledger_audit_step_ids=ledger_audit_step_ids,
        ledger_entry_digest=_str(preview.get("ledger_entry_digest")),
        closure_digest=_str(preview.get("closure_digest")),
        ledger_audit_digest=_str(preview.get("audit_digest")),
        seal_digest=seal_digest,
        seal_preview=f"{seal_id}:{seal_digest[:12]}",
        would_freeze_recovery_closure=True,
        would_preserve_audit_evidence=True,
        would_enable_governed_handoff=True,
        would_block_unreviewed_mutation=True,
        safety_note=(
            "Read-only recovery closure governance seal draft. It previews a "
            "future governance seal for handoff and accountability, but does "
            "not freeze, persist, or mutate durable memory."
        ),
        source_recovery_closure_ledger_audit_preview=dict(preview),
    )


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
    recovery_closure_ledger_audit: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealPreviewReport:
    return MemoryEvidenceRepairRecoveryClosureGovernanceSealPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_NO_ACTION_NEEDED,
        checks=(),
        seals=(),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_closure_ledger_audit_report=dict(
            recovery_closure_ledger_audit
        ),
    )


def _blocked_report(
    *,
    recovery_closure_ledger_audit: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairRecoveryClosureGovernanceSealCheck, ...],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealPreviewReport:
    return MemoryEvidenceRepairRecoveryClosureGovernanceSealPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_BLOCKED,
        checks=checks,
        seals=(),
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
        source_recovery_closure_ledger_audit_report=dict(
            recovery_closure_ledger_audit
        ),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealCheck:
    return MemoryEvidenceRepairRecoveryClosureGovernanceSealCheck(
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
) -> MemoryEvidenceRepairRecoveryClosureGovernanceSealCheck:
    return MemoryEvidenceRepairRecoveryClosureGovernanceSealCheck(
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
