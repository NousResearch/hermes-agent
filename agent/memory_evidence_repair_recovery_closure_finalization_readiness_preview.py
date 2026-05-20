"""Read-only recovery closure finalization readiness preview.

This layer aggregates the closure, ledger, ledger audit, governance seal, and
seal audit into a final readiness gate for a future manual closure. It never
finalizes a recovery, writes durable memory, or freezes real state.
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
from agent.memory_evidence_repair_recovery_closure_governance_seal_audit_preview import (
    GOVERNANCE_SEAL_AUDIT_STEP_STATUS_PASS,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_PREVIEW_STATUS_READY,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_NO_ACTION_NEEDED,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_READY,
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


RECOVERY_CLOSURE_FINALIZATION_STATUS_READY = (
    "recovery_closure_finalization_readiness_preview_ready"
)
RECOVERY_CLOSURE_FINALIZATION_STATUS_BLOCKED = "blocked"
RECOVERY_CLOSURE_FINALIZATION_STATUS_NO_ACTION_NEEDED = "no_action_needed"
RECOVERY_CLOSURE_FINALIZATION_DRAFT_STATUS_READY = (
    "recovery_closure_finalization_readiness_ready"
)
RECOVERY_CLOSURE_FINALIZATION_TYPE = (
    "memory_evidence_repair_recovery_closure_finalization_readiness"
)
RECOVERY_CLOSURE_FINALIZATION_DECISION = "ready_for_final_closure"
RECOVERY_CLOSURE_FINALIZATION_SCOPE = (
    "manual_memory_evidence_repair_recovery_finalization"
)

CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_READY = (
    "recovery_governance_seal_audit_ready"
)
CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_INTEGRITY = (
    "recovery_governance_seal_audit_integrity"
)
CHECK_RECOVERY_FINALIZATION_SOURCE_CHAIN = "recovery_finalization_source_chain"
CHECK_RECOVERY_FINALIZATION_REQUIREMENTS = "recovery_finalization_requirements"


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureFinalizationReadinessCheck:
    """One read-only finalization readiness check."""

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
class MemoryEvidenceRepairRecoveryClosureFinalizationReadinessPreview:
    """One read-only future finalization readiness draft."""

    id: str
    finalization_type: str
    status: str
    finalization_decision: str
    finalization_scope: str
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
    seal_digest: str
    seal_audit_digest: str
    ledger_audit_digest: str
    ledger_entry_digest: str
    closure_digest: str
    finalization_digest: str
    finalization_preview: str
    would_allow_final_closure: bool
    would_preserve_full_governance_chain: bool
    would_require_human_finalization: bool
    would_keep_read_only_until_manual_commit: bool
    safety_note: str
    source_recovery_closure_governance_seal_audit_preview: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "finalization_type": self.finalization_type,
            "status": self.status,
            "finalization_decision": self.finalization_decision,
            "finalization_scope": self.finalization_scope,
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
            "seal_digest": self.seal_digest,
            "seal_audit_digest": self.seal_audit_digest,
            "ledger_audit_digest": self.ledger_audit_digest,
            "ledger_entry_digest": self.ledger_entry_digest,
            "closure_digest": self.closure_digest,
            "finalization_digest": self.finalization_digest,
            "finalization_preview": self.finalization_preview,
            "would_allow_final_closure": self.would_allow_final_closure,
            "would_preserve_full_governance_chain": (
                self.would_preserve_full_governance_chain
            ),
            "would_require_human_finalization": self.would_require_human_finalization,
            "would_keep_read_only_until_manual_commit": (
                self.would_keep_read_only_until_manual_commit
            ),
            "safety_note": self.safety_note,
            "source_recovery_closure_governance_seal_audit_preview": dict(
                self.source_recovery_closure_governance_seal_audit_preview
            ),
        }


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryClosureFinalizationReadinessPreviewReport:
    """Read-only finalization readiness preview report."""

    generated_at: str
    status: str
    checks: tuple[MemoryEvidenceRepairRecoveryClosureFinalizationReadinessCheck, ...]
    readiness: tuple[
        MemoryEvidenceRepairRecoveryClosureFinalizationReadinessPreview, ...
    ]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    source_recovery_closure_governance_seal_audit_report: dict[str, Any]

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
            "readiness_count": len(self.readiness),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "recovery_closure_finalization_readiness_available": bool(
                self.readiness
            ),
            "recovery_closure_finalization_ready": (
                self.status == RECOVERY_CLOSURE_FINALIZATION_STATUS_READY
            ),
            "would_allow_final_closure_count": sum(
                1 for item in self.readiness if item.would_allow_final_closure
            ),
            "would_preserve_full_governance_chain_count": sum(
                1
                for item in self.readiness
                if item.would_preserve_full_governance_chain
            ),
            "would_require_human_finalization_count": sum(
                1 for item in self.readiness if item.would_require_human_finalization
            ),
            "would_keep_read_only_until_manual_commit_count": sum(
                1
                for item in self.readiness
                if item.would_keep_read_only_until_manual_commit
            ),
            "has_blocks": (
                self.status == RECOVERY_CLOSURE_FINALIZATION_STATUS_BLOCKED
            ),
            "requires_followup": bool(self.required_actions),
            "by_check_status": by_check_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
            "readiness": [item.to_dict() for item in self.readiness],
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "source_recovery_closure_governance_seal_audit_report": dict(
                self.source_recovery_closure_governance_seal_audit_report
            ),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_closure_finalization_readiness_preview(
    *,
    recovery_closure_governance_seal_audit: Mapping[str, Any] | None = None,
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessPreviewReport:
    """Build a read-only finalization readiness preview from a seal audit."""

    recovery_closure_governance_seal_audit = (
        recovery_closure_governance_seal_audit
        if isinstance(recovery_closure_governance_seal_audit, Mapping)
        else {}
    )
    report_status = _str(recovery_closure_governance_seal_audit.get("status"))
    preview = _extract_preview(recovery_closure_governance_seal_audit)

    if not preview:
        if (
            not recovery_closure_governance_seal_audit
            or report_status
            == RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_NO_ACTION_NEEDED
        ):
            return _empty_report(
                recovery_closure_governance_seal_audit=(
                    recovery_closure_governance_seal_audit
                )
            )
        source_blocking = tuple(
            _list_of_str(
                recovery_closure_governance_seal_audit.get("blocking_reasons")
            )
        )
        source_actions = tuple(
            _list_of_str(recovery_closure_governance_seal_audit.get("required_actions"))
        )
        return _blocked_report(
            recovery_closure_governance_seal_audit=(
                recovery_closure_governance_seal_audit
            ),
            checks=(
                _fail(
                    CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_READY,
                    "No recovery closure governance seal audit preview was supplied."
                    if not source_blocking
                    else "; ".join(source_blocking),
                    source_actions
                    or (
                        "produce_recovery_closure_governance_seal_audit_preview",
                    ),
                    {
                        "recovery_closure_governance_seal_audit_status": (
                            report_status
                        )
                    },
                ),
            ),
        )

    checks = (
        _seal_audit_ready_check(recovery_closure_governance_seal_audit, preview),
        _seal_audit_integrity_check(preview),
        _source_chain_check(preview),
        _finalization_requirements_check(preview),
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
        return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessPreviewReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status=RECOVERY_CLOSURE_FINALIZATION_STATUS_BLOCKED,
            checks=checks,
            readiness=(),
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            source_recovery_closure_governance_seal_audit_report=dict(
                recovery_closure_governance_seal_audit
            ),
        )

    readiness = _readiness_from_seal_audit(preview)
    return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_FINALIZATION_STATUS_READY,
        checks=checks,
        readiness=(readiness,),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_closure_governance_seal_audit_report=dict(
            recovery_closure_governance_seal_audit
        ),
    )


def empty_evidence_repair_recovery_closure_finalization_readiness_preview() -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessPreviewReport:
    """Return an empty read-only finalization readiness preview."""

    return _empty_report(recovery_closure_governance_seal_audit={})


def _extract_preview(
    recovery_closure_governance_seal_audit: Mapping[str, Any],
) -> dict[str, Any]:
    previews = recovery_closure_governance_seal_audit.get("previews")
    if isinstance(previews, list):
        for preview in previews:
            if isinstance(preview, Mapping):
                return dict(preview)
    if _str(recovery_closure_governance_seal_audit.get("id")).startswith(
        "recovery-closure-governance-seal-audit-"
    ):
        return dict(recovery_closure_governance_seal_audit)
    return {}


def _seal_audit_ready_check(
    recovery_closure_governance_seal_audit: Mapping[str, Any],
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessCheck:
    report_status = _str(recovery_closure_governance_seal_audit.get("status"))
    preview_status = _str(preview.get("status"))
    if (
        report_status
        and report_status != RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_READY
    ):
        return _fail(
            CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_READY,
            "Recovery closure governance seal audit report is not ready for finalization readiness preview.",
            tuple(
                _list_of_str(
                    recovery_closure_governance_seal_audit.get("required_actions")
                )
            )
            or ("produce_ready_recovery_closure_governance_seal_audit_preview",),
            {"report_status": report_status, "preview_status": preview_status},
        )
    if preview_status != RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_PREVIEW_STATUS_READY:
        return _fail(
            CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_READY,
            "Recovery closure governance seal audit preview is not ready for finalization readiness preview.",
            ("regenerate_recovery_closure_governance_seal_audit_preview",),
            {"report_status": report_status, "preview_status": preview_status},
        )
    return _pass(
        CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_READY,
        "Recovery closure governance seal audit preview is ready for finalization readiness preview.",
        {"seal_audit_preview_id": _str(preview.get("id"))},
    )


def _seal_audit_integrity_check(
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessCheck:
    expected_digest = _expected_seal_audit_digest(preview)
    actual_digest = _str(preview.get("audit_digest"))
    expected_id = (
        f"recovery-closure-governance-seal-audit-{expected_digest[:16]}"
        if expected_digest
        else ""
    )
    missing_fields: list[str] = []
    for field_name in (
        "id",
        "status",
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
        "ledger_audit_digest",
        "ledger_entry_digest",
        "closure_digest",
        "audit_digest",
        "audit_steps",
        "source_recovery_closure_governance_seal",
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
            CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_INTEGRITY,
            "Recovery closure governance seal audit digest, id, or required fields are invalid.",
            ("regenerate_recovery_closure_governance_seal_audit_preview",),
            {
                "expected_seal_audit_digest": expected_digest,
                "actual_seal_audit_digest": actual_digest,
                "expected_seal_audit_id": expected_id,
                "actual_seal_audit_id": _str(preview.get("id")),
                "missing_fields": missing_fields,
            },
        )
    return _pass(
        CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_INTEGRITY,
        "Recovery closure governance seal audit integrity checks pass.",
        {"seal_audit_preview_id": expected_id, "audit_digest": actual_digest},
    )


def _source_chain_check(
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessCheck:
    seal = _dict(preview.get("source_recovery_closure_governance_seal"))
    ledger_audit = _dict(seal.get("source_recovery_closure_ledger_audit_preview"))
    entry = _dict(ledger_audit.get("source_recovery_closure_ledger_entry"))
    missing_sources = [
        name
        for name, source in (
            ("source_recovery_closure_governance_seal", seal),
            ("source_recovery_closure_ledger_audit_preview", ledger_audit),
            ("source_recovery_closure_ledger_entry", entry),
        )
        if not source
    ]
    mismatches: list[dict[str, str]] = []
    for field_name in (
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
        "ledger_audit_digest",
        "ledger_entry_digest",
        "closure_digest",
    ):
        seal_key = "id" if field_name == "seal_id" else field_name
        _compare(mismatches, field_name, preview, field_name, seal, seal_key)

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
        ledger_key = "id" if field_name == "ledger_audit_preview_id" else field_name
        _compare(mismatches, field_name, preview, field_name, ledger_audit, ledger_key)
    if _str(preview.get("ledger_audit_digest")) != _str(
        ledger_audit.get("audit_digest")
    ):
        mismatches.append(
            {
                "field": "ledger_audit_digest",
                "left": _str(preview.get("ledger_audit_digest")),
                "right": _str(ledger_audit.get("audit_digest")),
                "right_key": "source_recovery_closure_ledger_audit_preview.audit_digest",
            }
        )
    _compare(mismatches, "ledger_entry_id", preview, "ledger_entry_id", entry, "id")
    _compare(mismatches, "closure_id", preview, "closure_id", entry, "closure_id")
    if _str(preview.get("ledger_entry_digest")) != _str(entry.get("entry_digest")):
        mismatches.append(
            {
                "field": "ledger_entry_digest",
                "left": _str(preview.get("ledger_entry_digest")),
                "right": _str(entry.get("entry_digest")),
                "right_key": "source_recovery_closure_ledger_entry.entry_digest",
            }
        )
    seal_audit_steps = _audit_steps(preview)
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
        or _str(seal.get("status")) != RECOVERY_CLOSURE_GOVERNANCE_SEAL_DRAFT_STATUS_READY
        or _str(ledger_audit.get("status"))
        != RECOVERY_CLOSURE_LEDGER_AUDIT_PREVIEW_STATUS_READY
        or _str(entry.get("status")) != RECOVERY_CLOSURE_LEDGER_ENTRY_STATUS_READY
        or non_pass_seal_audit_steps
        or non_pass_ledger_audit_steps
    ):
        return _fail(
            CHECK_RECOVERY_FINALIZATION_SOURCE_CHAIN,
            "Recovery closure finalization source chain is incomplete or inconsistent.",
            ("regenerate_recovery_closure_governance_seal_audit_preview",),
            {
                "missing_sources": missing_sources,
                "mismatches": mismatches,
                "seal_status": _str(seal.get("status")),
                "ledger_audit_status": _str(ledger_audit.get("status")),
                "entry_status": _str(entry.get("status")),
                "non_pass_seal_audit_step_ids": non_pass_seal_audit_steps,
                "non_pass_ledger_audit_step_ids": non_pass_ledger_audit_steps,
            },
        )
    return _pass(
        CHECK_RECOVERY_FINALIZATION_SOURCE_CHAIN,
        "Recovery closure finalization source chain preserves seal, ledger audit, ledger entry, and closure evidence.",
        {
            "seal_audit_preview_id": _str(preview.get("id")),
            "seal_id": _str(preview.get("seal_id")),
            "ledger_audit_preview_id": _str(preview.get("ledger_audit_preview_id")),
            "ledger_entry_id": _str(preview.get("ledger_entry_id")),
            "seal_audit_step_count": len(seal_audit_steps),
            "ledger_audit_step_count": len(ledger_audit_steps),
        },
    )


def _finalization_requirements_check(
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessCheck:
    required_preview_flags = (
        "would_verify_seal_integrity",
        "would_verify_governance_handoff",
        "would_verify_source_audit_evidence",
        "would_verify_read_only_constraints",
    )
    missing_preview_flags = [
        field_name
        for field_name in required_preview_flags
        if preview.get(field_name) is not True
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
        if not _list_of_str(preview.get(field_name))
    ]
    seal = _dict(preview.get("source_recovery_closure_governance_seal"))
    required_seal_flags = (
        "would_freeze_recovery_closure",
        "would_preserve_audit_evidence",
        "would_enable_governed_handoff",
        "would_block_unreviewed_mutation",
    )
    missing_seal_flags = [
        field_name for field_name in required_seal_flags if seal.get(field_name) is not True
    ]
    categories = {_str(step.get("category")) for step in _audit_steps(preview)}
    required_categories = {
        "seal_integrity",
        "ledger_audit_source",
        "ledger_entry_source",
        "closure_trace",
        "governance_controls",
        "read_only",
    }
    missing_categories = sorted(required_categories - categories)
    if missing_preview_flags or missing_lists or missing_seal_flags or missing_categories:
        return _fail(
            CHECK_RECOVERY_FINALIZATION_REQUIREMENTS,
            "Recovery closure finalization readiness requirements are incomplete.",
            ("regenerate_recovery_closure_governance_seal_audit_preview",),
            {
                "missing_preview_flags": missing_preview_flags,
                "missing_lists": missing_lists,
                "missing_seal_flags": missing_seal_flags,
                "missing_categories": missing_categories,
            },
        )
    return _pass(
        CHECK_RECOVERY_FINALIZATION_REQUIREMENTS,
        "Recovery closure finalization readiness covers integrity, handoff, source evidence, and read-only constraints.",
        {
            "seal_audit_step_count": len(_audit_steps(preview)),
            "milestone_count": len(_list_of_str(preview.get("milestone_ids"))),
        },
    )


def _readiness_from_seal_audit(
    preview: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessPreview:
    operation_ids = tuple(_list_of_str(preview.get("operation_ids")))
    recovery_step_ids = tuple(_list_of_str(preview.get("recovery_step_ids")))
    future_mutation_step_ids = tuple(_list_of_str(preview.get("future_mutation_step_ids")))
    completion_audit_step_ids = tuple(
        _list_of_str(preview.get("completion_audit_step_ids"))
    )
    milestone_ids = tuple(_list_of_str(preview.get("milestone_ids")))
    ledger_audit_step_ids = tuple(_list_of_str(preview.get("ledger_audit_step_ids")))
    seal_audit_step_ids = tuple(_list_of_str(preview.get("seal_audit_step_ids")))
    finalization_seed = {
        "finalization_type": RECOVERY_CLOSURE_FINALIZATION_TYPE,
        "finalization_decision": RECOVERY_CLOSURE_FINALIZATION_DECISION,
        "finalization_scope": RECOVERY_CLOSURE_FINALIZATION_SCOPE,
        "seal_audit_preview_id": _str(preview.get("id")),
        "seal_id": _str(preview.get("seal_id")),
        "ledger_audit_preview_id": _str(preview.get("ledger_audit_preview_id")),
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
        "seal_audit_step_ids": seal_audit_step_ids,
        "seal_digest": _str(preview.get("seal_digest")),
        "seal_audit_digest": _str(preview.get("audit_digest")),
        "ledger_audit_digest": _str(preview.get("ledger_audit_digest")),
        "ledger_entry_digest": _str(preview.get("ledger_entry_digest")),
        "closure_digest": _str(preview.get("closure_digest")),
    }
    finalization_digest = _digest(finalization_seed)
    finalization_id = (
        f"recovery-closure-finalization-readiness-{finalization_digest[:16]}"
    )
    return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessPreview(
        id=finalization_id,
        finalization_type=RECOVERY_CLOSURE_FINALIZATION_TYPE,
        status=RECOVERY_CLOSURE_FINALIZATION_DRAFT_STATUS_READY,
        finalization_decision=RECOVERY_CLOSURE_FINALIZATION_DECISION,
        finalization_scope=RECOVERY_CLOSURE_FINALIZATION_SCOPE,
        seal_audit_preview_id=_str(preview.get("id")),
        seal_id=_str(preview.get("seal_id")),
        ledger_audit_preview_id=_str(preview.get("ledger_audit_preview_id")),
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
        seal_audit_step_ids=seal_audit_step_ids,
        seal_digest=_str(preview.get("seal_digest")),
        seal_audit_digest=_str(preview.get("audit_digest")),
        ledger_audit_digest=_str(preview.get("ledger_audit_digest")),
        ledger_entry_digest=_str(preview.get("ledger_entry_digest")),
        closure_digest=_str(preview.get("closure_digest")),
        finalization_digest=finalization_digest,
        finalization_preview=f"{finalization_id}:{finalization_digest[:12]}",
        would_allow_final_closure=True,
        would_preserve_full_governance_chain=True,
        would_require_human_finalization=True,
        would_keep_read_only_until_manual_commit=True,
        safety_note=(
            "Read-only recovery closure finalization readiness preview. It "
            "allows a later human-governed final closure only if the complete "
            "seal and audit chain remains intact; it does not finalize or "
            "mutate durable memory."
        ),
        source_recovery_closure_governance_seal_audit_preview=dict(preview),
    )


def _expected_seal_audit_digest(preview: Mapping[str, Any]) -> str:
    seed = {
        "seal_id": _str(preview.get("seal_id")),
        "ledger_audit_preview_id": _str(preview.get("ledger_audit_preview_id")),
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
        "seal_audit_step_ids": tuple(_list_of_str(preview.get("seal_audit_step_ids"))),
        "seal_digest": _str(preview.get("seal_digest")),
        "ledger_audit_digest": _str(preview.get("ledger_audit_digest")),
        "ledger_entry_digest": _str(preview.get("ledger_entry_digest")),
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
    recovery_closure_governance_seal_audit: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessPreviewReport:
    return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_FINALIZATION_STATUS_NO_ACTION_NEEDED,
        checks=(),
        readiness=(),
        blocking_reasons=(),
        required_actions=(),
        source_recovery_closure_governance_seal_audit_report=dict(
            recovery_closure_governance_seal_audit
        ),
    )


def _blocked_report(
    *,
    recovery_closure_governance_seal_audit: Mapping[str, Any],
    checks: tuple[
        MemoryEvidenceRepairRecoveryClosureFinalizationReadinessCheck, ...
    ],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessPreviewReport:
    return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessPreviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_CLOSURE_FINALIZATION_STATUS_BLOCKED,
        checks=checks,
        readiness=(),
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
        source_recovery_closure_governance_seal_audit_report=dict(
            recovery_closure_governance_seal_audit
        ),
    )


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessCheck:
    return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessCheck(
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
) -> MemoryEvidenceRepairRecoveryClosureFinalizationReadinessCheck:
    return MemoryEvidenceRepairRecoveryClosureFinalizationReadinessCheck(
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
