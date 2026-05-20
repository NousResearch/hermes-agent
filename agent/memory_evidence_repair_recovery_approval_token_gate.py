"""Read-only recovery approval token verification gate.

The gate verifies a drafted recovery human approval token against the current
recovery execution preview, confirmation text, expiry window, and one-time-use
hints. It never writes to durable memory stores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from agent.memory_evidence_repair_manual_commit_dry_run import (
    CHECK_STATUS_FAIL,
    CHECK_STATUS_PASS,
)
from agent.memory_evidence_repair_recovery_execution_preview import (
    RECOVERY_EXECUTION_STATUS_NO_ACTION_NEEDED,
    RECOVERY_EXECUTION_STATUS_READY,
)
from agent.memory_evidence_repair_recovery_human_approval_token import (
    RECOVERY_APPROVAL_TOKEN_SCOPE,
    RECOVERY_APPROVAL_TOKEN_TYPE,
    RECOVERY_TOKEN_STATUS_DRAFT_READY,
    RECOVERY_TOKEN_STATUS_NO_ACTION_NEEDED,
)


RECOVERY_TOKEN_GATE_STATUS_VERIFIED = "verified_for_manual_recovery"
RECOVERY_TOKEN_GATE_STATUS_BLOCKED = "blocked"
RECOVERY_TOKEN_GATE_STATUS_NO_ACTION_NEEDED = "no_action_needed"

CHECK_RECOVERY_TOKEN_SHAPE = "recovery_token_shape"
CHECK_RECOVERY_CONFIRMATION_TEXT = "recovery_confirmation_text"
CHECK_RECOVERY_TOKEN_DIGEST = "recovery_token_digest"
CHECK_EXECUTION_PREVIEW_DIGEST = "execution_preview_digest"
CHECK_FUTURE_MUTATION_STEPS = "future_mutation_steps"
CHECK_RECOVERY_EXPIRY = "recovery_expiry"
CHECK_RECOVERY_ONE_TIME_CONSTRAINTS = "recovery_one_time_constraints"
CHECK_RECOVERY_REUSE = "recovery_reuse"


@dataclass(frozen=True)
class MemoryEvidenceRepairRecoveryApprovalTokenGateCheck:
    """One read-only recovery approval token verification check."""

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
class MemoryEvidenceRepairRecoveryApprovalTokenGateReport:
    """Read-only recovery approval token gate report."""

    generated_at: str
    status: str
    token_id: str
    execution_preview_status: str
    checks: tuple[MemoryEvidenceRepairRecoveryApprovalTokenGateCheck, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    verified_operation_ids: tuple[str, ...]
    verified_step_ids: tuple[str, ...]
    verified_future_mutation_step_ids: tuple[str, ...]
    source_token: dict[str, Any]
    source_recovery_execution_preview: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        by_check_status: dict[str, int] = {}
        for check in self.checks:
            by_check_status[check.status] = by_check_status.get(check.status, 0) + 1
        return {
            "status": self.status,
            "token_id": self.token_id,
            "check_count": len(self.checks),
            "pass_count": by_check_status.get(CHECK_STATUS_PASS, 0),
            "fail_count": by_check_status.get(CHECK_STATUS_FAIL, 0),
            "blocking_reason_count": len(self.blocking_reasons),
            "required_action_count": len(self.required_actions),
            "verified_operation_count": len(self.verified_operation_ids),
            "verified_step_count": len(self.verified_step_ids),
            "verified_future_mutation_step_count": len(
                self.verified_future_mutation_step_ids
            ),
            "manual_recovery_verified": self.status == RECOVERY_TOKEN_GATE_STATUS_VERIFIED,
            "has_blocks": self.status == RECOVERY_TOKEN_GATE_STATUS_BLOCKED,
            "requires_followup": bool(self.required_actions),
            "by_check_status": by_check_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "token_id": self.token_id,
            "execution_preview_status": self.execution_preview_status,
            "checks": [check.to_dict() for check in self.checks],
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "verified_operation_ids": list(self.verified_operation_ids),
            "verified_step_ids": list(self.verified_step_ids),
            "verified_future_mutation_step_ids": list(
                self.verified_future_mutation_step_ids
            ),
            "source_token": dict(self.source_token),
            "source_recovery_execution_preview": dict(
                self.source_recovery_execution_preview
            ),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_recovery_approval_token_gate(
    *,
    recovery_approval_token: Mapping[str, Any] | None = None,
    recovery_execution: Mapping[str, Any] | None = None,
    confirmation_text: str = "",
    used_token_ids: Sequence[str] | None = None,
    current_time: str | datetime | None = None,
) -> MemoryEvidenceRepairRecoveryApprovalTokenGateReport:
    """Verify a recovery human approval token without mutating memory."""

    token, issued_at = _extract_token_and_issued_at(recovery_approval_token)
    explicit_execution = (
        recovery_execution if isinstance(recovery_execution, Mapping) else {}
    )
    token_execution = token.get("source_execution_preview") if isinstance(token, Mapping) else {}
    current_execution = explicit_execution or (
        token_execution if isinstance(token_execution, Mapping) else {}
    )
    current_execution = current_execution if isinstance(current_execution, Mapping) else {}

    token_report_status = (
        _str(recovery_approval_token.get("status"))
        if isinstance(recovery_approval_token, Mapping)
        else ""
    )
    if not token:
        execution_status = _str(current_execution.get("status"))
        if (
            token_report_status == RECOVERY_TOKEN_STATUS_NO_ACTION_NEEDED
            or execution_status == RECOVERY_EXECUTION_STATUS_NO_ACTION_NEEDED
            or (not current_execution and not token_report_status)
        ):
            return _empty_report(recovery_execution=current_execution)
        source_blocking = (
            _list(recovery_approval_token.get("blocking_reasons"))
            if isinstance(recovery_approval_token, Mapping)
            else []
        )
        source_actions = (
            _list(recovery_approval_token.get("required_actions"))
            if isinstance(recovery_approval_token, Mapping)
            else []
        )
        return _blocked_report(
            token_id="",
            recovery_execution=current_execution,
            checks=(
                _fail(
                    CHECK_RECOVERY_TOKEN_SHAPE,
                    "No recovery approval token was supplied."
                    if not source_blocking
                    else "; ".join(_str(item) for item in source_blocking if _str(item)),
                    tuple(_str(item) for item in source_actions if _str(item))
                    or ("draft_recovery_human_approval_token",),
                    {"recovery_approval_token_status": token_report_status},
                ),
            ),
        )

    checks = tuple(
        check
        for check in (
            _token_shape_check(token),
            _confirmation_check(token, confirmation_text),
            _token_digest_check(token),
            _execution_preview_digest_check(token, current_execution),
            _future_mutation_step_check(token, current_execution),
            _expiry_check(token, issued_at, current_time),
            _one_time_constraint_check(token),
            _reuse_check(token, used_token_ids),
        )
        if check is not None
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
    status = (
        RECOVERY_TOKEN_GATE_STATUS_VERIFIED
        if checks and not blocking_reasons
        else RECOVERY_TOKEN_GATE_STATUS_BLOCKED
    )
    return MemoryEvidenceRepairRecoveryApprovalTokenGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        token_id=_str(token.get("id")),
        execution_preview_status=_str(current_execution.get("status")),
        checks=checks,
        blocking_reasons=blocking_reasons,
        required_actions=required_actions,
        verified_operation_ids=tuple(_list_of_str(current_execution.get("operation_ids"))),
        verified_step_ids=tuple(_list_of_str(current_execution.get("step_ids"))),
        verified_future_mutation_step_ids=tuple(_future_mutation_step_ids(current_execution)),
        source_token=dict(token),
        source_recovery_execution_preview=dict(current_execution),
    )


def empty_evidence_repair_recovery_approval_token_gate() -> MemoryEvidenceRepairRecoveryApprovalTokenGateReport:
    """Return an empty read-only recovery approval token gate report."""

    return _empty_report(recovery_execution={})


def _empty_report(
    *,
    recovery_execution: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryApprovalTokenGateReport:
    return MemoryEvidenceRepairRecoveryApprovalTokenGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_TOKEN_GATE_STATUS_NO_ACTION_NEEDED,
        token_id="",
        execution_preview_status=_str(recovery_execution.get("status")),
        checks=(),
        blocking_reasons=(),
        required_actions=(),
        verified_operation_ids=(),
        verified_step_ids=(),
        verified_future_mutation_step_ids=(),
        source_token={},
        source_recovery_execution_preview=dict(recovery_execution),
    )


def _blocked_report(
    *,
    token_id: str,
    recovery_execution: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairRecoveryApprovalTokenGateCheck, ...],
) -> MemoryEvidenceRepairRecoveryApprovalTokenGateReport:
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
    return MemoryEvidenceRepairRecoveryApprovalTokenGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=RECOVERY_TOKEN_GATE_STATUS_BLOCKED,
        token_id=token_id,
        execution_preview_status=_str(recovery_execution.get("status")),
        checks=checks,
        blocking_reasons=blocking_reasons,
        required_actions=required_actions,
        verified_operation_ids=(),
        verified_step_ids=(),
        verified_future_mutation_step_ids=(),
        source_token={},
        source_recovery_execution_preview=dict(recovery_execution),
    )


def _extract_token_and_issued_at(
    recovery_approval_token: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    if not isinstance(recovery_approval_token, Mapping):
        return {}, ""
    report_issued_at = _str(recovery_approval_token.get("generated_at"))
    token: Mapping[str, Any] | None = None
    tokens = recovery_approval_token.get("tokens")
    if isinstance(tokens, list):
        token = next((item for item in tokens if isinstance(item, Mapping)), None)
    elif _str(recovery_approval_token.get("id")):
        token = recovery_approval_token
    if not isinstance(token, Mapping):
        return {}, report_issued_at
    issued_at = (
        _str(token.get("issued_at"))
        or _str(token.get("generated_at"))
        or report_issued_at
    )
    return dict(token), issued_at


def _token_shape_check(
    token: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryApprovalTokenGateCheck:
    reasons: list[str] = []
    if not _str(token.get("id")).startswith("recovery-approval-token-"):
        reasons.append("Recovery token id is missing or invalid.")
    if _str(token.get("token_type")) != RECOVERY_APPROVAL_TOKEN_TYPE:
        reasons.append("Token type is not a memory evidence repair recovery token.")
    if _str(token.get("scope")) != RECOVERY_APPROVAL_TOKEN_SCOPE:
        reasons.append("Token scope does not match manual memory evidence repair recovery.")
    if _str(token.get("status")) != RECOVERY_TOKEN_STATUS_DRAFT_READY:
        reasons.append("Recovery token is not a ready approval draft.")
    if reasons:
        return _fail(
            CHECK_RECOVERY_TOKEN_SHAPE,
            " ".join(reasons),
            ("regenerate_recovery_human_approval_token",),
            {"token_id": _str(token.get("id"))},
        )
    return _pass(
        CHECK_RECOVERY_TOKEN_SHAPE,
        "Recovery approval token shape is valid.",
        {"token_id": _str(token.get("id"))},
    )


def _confirmation_check(
    token: Mapping[str, Any],
    confirmation_text: str,
) -> MemoryEvidenceRepairRecoveryApprovalTokenGateCheck:
    expected = _str(token.get("required_confirmation_text"))
    actual = _str(confirmation_text)
    constraints = _dict(token.get("one_time_constraints"))
    constraint_expected = _str(constraints.get("requires_exact_confirmation_text"))
    expected_values = {item for item in (expected, constraint_expected) if item}
    if not actual or actual not in expected_values:
        return _fail(
            CHECK_RECOVERY_CONFIRMATION_TEXT,
            "Confirmation text does not exactly match the recovery approval token.",
            ("provide_exact_recovery_confirmation_text",),
            {"expected_confirmation_text": expected or constraint_expected, "provided": actual},
        )
    return _pass(
        CHECK_RECOVERY_CONFIRMATION_TEXT,
        "Confirmation text matches the recovery approval token.",
        {"provided": actual},
    )


def _token_digest_check(
    token: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryApprovalTokenGateCheck:
    expected_digest = _expected_token_digest(token)
    actual_digest = _str(token.get("token_digest"))
    expected_id = f"recovery-approval-token-{expected_digest[:16]}" if expected_digest else ""
    if actual_digest != expected_digest or _str(token.get("id")) != expected_id:
        return _fail(
            CHECK_RECOVERY_TOKEN_DIGEST,
            "Recovery approval token digest or id does not match its declared payload.",
            ("regenerate_recovery_human_approval_token",),
            {
                "expected_token_digest": expected_digest,
                "actual_token_digest": actual_digest,
                "expected_token_id": expected_id,
                "actual_token_id": _str(token.get("id")),
            },
        )
    return _pass(
        CHECK_RECOVERY_TOKEN_DIGEST,
        "Recovery approval token digest matches its declared payload.",
        {"token_digest": actual_digest, "token_id": expected_id},
    )


def _execution_preview_digest_check(
    token: Mapping[str, Any],
    recovery_execution: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryApprovalTokenGateCheck:
    if _str(recovery_execution.get("status")) != RECOVERY_EXECUTION_STATUS_READY:
        return _fail(
            CHECK_EXECUTION_PREVIEW_DIGEST,
            "Current recovery execution preview is not ready.",
            ("produce_ready_recovery_execution_preview",),
            {"recovery_execution_status": _str(recovery_execution.get("status"))},
        )
    actual = _execution_preview_digest(recovery_execution)
    expected = _str(token.get("execution_preview_digest"))
    if not expected or actual != expected:
        return _fail(
            CHECK_EXECUTION_PREVIEW_DIGEST,
            "Current recovery execution preview digest does not match the token.",
            ("regenerate_recovery_human_approval_token_for_current_execution_preview",),
            {
                "expected_execution_preview_digest": expected,
                "actual_execution_preview_digest": actual,
            },
        )
    return _pass(
        CHECK_EXECUTION_PREVIEW_DIGEST,
        "Current recovery execution preview digest matches the token.",
        {"execution_preview_digest": actual},
    )


def _future_mutation_step_check(
    token: Mapping[str, Any],
    recovery_execution: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryApprovalTokenGateCheck:
    expected = _str_list(token.get("future_mutation_step_ids"))
    actual = _future_mutation_step_ids(recovery_execution)
    constraints = _dict(token.get("one_time_constraints"))
    constraint_expected = _str_list(
        constraints.get("requires_same_future_mutation_step_ids")
    )
    if expected != actual or (constraint_expected and constraint_expected != actual):
        return _fail(
            CHECK_FUTURE_MUTATION_STEPS,
            "Current future mutation steps do not match the recovery approval token.",
            ("regenerate_recovery_human_approval_token_for_current_execution_preview",),
            {
                "expected_future_mutation_step_ids": expected,
                "constraint_future_mutation_step_ids": constraint_expected,
                "actual_future_mutation_step_ids": actual,
            },
        )
    return _pass(
        CHECK_FUTURE_MUTATION_STEPS,
        "Current future mutation steps match the recovery approval token.",
        {"future_mutation_step_ids": actual},
    )


def _expiry_check(
    token: Mapping[str, Any],
    issued_at: str,
    current_time: str | datetime | None,
) -> MemoryEvidenceRepairRecoveryApprovalTokenGateCheck:
    issued = _parse_datetime(issued_at)
    now = _parse_datetime(current_time) or datetime.now(timezone.utc)
    expires_in_minutes = _positive_int(token.get("expires_in_minutes")) or 15
    if issued is None:
        return _fail(
            CHECK_RECOVERY_EXPIRY,
            "Recovery approval token issue time is missing or invalid.",
            ("provide_recovery_token_generated_at_or_regenerate_token",),
            {"issued_at": issued_at},
        )
    expires_at = issued + timedelta(minutes=expires_in_minutes)
    if now > expires_at:
        return _fail(
            CHECK_RECOVERY_EXPIRY,
            "Recovery approval token has expired.",
            ("regenerate_recovery_human_approval_token",),
            {
                "issued_at": issued.isoformat(),
                "expires_at": expires_at.isoformat(),
                "current_time": now.isoformat(),
            },
        )
    return _pass(
        CHECK_RECOVERY_EXPIRY,
        "Recovery approval token is within its expiry window.",
        {
            "issued_at": issued.isoformat(),
            "expires_at": expires_at.isoformat(),
            "current_time": now.isoformat(),
        },
    )


def _one_time_constraint_check(
    token: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryApprovalTokenGateCheck:
    constraints = _dict(token.get("one_time_constraints"))
    if constraints.get("one_time_use") is not True:
        return _fail(
            CHECK_RECOVERY_ONE_TIME_CONSTRAINTS,
            "Recovery approval token does not declare one-time-use constraints.",
            ("regenerate_recovery_human_approval_token",),
            {"one_time_constraints": constraints},
        )
    expected_execution = _str(
        constraints.get("requires_exact_execution_preview_digest")
    )
    if expected_execution != _str(token.get("execution_preview_digest")):
        return _fail(
            CHECK_RECOVERY_ONE_TIME_CONSTRAINTS,
            "Recovery token execution-preview constraint does not match its digest.",
            ("regenerate_recovery_human_approval_token",),
            {"one_time_constraints": constraints},
        )
    if _str(constraints.get("requires_exact_decision_id")) != _str(token.get("decision_id")):
        return _fail(
            CHECK_RECOVERY_ONE_TIME_CONSTRAINTS,
            "Recovery token decision constraint does not match its decision id.",
            ("regenerate_recovery_human_approval_token",),
            {"one_time_constraints": constraints},
        )
    return _pass(
        CHECK_RECOVERY_ONE_TIME_CONSTRAINTS,
        "Recovery approval token one-time constraints are declared.",
        {"one_time_constraints": constraints},
    )


def _reuse_check(
    token: Mapping[str, Any],
    used_token_ids: Sequence[str] | None,
) -> MemoryEvidenceRepairRecoveryApprovalTokenGateCheck:
    token_id = _str(token.get("id"))
    used = {_str(item) for item in used_token_ids or [] if _str(item)}
    if token_id and token_id in used:
        return _fail(
            CHECK_RECOVERY_REUSE,
            "Recovery approval token is already marked as used.",
            ("regenerate_recovery_human_approval_token",),
            {"token_id": token_id},
        )
    return _pass(
        CHECK_RECOVERY_REUSE,
        "Recovery approval token is not marked as used in the supplied read-only ledger.",
        {"token_id": token_id, "used_token_id_count": len(used)},
    )


def _expected_token_digest(token: Mapping[str, Any]) -> str:
    token_seed = {
        "token_type": RECOVERY_APPROVAL_TOKEN_TYPE,
        "scope": RECOVERY_APPROVAL_TOKEN_SCOPE,
        "approver": _str(token.get("approver")) or "human",
        "approval_reason": _str(token.get("approval_reason")),
        "execution_preview_digest": _str(token.get("execution_preview_digest")),
        "execution_preview_id": _str(token.get("execution_preview_id")),
        "decision_id": _str(token.get("decision_id")),
        "route": _str(token.get("route")),
        "operation_ids": _str_list(token.get("operation_ids")),
        "failed_audit_step_ids": _str_list(token.get("failed_audit_step_ids")),
        "step_ids": _str_list(token.get("step_ids")),
        "future_mutation_step_ids": _str_list(token.get("future_mutation_step_ids")),
        "expires_in_minutes": _positive_int(token.get("expires_in_minutes")) or 15,
    }
    return _digest(token_seed)


def _execution_preview_digest(recovery_execution: Mapping[str, Any]) -> str:
    existing = _str(recovery_execution.get("preview_digest"))
    if existing:
        return existing
    return _digest(
        {
            "id": _str(recovery_execution.get("id")),
            "route": _str(recovery_execution.get("route")),
            "decision_id": _str(recovery_execution.get("decision_id")),
            "operation_ids": tuple(_list_of_str(recovery_execution.get("operation_ids"))),
            "failed_audit_step_ids": tuple(
                _list_of_str(recovery_execution.get("failed_audit_step_ids"))
            ),
            "step_ids": tuple(_list_of_str(recovery_execution.get("step_ids"))),
        }
    )


def _future_mutation_step_ids(recovery_execution: Mapping[str, Any]) -> list[str]:
    return [
        _str(step.get("id"))
        for step in _list(recovery_execution.get("steps"))
        if isinstance(step, Mapping) and step.get("future_would_mutate_memory") is True
    ]


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairRecoveryApprovalTokenGateCheck:
    return MemoryEvidenceRepairRecoveryApprovalTokenGateCheck(
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
) -> MemoryEvidenceRepairRecoveryApprovalTokenGateCheck:
    return MemoryEvidenceRepairRecoveryApprovalTokenGateCheck(
        id=check_id,
        status=CHECK_STATUS_FAIL,
        reason=reason,
        required_actions=tuple(required_actions),
        details=dict(details),
    )


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _list_of_str(value: Any) -> list[str]:
    return [_str(item) for item in value] if isinstance(value, list) else []


def _str_list(value: Any) -> list[str]:
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
