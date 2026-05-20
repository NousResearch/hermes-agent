"""Read-only approval token verification gate for memory evidence repair.

The gate verifies a drafted human approval token against the current dry-run,
confirmation text, expiry window, and one-time-use hints. It never writes to
durable memory stores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from agent.memory_evidence_repair_human_approval_token import (
    APPROVAL_TOKEN_SCOPE,
    APPROVAL_TOKEN_TYPE,
    TOKEN_STATUS_DRAFT_READY,
    TOKEN_STATUS_NO_ACTION_NEEDED,
)
from agent.memory_evidence_repair_manual_commit_dry_run import (
    CHECK_STATUS_FAIL,
    CHECK_STATUS_PASS,
    DRY_RUN_STATUS_NO_ACTION_NEEDED,
    DRY_RUN_STATUS_READY_FOR_MANUAL_COMMIT,
)


TOKEN_GATE_STATUS_VERIFIED = "verified_for_manual_commit"
TOKEN_GATE_STATUS_BLOCKED = "blocked"
TOKEN_GATE_STATUS_NO_ACTION_NEEDED = "no_action_needed"

CHECK_TOKEN_SHAPE = "token_shape"
CHECK_CONFIRMATION_TEXT = "confirmation_text"
CHECK_TOKEN_DIGEST = "token_digest"
CHECK_DRY_RUN_DIGEST = "dry_run_digest"
CHECK_PATCH_DIGESTS = "patch_digests"
CHECK_EXPIRY = "expiry"
CHECK_ONE_TIME_CONSTRAINTS = "one_time_constraints"
CHECK_REUSE = "reuse"


@dataclass(frozen=True)
class MemoryEvidenceRepairApprovalTokenGateCheck:
    """One read-only approval token verification check."""

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
class MemoryEvidenceRepairApprovalTokenGateReport:
    """Read-only approval token gate report."""

    generated_at: str
    status: str
    token_id: str
    dry_run_status: str
    checks: tuple[MemoryEvidenceRepairApprovalTokenGateCheck, ...]
    blocking_reasons: tuple[str, ...]
    required_actions: tuple[str, ...]
    verified_operation_ids: tuple[str, ...]
    verified_candidate_ids: tuple[str, ...]
    verified_patch_digests: tuple[str, ...]
    source_token: dict[str, Any]
    source_dry_run: dict[str, Any]

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
            "manual_commit_verified": self.status == TOKEN_GATE_STATUS_VERIFIED,
            "has_blocks": self.status == TOKEN_GATE_STATUS_BLOCKED,
            "requires_followup": bool(self.required_actions),
            "by_check_status": by_check_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "token_id": self.token_id,
            "dry_run_status": self.dry_run_status,
            "checks": [check.to_dict() for check in self.checks],
            "blocking_reasons": list(self.blocking_reasons),
            "required_actions": list(self.required_actions),
            "verified_operation_ids": list(self.verified_operation_ids),
            "verified_candidate_ids": list(self.verified_candidate_ids),
            "verified_patch_digests": list(self.verified_patch_digests),
            "source_token": dict(self.source_token),
            "source_dry_run": dict(self.source_dry_run),
            "read_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        }


def build_evidence_repair_approval_token_gate(
    *,
    approval_token: Mapping[str, Any] | None = None,
    dry_run: Mapping[str, Any] | None = None,
    confirmation_text: str = "",
    used_token_ids: Sequence[str] | None = None,
    current_time: str | datetime | None = None,
) -> MemoryEvidenceRepairApprovalTokenGateReport:
    """Verify a human approval token without mutating memory."""

    token, issued_at = _extract_token_and_issued_at(approval_token)
    explicit_dry_run = dry_run if isinstance(dry_run, Mapping) else {}
    token_dry_run = token.get("source_dry_run") if isinstance(token, Mapping) else {}
    current_dry_run = explicit_dry_run or (
        token_dry_run if isinstance(token_dry_run, Mapping) else {}
    )
    current_dry_run = current_dry_run if isinstance(current_dry_run, Mapping) else {}

    approval_token_status = (
        _str(approval_token.get("status")) if isinstance(approval_token, Mapping) else ""
    )
    if not token:
        dry_run_status = _str(current_dry_run.get("status"))
        if (
            approval_token_status == TOKEN_STATUS_NO_ACTION_NEEDED
            or dry_run_status == DRY_RUN_STATUS_NO_ACTION_NEEDED
            or (not current_dry_run and not approval_token_status)
        ):
            return _empty_report(dry_run=current_dry_run)
        source_blocking = _list(approval_token.get("blocking_reasons")) if isinstance(approval_token, Mapping) else []
        source_actions = _list(approval_token.get("required_actions")) if isinstance(approval_token, Mapping) else []
        return _blocked_report(
            token_id="",
            dry_run=current_dry_run,
            checks=(
                _fail(
                    CHECK_TOKEN_SHAPE,
                    "No approval token was supplied."
                    if not source_blocking
                    else "; ".join(_str(item) for item in source_blocking if _str(item)),
                    tuple(_str(item) for item in source_actions if _str(item))
                    or ("draft_human_approval_token",),
                    {"approval_token_status": approval_token_status},
                ),
            ),
        )

    checks = tuple(
        check
        for check in (
            _token_shape_check(token),
            _confirmation_check(token, confirmation_text),
            _token_digest_check(token),
            _dry_run_digest_check(token, current_dry_run),
            _patch_digest_check(token, current_dry_run),
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
        TOKEN_GATE_STATUS_VERIFIED
        if checks and not blocking_reasons
        else TOKEN_GATE_STATUS_BLOCKED
    )
    operations = _operation_rows(current_dry_run)
    return MemoryEvidenceRepairApprovalTokenGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        token_id=_str(token.get("id")),
        dry_run_status=_str(current_dry_run.get("status")),
        checks=checks,
        blocking_reasons=blocking_reasons,
        required_actions=required_actions,
        verified_operation_ids=tuple(_str(operation.get("id")) for operation in operations),
        verified_candidate_ids=tuple(_str(operation.get("candidate_id")) for operation in operations),
        verified_patch_digests=tuple(_str(operation.get("patch_digest")) for operation in operations),
        source_token=dict(token),
        source_dry_run=dict(current_dry_run),
    )


def empty_evidence_repair_approval_token_gate() -> MemoryEvidenceRepairApprovalTokenGateReport:
    """Return an empty read-only approval token gate report."""

    return _empty_report(dry_run={})


def _empty_report(*, dry_run: Mapping[str, Any]) -> MemoryEvidenceRepairApprovalTokenGateReport:
    return MemoryEvidenceRepairApprovalTokenGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=TOKEN_GATE_STATUS_NO_ACTION_NEEDED,
        token_id="",
        dry_run_status=_str(dry_run.get("status")),
        checks=(),
        blocking_reasons=(),
        required_actions=(),
        verified_operation_ids=(),
        verified_candidate_ids=(),
        verified_patch_digests=(),
        source_token={},
        source_dry_run=dict(dry_run),
    )


def _blocked_report(
    *,
    token_id: str,
    dry_run: Mapping[str, Any],
    checks: tuple[MemoryEvidenceRepairApprovalTokenGateCheck, ...],
) -> MemoryEvidenceRepairApprovalTokenGateReport:
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
    return MemoryEvidenceRepairApprovalTokenGateReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=TOKEN_GATE_STATUS_BLOCKED,
        token_id=token_id,
        dry_run_status=_str(dry_run.get("status")),
        checks=checks,
        blocking_reasons=blocking_reasons,
        required_actions=required_actions,
        verified_operation_ids=(),
        verified_candidate_ids=(),
        verified_patch_digests=(),
        source_token={},
        source_dry_run=dict(dry_run),
    )


def _extract_token_and_issued_at(
    approval_token: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    if not isinstance(approval_token, Mapping):
        return {}, ""
    report_issued_at = _str(approval_token.get("generated_at"))
    token: Mapping[str, Any] | None = None
    tokens = approval_token.get("tokens")
    if isinstance(tokens, list):
        token = next((item for item in tokens if isinstance(item, Mapping)), None)
    elif _str(approval_token.get("id")):
        token = approval_token
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
) -> MemoryEvidenceRepairApprovalTokenGateCheck:
    reasons: list[str] = []
    if not _str(token.get("id")).startswith("approval-token-"):
        reasons.append("Token id is missing or invalid.")
    if _str(token.get("token_type")) != APPROVAL_TOKEN_TYPE:
        reasons.append("Token type is not a memory evidence repair human approval token.")
    if _str(token.get("scope")) != APPROVAL_TOKEN_SCOPE:
        reasons.append("Token scope does not match manual memory evidence repair commit.")
    if _str(token.get("status")) != TOKEN_STATUS_DRAFT_READY:
        reasons.append("Token is not a ready approval draft.")
    if reasons:
        return _fail(
            CHECK_TOKEN_SHAPE,
            " ".join(reasons),
            ("regenerate_human_approval_token",),
            {"token_id": _str(token.get("id"))},
        )
    return _pass(
        CHECK_TOKEN_SHAPE,
        "Approval token shape is valid.",
        {"token_id": _str(token.get("id"))},
    )


def _confirmation_check(
    token: Mapping[str, Any],
    confirmation_text: str,
) -> MemoryEvidenceRepairApprovalTokenGateCheck:
    expected = _str(token.get("required_confirmation_text"))
    actual = _str(confirmation_text)
    constraints = _dict(token.get("one_time_constraints"))
    constraint_expected = _str(constraints.get("requires_exact_confirmation_text"))
    expected_values = {item for item in (expected, constraint_expected) if item}
    if not actual or actual not in expected_values:
        return _fail(
            CHECK_CONFIRMATION_TEXT,
            "Confirmation text does not exactly match the approval token.",
            ("provide_exact_confirmation_text",),
            {"expected_confirmation_text": expected or constraint_expected, "provided": actual},
        )
    return _pass(
        CHECK_CONFIRMATION_TEXT,
        "Confirmation text matches the approval token.",
        {"provided": actual},
    )


def _token_digest_check(
    token: Mapping[str, Any],
) -> MemoryEvidenceRepairApprovalTokenGateCheck:
    expected_digest = _expected_token_digest(token)
    actual_digest = _str(token.get("token_digest"))
    expected_id = f"approval-token-{expected_digest[:16]}" if expected_digest else ""
    if actual_digest != expected_digest or _str(token.get("id")) != expected_id:
        return _fail(
            CHECK_TOKEN_DIGEST,
            "Approval token digest or id does not match its declared payload.",
            ("regenerate_human_approval_token",),
            {
                "expected_token_digest": expected_digest,
                "actual_token_digest": actual_digest,
                "expected_token_id": expected_id,
                "actual_token_id": _str(token.get("id")),
            },
        )
    return _pass(
        CHECK_TOKEN_DIGEST,
        "Approval token digest matches its declared payload.",
        {"token_digest": actual_digest, "token_id": expected_id},
    )


def _dry_run_digest_check(
    token: Mapping[str, Any],
    dry_run: Mapping[str, Any],
) -> MemoryEvidenceRepairApprovalTokenGateCheck:
    if _str(dry_run.get("status")) != DRY_RUN_STATUS_READY_FOR_MANUAL_COMMIT:
        return _fail(
            CHECK_DRY_RUN_DIGEST,
            "Current dry-run is not ready for manual commit.",
            ("produce_ready_manual_commit_dry_run",),
            {"dry_run_status": _str(dry_run.get("status"))},
        )
    actual = _dry_run_digest(dry_run)
    expected = _str(token.get("dry_run_digest"))
    if not expected or actual != expected:
        return _fail(
            CHECK_DRY_RUN_DIGEST,
            "Current dry-run digest does not match the approval token.",
            ("regenerate_human_approval_token_for_current_dry_run",),
            {"expected_dry_run_digest": expected, "actual_dry_run_digest": actual},
        )
    return _pass(
        CHECK_DRY_RUN_DIGEST,
        "Current dry-run digest matches the approval token.",
        {"dry_run_digest": actual},
    )


def _patch_digest_check(
    token: Mapping[str, Any],
    dry_run: Mapping[str, Any],
) -> MemoryEvidenceRepairApprovalTokenGateCheck:
    expected = _str_list(token.get("patch_digests"))
    actual = [_str(operation.get("patch_digest")) for operation in _operation_rows(dry_run)]
    constraints = _dict(token.get("one_time_constraints"))
    constraint_expected = _str_list(constraints.get("requires_same_patch_digests"))
    if expected != actual or (constraint_expected and constraint_expected != actual):
        return _fail(
            CHECK_PATCH_DIGESTS,
            "Current dry-run patch digests do not match the approval token.",
            ("regenerate_human_approval_token_for_current_patch_set",),
            {
                "expected_patch_digests": expected,
                "constraint_patch_digests": constraint_expected,
                "actual_patch_digests": actual,
            },
        )
    return _pass(
        CHECK_PATCH_DIGESTS,
        "Current dry-run patch digests match the approval token.",
        {"patch_digests": actual},
    )


def _expiry_check(
    token: Mapping[str, Any],
    issued_at: str,
    current_time: str | datetime | None,
) -> MemoryEvidenceRepairApprovalTokenGateCheck:
    issued = _parse_datetime(issued_at)
    now = _parse_datetime(current_time) or datetime.now(timezone.utc)
    expires_in_minutes = _positive_int(token.get("expires_in_minutes")) or 30
    if issued is None:
        return _fail(
            CHECK_EXPIRY,
            "Approval token issue time is missing or invalid.",
            ("provide_token_generated_at_or_regenerate_token",),
            {"issued_at": issued_at},
        )
    expires_at = issued + timedelta(minutes=expires_in_minutes)
    if now > expires_at:
        return _fail(
            CHECK_EXPIRY,
            "Approval token has expired.",
            ("regenerate_human_approval_token",),
            {
                "issued_at": issued.isoformat(),
                "expires_at": expires_at.isoformat(),
                "current_time": now.isoformat(),
            },
        )
    return _pass(
        CHECK_EXPIRY,
        "Approval token is within its expiry window.",
        {
            "issued_at": issued.isoformat(),
            "expires_at": expires_at.isoformat(),
            "current_time": now.isoformat(),
        },
    )


def _one_time_constraint_check(
    token: Mapping[str, Any],
) -> MemoryEvidenceRepairApprovalTokenGateCheck:
    constraints = _dict(token.get("one_time_constraints"))
    if constraints.get("one_time_use") is not True:
        return _fail(
            CHECK_ONE_TIME_CONSTRAINTS,
            "Approval token does not declare one-time-use constraints.",
            ("regenerate_human_approval_token",),
            {"one_time_constraints": constraints},
        )
    expected_dry_run = _str(constraints.get("requires_exact_dry_run_digest"))
    if expected_dry_run != _str(token.get("dry_run_digest")):
        return _fail(
            CHECK_ONE_TIME_CONSTRAINTS,
            "Approval token dry-run constraint does not match its digest.",
            ("regenerate_human_approval_token",),
            {"one_time_constraints": constraints},
        )
    return _pass(
        CHECK_ONE_TIME_CONSTRAINTS,
        "Approval token one-time constraints are declared.",
        {"one_time_constraints": constraints},
    )


def _reuse_check(
    token: Mapping[str, Any],
    used_token_ids: Sequence[str] | None,
) -> MemoryEvidenceRepairApprovalTokenGateCheck:
    token_id = _str(token.get("id"))
    used = {_str(item) for item in used_token_ids or [] if _str(item)}
    if token_id and token_id in used:
        return _fail(
            CHECK_REUSE,
            "Approval token is already marked as used.",
            ("regenerate_human_approval_token",),
            {"token_id": token_id},
        )
    return _pass(
        CHECK_REUSE,
        "Approval token is not marked as used in the supplied read-only ledger.",
        {"token_id": token_id, "used_token_id_count": len(used)},
    )


def _expected_token_digest(token: Mapping[str, Any]) -> str:
    token_seed = {
        "token_type": APPROVAL_TOKEN_TYPE,
        "scope": APPROVAL_TOKEN_SCOPE,
        "approver": _str(token.get("approver")) or "human",
        "approval_reason": _str(token.get("approval_reason")),
        "dry_run_digest": _str(token.get("dry_run_digest")),
        "operation_ids": _str_list(token.get("operation_ids")),
        "ledger_entry_ids": _str_list(token.get("ledger_entry_ids")),
        "candidate_ids": _str_list(token.get("candidate_ids")),
        "patch_digests": _str_list(token.get("patch_digests")),
        "expires_in_minutes": _positive_int(token.get("expires_in_minutes")) or 30,
    }
    return _digest(token_seed)


def _dry_run_digest(dry_run: Mapping[str, Any]) -> str:
    return _digest(
        {
            "status": dry_run.get("status"),
            "operations": _operation_rows(dry_run),
            "checks": dry_run.get("checks"),
        }
    )


def _operation_rows(dry_run: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        operation
        for operation in _list(dry_run.get("operations"))
        if isinstance(operation, Mapping)
    ]


def _pass(
    check_id: str,
    reason: str,
    details: Mapping[str, Any],
) -> MemoryEvidenceRepairApprovalTokenGateCheck:
    return MemoryEvidenceRepairApprovalTokenGateCheck(
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
) -> MemoryEvidenceRepairApprovalTokenGateCheck:
    return MemoryEvidenceRepairApprovalTokenGateCheck(
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
