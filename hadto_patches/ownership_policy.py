"""Canonical ownership policy for backlog selection and Linear writeback."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import re
import threading
from typing import Any, Dict, Iterator, List, Optional


_OWNER_LABEL_PREFIX = "owner:"
_TERMINAL_STATES = {"done", "completed", "closed", "canceled", "cancelled"}
_THREAD_LOCAL = threading.local()
_ORCHESTRATOR_DEDUPE_PREFIX = "workspace-orchestrator"


@dataclass(frozen=True)
class OwnershipPolicyOverride:
    """Explicit operator override scoped to the current thread."""

    reason: str


@dataclass(frozen=True)
class IssueOwnershipFacts:
    """Facts needed to decide issue selection and ownership writes."""

    issue_id: Optional[str] = None
    issue_key: Optional[str] = None
    project_name: Optional[str] = None
    state_name: Optional[str] = None
    state_type: Optional[str] = None
    issue_type: Optional[str] = None
    repo_resolved: Optional[bool] = None
    planning_only: bool = False
    already_undelegated: bool = False
    assignee_id: Optional[str] = None
    assignee_name: Optional[str] = None
    assignee_email: Optional[str] = None
    assignee_is_human: bool = False
    delegate_id: Optional[str] = None
    delegate_name: Optional[str] = None
    delegate_is_hermes: bool = False
    label_names: tuple[str, ...] = ()
    required_owner_label: Optional[str] = None
    issue_text_has_guard: bool = False
    issue_text_guard_reason: Optional[str] = None
    explicit_override: bool = False
    explicit_override_reason: Optional[str] = None


@dataclass(frozen=True)
class OwnershipDimensionDecision:
    allowed: bool
    reason: str
    detail: Optional[str] = None


@dataclass(frozen=True)
class OwnershipPolicyDecision:
    selectable: OwnershipDimensionDecision
    commentable: OwnershipDimensionDecision
    ownable: OwnershipDimensionDecision
    delegateable: OwnershipDimensionDecision
    assignable: OwnershipDimensionDecision
    override_reason: Optional[str] = None


@dataclass(frozen=True)
class OwnershipDecisionAuditRecord:
    """Structured operator-visible record for a backlog ownership decision."""

    action: str
    outcome: str
    reason: str
    issue_id: Optional[str] = None
    issue_key: Optional[str] = None
    dedupe_key: Optional[str] = None
    policy_reason: Optional[str] = None
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "action": self.action,
            "outcome": self.outcome,
            "reason": self.reason,
        }
        for key in ("issue_id", "issue_key", "dedupe_key", "policy_reason", "detail"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result

    def status_line(self) -> str:
        parts = ["ownership_decision"]
        issue = self.issue_key or self.issue_id
        if issue:
            parts.append(_status_token("issue", issue))
        if self.dedupe_key:
            parts.append(_status_token("dedupe_key", self.dedupe_key))
        parts.extend(
            [
                _status_token("action", self.action),
                _status_token("outcome", self.outcome),
                _status_token("reason", self.reason),
            ]
        )
        if self.policy_reason:
            parts.append(_status_token("policy_reason", self.policy_reason))
        if self.detail:
            parts.append(_status_token("detail", self.detail))
        return " ".join(parts)


def _normalize(value: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _normalize_owner_label(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def _status_token(key: str, value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        text = "unknown"
    if re.search(r"\s", text):
        text = json.dumps(text)
    return f"{key}={text}"


def canonical_ownership_dedupe_key(issue_key: Optional[str]) -> Optional[str]:
    """Return the stable Linear comment/update key for workspace orchestration."""

    clean = str(issue_key or "").strip().upper()
    if not clean:
        return None
    return f"{_ORCHESTRATOR_DEDUPE_PREFIX}:{clean}"


def _override_stack() -> list[OwnershipPolicyOverride]:
    stack = getattr(_THREAD_LOCAL, "override_stack", None)
    if stack is None:
        stack = []
        _THREAD_LOCAL.override_stack = stack
    return stack


def current_ownership_policy_override() -> Optional[OwnershipPolicyOverride]:
    stack = _override_stack()
    return stack[-1] if stack else None


@contextmanager
def ownership_policy_override(reason: str) -> Iterator[OwnershipPolicyOverride]:
    clean_reason = str(reason or "").strip()
    if not clean_reason:
        raise ValueError("ownership policy override requires a reason")
    override = OwnershipPolicyOverride(reason=clean_reason)
    stack = _override_stack()
    stack.append(override)
    try:
        yield override
    finally:
        stack.pop()


def _input_override(facts: IssueOwnershipFacts) -> Optional[OwnershipPolicyOverride]:
    if not facts.explicit_override:
        return None
    reason = str(facts.explicit_override_reason or "").strip()
    if not reason:
        reason = "explicit ownership policy override"
    return OwnershipPolicyOverride(reason=reason)


def _active_override(facts: IssueOwnershipFacts) -> Optional[OwnershipPolicyOverride]:
    return _input_override(facts) or current_ownership_policy_override()


def _owner_label_denial(facts: IssueOwnershipFacts) -> Optional[OwnershipDimensionDecision]:
    required = _normalize_owner_label(facts.required_owner_label)
    if not required:
        return None

    owner_labels = sorted(
        {
            normalized
            for label in facts.label_names
            if (normalized := _normalize_owner_label(label)).startswith(_OWNER_LABEL_PREFIX)
        }
    )
    if required in owner_labels and len(owner_labels) == 1:
        return None
    if len(owner_labels) > 1:
        return OwnershipDimensionDecision(False, "owner_label_conflict", ",".join(owner_labels))
    if owner_labels:
        return OwnershipDimensionDecision(False, "owner_label_mismatch", owner_labels[0])
    return OwnershipDimensionDecision(False, "owner_label_missing", required)


def _denied_decision(
    *,
    selectable: OwnershipDimensionDecision,
    commentable: OwnershipDimensionDecision,
    own_reason: str,
    own_detail: Optional[str],
    override: Optional[OwnershipPolicyOverride],
) -> OwnershipPolicyDecision:
    ownable = OwnershipDimensionDecision(False, own_reason, own_detail)
    return OwnershipPolicyDecision(
        selectable=selectable,
        commentable=commentable,
        ownable=ownable,
        delegateable=ownable,
        assignable=ownable,
        override_reason=override.reason if override else None,
    )


def evaluate_ownership_policy(facts: IssueOwnershipFacts) -> OwnershipPolicyDecision:
    """Return independent selectable/commentable/ownable decisions for an issue."""

    override = _active_override(facts)
    state_values = [
        value
        for value in (
            _normalize(facts.state_type),
            _normalize(facts.state_name),
            _normalize(facts.issue_type),
        )
        if value
    ]
    terminal_state = next(
        (value for value in state_values if value in _TERMINAL_STATES),
        None,
    )
    human_owned = facts.assignee_is_human and not facts.delegate_is_hermes

    commentable = OwnershipDimensionDecision(True, "status_comments_allowed")

    if facts.issue_text_has_guard:
        detail = facts.issue_text_guard_reason
        return _denied_decision(
            selectable=OwnershipDimensionDecision(False, "issue_text_guard", detail),
            commentable=commentable,
            own_reason="issue_text_guard",
            own_detail=detail,
            override=override,
        )

    if terminal_state:
        return _denied_decision(
            selectable=OwnershipDimensionDecision(False, "terminal_state", terminal_state),
            commentable=commentable,
            own_reason="terminal_state",
            own_detail=terminal_state,
            override=override,
        )

    owner_label_denial = _owner_label_denial(facts)
    if owner_label_denial:
        return _denied_decision(
            selectable=owner_label_denial,
            commentable=commentable,
            own_reason=owner_label_denial.reason,
            own_detail=owner_label_denial.detail,
            override=override,
        )

    if human_owned:
        return _denied_decision(
            selectable=OwnershipDimensionDecision(False, "human_owned", facts.assignee_name),
            commentable=commentable,
            own_reason="human_owned",
            own_detail=facts.assignee_name,
            override=override,
        )

    detail = "normal_backlog"
    if facts.delegate_is_hermes:
        detail = "hermes_delegate"
    return OwnershipPolicyDecision(
        selectable=OwnershipDimensionDecision(True, "selected", detail),
        commentable=commentable,
        ownable=OwnershipDimensionDecision(True, "ownership_allowed", detail),
        delegateable=OwnershipDimensionDecision(True, "delegate_allowed", detail),
        assignable=OwnershipDimensionDecision(True, "assign_allowed", detail),
        override_reason=override.reason if override else None,
    )


def _record(
    *,
    action: str,
    outcome: str,
    reason: str,
    issue_id: Optional[str],
    issue_key: Optional[str],
    dedupe_key: Optional[str],
    policy_reason: Optional[str] = None,
    detail: Optional[str] = None,
) -> OwnershipDecisionAuditRecord:
    return OwnershipDecisionAuditRecord(
        action=action,
        outcome=outcome,
        reason=reason,
        issue_id=issue_id,
        issue_key=issue_key,
        dedupe_key=dedupe_key,
        policy_reason=policy_reason,
        detail=detail,
    )


def _mutation_record(
    *,
    action: str,
    changed: bool,
    decision: OwnershipDimensionDecision,
    allowed_reason: str,
    denied_reason: str,
    changed_outcome: str,
    issue_id: Optional[str],
    issue_key: Optional[str],
    dedupe_key: Optional[str],
    repo_resolved: bool,
    planning_only: bool,
    writeback_skipped: bool,
) -> OwnershipDecisionAuditRecord:
    policy_reason = allowed_reason if decision.allowed else denied_reason
    if not repo_resolved:
        return _record(
            action=action,
            outcome="denied",
            reason="repo_unresolved",
            issue_id=issue_id,
            issue_key=issue_key,
            dedupe_key=dedupe_key,
            policy_reason=denied_reason,
        )
    if planning_only:
        return _record(
            action=action,
            outcome="skipped",
            reason="planning_only",
            issue_id=issue_id,
            issue_key=issue_key,
            dedupe_key=dedupe_key,
            policy_reason=policy_reason,
            detail=decision.reason,
        )
    if not decision.allowed:
        return _record(
            action=action,
            outcome="denied",
            reason=decision.reason,
            issue_id=issue_id,
            issue_key=issue_key,
            dedupe_key=dedupe_key,
            policy_reason=denied_reason,
            detail=decision.detail,
        )
    if writeback_skipped or not changed:
        return _record(
            action=action,
            outcome="skipped",
            reason="writeback_skipped",
            issue_id=issue_id,
            issue_key=issue_key,
            dedupe_key=dedupe_key,
            policy_reason=allowed_reason,
            detail=decision.detail,
        )
    return _record(
        action=action,
        outcome=changed_outcome,
        reason=allowed_reason,
        issue_id=issue_id,
        issue_key=issue_key,
        dedupe_key=dedupe_key,
        policy_reason=allowed_reason,
        detail=decision.detail,
    )


def build_ownership_decision_audit_records(
    facts: IssueOwnershipFacts,
    decision: Optional[OwnershipPolicyDecision] = None,
    *,
    issue_id: Optional[str] = None,
    issue_key: Optional[str] = None,
    selected: bool = False,
    live_execution_started: bool = False,
    comment_attempted: bool = False,
    comment_written: bool = False,
    delegate_attempted: bool = False,
    delegated: bool = False,
    assign_attempted: bool = False,
    assigned: bool = False,
    undelegate_attempted: bool = False,
    undelegated: bool = False,
    writeback_skipped: bool = False,
    repo_resolved: Optional[bool] = None,
    planning_only: Optional[bool] = None,
    already_undelegated: Optional[bool] = None,
) -> List[OwnershipDecisionAuditRecord]:
    """Build structured audit records for workspace-backlog ownership decisions.

    The records are intentionally write-neutral: callers can log or include them
    in a canonical Linear comment without changing dedupe/writeback behavior.
    """

    decision = decision or evaluate_ownership_policy(facts)
    resolved_issue_id = issue_id or facts.issue_id
    resolved_issue_key = issue_key or facts.issue_key
    dedupe_key = canonical_ownership_dedupe_key(resolved_issue_key)
    resolved_repo = facts.repo_resolved if repo_resolved is None else repo_resolved
    repo_ok = True if resolved_repo is None else bool(resolved_repo)
    planning = facts.planning_only if planning_only is None else bool(planning_only)
    already_clear = (
        facts.already_undelegated
        if already_undelegated is None
        else bool(already_undelegated)
    )

    records: List[OwnershipDecisionAuditRecord] = []
    if selected and decision.selectable.allowed:
        records.append(
            _record(
                action="select",
                outcome="selected",
                reason="selected",
                issue_id=resolved_issue_id,
                issue_key=resolved_issue_key,
                dedupe_key=dedupe_key,
                policy_reason=decision.selectable.reason,
                detail=decision.selectable.detail,
            )
        )
    else:
        records.append(
            _record(
                action="select",
                outcome="denied" if not decision.selectable.allowed else "skipped",
                reason=decision.selectable.reason if not decision.selectable.allowed else "writeback_skipped",
                issue_id=resolved_issue_id,
                issue_key=resolved_issue_key,
                dedupe_key=dedupe_key,
                policy_reason=decision.selectable.reason,
                detail=decision.selectable.detail,
            )
        )

    if selected or live_execution_started or planning or not repo_ok:
        execution_reason = "live_execution"
        execution_outcome = "started"
        execution_detail = None
        if not repo_ok:
            execution_reason = "repo_unresolved"
            execution_outcome = "skipped"
        elif planning:
            execution_reason = "planning_only"
            execution_outcome = "skipped"
        records.append(
            _record(
                action="execute",
                outcome=execution_outcome,
                reason=execution_reason,
                issue_id=resolved_issue_id,
                issue_key=resolved_issue_key,
                dedupe_key=dedupe_key,
                detail=execution_detail,
            )
        )

    if comment_attempted:
        if not decision.commentable.allowed:
            records.append(
                _record(
                    action="comment",
                    outcome="denied",
                    reason=decision.commentable.reason,
                    issue_id=resolved_issue_id,
                    issue_key=resolved_issue_key,
                    dedupe_key=dedupe_key,
                    policy_reason=decision.commentable.reason,
                    detail=decision.commentable.detail,
                )
            )
        elif comment_written:
            records.append(
                _record(
                    action="comment",
                    outcome="commented",
                    reason="commented",
                    issue_id=resolved_issue_id,
                    issue_key=resolved_issue_key,
                    dedupe_key=dedupe_key,
                    policy_reason=decision.commentable.reason,
                )
            )
        else:
            records.append(
                _record(
                    action="comment",
                    outcome="skipped",
                    reason="writeback_skipped",
                    issue_id=resolved_issue_id,
                    issue_key=resolved_issue_key,
                    dedupe_key=dedupe_key,
                    policy_reason=decision.commentable.reason,
                )
            )

    if delegate_attempted:
        records.append(
            _mutation_record(
                action="delegate",
                changed=delegated,
                decision=decision.delegateable,
                allowed_reason="delegate_allowed",
                denied_reason="delegate_denied",
                changed_outcome="delegated",
                issue_id=resolved_issue_id,
                issue_key=resolved_issue_key,
                dedupe_key=dedupe_key,
                repo_resolved=repo_ok,
                planning_only=planning,
                writeback_skipped=writeback_skipped,
            )
        )

    if assign_attempted:
        records.append(
            _mutation_record(
                action="assign",
                changed=assigned,
                decision=decision.assignable,
                allowed_reason="assign_allowed",
                denied_reason="assign_denied",
                changed_outcome="assigned",
                issue_id=resolved_issue_id,
                issue_key=resolved_issue_key,
                dedupe_key=dedupe_key,
                repo_resolved=repo_ok,
                planning_only=planning,
                writeback_skipped=writeback_skipped,
            )
        )

    if undelegate_attempted:
        if already_clear:
            records.append(
                _record(
                    action="undelegate",
                    outcome="skipped",
                    reason="already_undelegated",
                    issue_id=resolved_issue_id,
                    issue_key=resolved_issue_key,
                    dedupe_key=dedupe_key,
                )
            )
        else:
            records.append(
                _mutation_record(
                    action="undelegate",
                    changed=undelegated,
                    decision=decision.delegateable,
                    allowed_reason="delegate_allowed",
                    denied_reason="delegate_denied",
                    changed_outcome="undelegated",
                    issue_id=resolved_issue_id,
                    issue_key=resolved_issue_key,
                    dedupe_key=dedupe_key,
                    repo_resolved=repo_ok,
                    planning_only=planning,
                    writeback_skipped=writeback_skipped,
                )
            )

    return records


def format_ownership_decision_audit(
    records: List[OwnershipDecisionAuditRecord],
) -> str:
    """Return terse status/log/output lines for ownership audit records."""

    return "\n".join(record.status_line() for record in records)
