"""Review gate for managed agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Mapping

import yaml

from .event_log import ManagedAgentEventLog


class ReviewGateError(ValueError):
    """Raised when review configuration or inputs are malformed."""


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    if isinstance(values, list | tuple | set):
        return [str(item).strip() for item in values if str(item).strip()]
    value = str(values).strip()
    return [value] if value else []


def _normalize_risk(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text else "R0"


def _extract_changed_paths(task: Any) -> list[str]:
    for field_name in (
        "changed_files",
        "touched_files",
        "touches",
        "paths_changed",
        "files_changed_list",
        "changed_paths",
    ):
        values = _get_attr(task, field_name, None)
        paths = _normalize_list(values)
        if paths:
            return paths
    return []


def _extract_actions(task: Any) -> list[str]:
    for field_name in ("actions", "action_types", "action_type"):
        values = _get_attr(task, field_name, None)
        actions = _normalize_list(values)
        if actions:
            return actions
    return []


@dataclass(frozen=True, slots=True)
class ReviewSeverity:
    p0: int = 0
    p1: int = 0
    p2: int = 0
    p3: int = 0

    def has_blocking(self) -> bool:
        return self.p0 > 0 or self.p1 > 0

    def to_dict(self) -> dict[str, int]:
        return {
            "p0": self.p0,
            "p1": self.p1,
            "p2": self.p2,
            "p3": self.p3,
        }

    @classmethod
    def from_raw(cls, value: Any) -> "ReviewSeverity":
        if isinstance(value, ReviewSeverity):
            return value
        if isinstance(value, Mapping):
            return cls(
                p0=int(value.get("p0", 0) or 0),
                p1=int(value.get("p1", 0) or 0),
                p2=int(value.get("p2", 0) or 0),
                p3=int(value.get("p3", 0) or 0),
            )
        return cls()


@dataclass(slots=True)
class ReviewRequirement:
    task_id: str
    required_reviewers: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    matched_triggers: list[str] = field(default_factory=list)
    requires_review: bool = False
    requires_human_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "required_reviewers": list(self.required_reviewers),
            "reasons": list(self.reasons),
            "matched_triggers": list(self.matched_triggers),
            "requires_review": self.requires_review,
            "requires_human_approval": self.requires_human_approval,
        }


@dataclass(slots=True)
class ReviewResult:
    task_id: str
    reviewer: str
    decision: str
    severity: ReviewSeverity = field(default_factory=ReviewSeverity)
    summary: str = ""
    required_fixes: list[str] = field(default_factory=list)
    optional_fixes: list[str] = field(default_factory=list)
    required_reviewers: list[str] = field(default_factory=list)
    approvers: list[str] = field(default_factory=list)
    task_executor: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "reviewer": self.reviewer,
            "decision": self.decision,
            "severity": self.severity.to_dict(),
            "summary": self.summary,
            "required_fixes": list(self.required_fixes),
            "optional_fixes": list(self.optional_fixes),
            "required_reviewers": list(self.required_reviewers),
            "approvers": list(self.approvers),
            "task_executor": self.task_executor,
        }


@dataclass(slots=True)
class ReviewRules:
    version: str
    review_required_when: tuple[dict[str, Any], ...]
    reviewers: dict[str, dict[str, Any]]
    source_path: Path | None = None


@dataclass(slots=True)
class ReviewGate:
    rules: ReviewRules
    event_log: ManagedAgentEventLog | None = None

    def build_requirement(self, task: Any) -> ReviewRequirement:
        task_id = str(_get_attr(task, "task_id", "unknown")).strip() or "unknown"
        risk_level = _normalize_risk(_get_attr(task, "risk_level", "R0"))
        changed_paths = _extract_changed_paths(task)
        actions = _extract_actions(task)
        files_changed = self._extract_files_changed(task, changed_paths)

        reasons: list[str] = []
        matched_triggers: list[str] = []

        for rule in self.rules.review_required_when:
            if self._matches_trigger(rule, risk_level, files_changed, changed_paths, actions):
                matched_triggers.append(self._describe_trigger(rule))
                reasons.append(self._describe_reason(rule, risk_level, files_changed))

        required_reviewers = list(self._reviewers_for_risk(risk_level))
        requires_review = bool(matched_triggers)
        requires_human_approval = bool(self.rules.reviewers.get(risk_level, {}).get("requires_human_approval", False))

        if requires_review and risk_level == "R4":
            requires_human_approval = True

        return ReviewRequirement(
            task_id=task_id,
            required_reviewers=required_reviewers,
            reasons=reasons,
            matched_triggers=matched_triggers,
            requires_review=requires_review,
            requires_human_approval=requires_human_approval,
        )

    def should_review(self, task: Any) -> bool:
        return self.build_requirement(task).requires_review

    def required_reviewers(self, task: Any) -> list[str]:
        return self.build_requirement(task).required_reviewers

    def requires_human_approval(self, task: Any) -> bool:
        return self.build_requirement(task).requires_human_approval

    def request_review(
        self,
        *,
        task: Any,
        reviewer: str,
        session_id: str,
        subject_agent: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self.event_log is None:
            return
        task_id = str(_get_attr(task, "task_id", "unknown")).strip() or "unknown"
        self.event_log.log_review_requested(
            task_id=task_id,
            session_id=session_id,
            reviewer=reviewer,
            subject_agent=subject_agent or str(_get_attr(task, "owner_agent", "")),
            metadata=metadata or {},
        )

    def complete_review(
        self,
        *,
        task: Any,
        reviewer: str,
        session_id: str,
        decision: str,
        severity: ReviewSeverity | Mapping[str, Any] | None = None,
        summary: str = "",
        required_fixes: list[str] | None = None,
        optional_fixes: list[str] | None = None,
        required_reviewers: list[str] | None = None,
        approvers: list[str] | None = None,
        task_executor: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> ReviewResult:
        severity_obj = ReviewSeverity.from_raw(severity)
        result = ReviewResult(
            task_id=str(_get_attr(task, "task_id", "unknown")).strip() or "unknown",
            reviewer=reviewer,
            decision=decision,
            severity=severity_obj,
            summary=summary,
            required_fixes=list(required_fixes or []),
            optional_fixes=list(optional_fixes or []),
            required_reviewers=list(required_reviewers or self.required_reviewers(task)),
            approvers=list(approvers or [reviewer]),
            task_executor=task_executor or str(_get_attr(task, "owner_agent", "")),
        )
        if self.event_log is not None:
            self.event_log.log_review_completed(
                task_id=result.task_id,
                session_id=session_id,
                reviewer=reviewer,
                decision=decision,
                summary=summary,
                metadata={
                    "severity": severity_obj.to_dict(),
                    "required_fixes": list(result.required_fixes),
                    "optional_fixes": list(result.optional_fixes),
                    **dict(metadata or {}),
                },
            )
        return result

    def can_close(self, review_result: ReviewResult) -> bool:
        if review_result.severity.has_blocking():
            return False
        if review_result.required_fixes:
            return False
        if review_result.required_reviewers:
            approvers = set(review_result.approvers)
            if not set(review_result.required_reviewers).issubset(approvers):
                return False
        if review_result.task_executor and review_result.approvers:
            if set(review_result.approvers) == {review_result.task_executor}:
                return False
        if review_result.decision not in {"pass", "pass_with_notes"}:
            return False
        return True

    def is_blocked(self, review_result: ReviewResult) -> bool:
        return not self.can_close(review_result)

    @staticmethod
    def _extract_files_changed(task: Any, changed_paths: list[str]) -> int:
        raw = _get_attr(task, "files_changed", None)
        if raw is None:
            raw = _get_attr(task, "changed_files_count", None)
        if raw is None:
            raw = _get_attr(task, "files_changed_count", None)
        if raw is None and changed_paths:
            return len(changed_paths)
        try:
            return int(raw)
        except Exception:
            return 0

    def _reviewers_for_risk(self, risk_level: str) -> list[str]:
        rule = self.rules.reviewers.get(risk_level, {})
        required = _normalize_list(rule.get("required"))
        optional = _normalize_list(rule.get("optional"))
        return required or optional

    @staticmethod
    def _matches_trigger(
        rule: Mapping[str, Any],
        risk_level: str,
        files_changed: int,
        changed_paths: list[str],
        actions: list[str],
    ) -> bool:
        if "risk_level" in rule:
            return risk_level in _normalize_list(rule.get("risk_level"))
        if "files_changed_gte" in rule:
            try:
                return files_changed >= int(rule.get("files_changed_gte", 0))
            except Exception:
                return False
        if "touches" in rule:
            patterns = _normalize_list(rule.get("touches"))
            return any(
                fnmatch(path, pattern)
                for path in changed_paths
                for pattern in patterns
            )
        if "actions" in rule:
            allowed = set(_normalize_list(rule.get("actions")))
            return bool(allowed.intersection(actions))
        return False

    @staticmethod
    def _describe_trigger(rule: Mapping[str, Any]) -> str:
        if "risk_level" in rule:
            return f"risk_level:{','.join(_normalize_list(rule.get('risk_level')))}"
        if "files_changed_gte" in rule:
            return f"files_changed_gte:{rule.get('files_changed_gte')}"
        if "touches" in rule:
            return f"touches:{','.join(_normalize_list(rule.get('touches')))}"
        if "actions" in rule:
            return f"actions:{','.join(_normalize_list(rule.get('actions')))}"
        return "unknown"

    @staticmethod
    def _describe_reason(rule: Mapping[str, Any], risk_level: str, files_changed: int) -> str:
        if "risk_level" in rule:
            return f"risk_level {risk_level} requires review"
        if "files_changed_gte" in rule:
            return f"files_changed={files_changed} >= {rule.get('files_changed_gte')}"
        if "touches" in rule:
            return "changed paths matched review-sensitive patterns"
        if "actions" in rule:
            return "task action matched review-sensitive operation"
        return "review required"


def load_review_rules(path: str | Path) -> ReviewRules:
    rules_path = Path(path)
    data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ReviewGateError("Review rules document must be a mapping")

    version = str(data.get("version") or "").strip()
    if not version:
        raise ReviewGateError("Review rules document is missing version")

    review_required_when = data.get("review_required_when") or []
    if not isinstance(review_required_when, list):
        raise ReviewGateError("review_required_when must be a list")

    reviewers = data.get("reviewers") or {}
    if not isinstance(reviewers, Mapping):
        raise ReviewGateError("reviewers must be a mapping")

    normalized_rules: list[dict[str, Any]] = []
    for rule in review_required_when:
        if not isinstance(rule, Mapping):
            raise ReviewGateError("Each review rule must be a mapping")
        normalized_rules.append(dict(rule))

    normalized_reviewers: dict[str, dict[str, Any]] = {}
    for risk_level, config in reviewers.items():
        if not isinstance(config, Mapping):
            raise ReviewGateError("Each reviewer config must be a mapping")
        normalized_reviewers[str(risk_level)] = {
            "required": _normalize_list(config.get("required")),
            "optional": _normalize_list(config.get("optional")),
            "requires_human_approval": bool(config.get("requires_human_approval", False)),
        }

    return ReviewRules(
        version=version,
        review_required_when=tuple(normalized_rules),
        reviewers=normalized_reviewers,
        source_path=rules_path,
    )


def load_review_gate(
    path: str | Path,
    *,
    event_log: ManagedAgentEventLog | None = None,
) -> ReviewGate:
    return ReviewGate(rules=load_review_rules(path), event_log=event_log)
