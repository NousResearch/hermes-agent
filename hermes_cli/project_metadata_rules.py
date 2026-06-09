"""Pure Project metadata rules for GitHub issue workflows.

These helpers intentionally avoid network access.  They encode small, testable
contracts for Ryan's Project board metadata so listener/audit code and skills
can share golden cases instead of relying only on prompt discipline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProjectFieldUpdate:
    field_name: str
    value: str
    reason: str


@dataclass(frozen=True)
class IssueMetadataSnapshot:
    number: int
    title: str
    state: str
    project_item_present: bool
    status: str | None = None
    iteration: str | None = None
    priority: str | None = None
    execution_mode: str | None = None
    deprecated_is_planned: str | None = None
    body: str = ""
    parent_priority: str | None = None
    explicit_priority_override: bool = False


@dataclass(frozen=True)
class MetadataRuleEvaluation:
    required_updates: list[ProjectFieldUpdate] = field(default_factory=list)
    review_required: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionOutcome:
    issue_closed: bool
    project_owner: str | None = None
    project_number: int | None = None


def evaluate_project_metadata(snapshot: IssueMetadataSnapshot) -> MetadataRuleEvaluation:
    """Evaluate deterministic Project field expectations for one issue.

    The function returns two classes of findings:
    - ``required_updates`` for safe deterministic field updates;
    - ``review_required`` where the field is required but the correct value
      depends on work history, closed_at/iteration inference, or Ryan's rubric.
    """
    required_updates: list[ProjectFieldUpdate] = []
    review_required: list[str] = []
    violations: list[str] = []

    if not snapshot.project_item_present:
        violations.append("Issue must be present on Ryan's Project board before metadata can be verified.")
        return MetadataRuleEvaluation(required_updates, review_required, violations)

    state = snapshot.state.lower()
    if state == "closed":
        if snapshot.status != "Done":
            required_updates.append(
                ProjectFieldUpdate("Status", "Done", reason="closed issues must be marked Done on the Project board")
            )
        if not snapshot.iteration:
            review_required.append(
                "Iteration is required for closed Project issues; infer from closed_at/current workflow before setting."
            )
        if not snapshot.priority:
            review_required.append("Priority is required for closed Project issues; use Ryan's priority rubric before setting.")
        if not snapshot.execution_mode:
            review_required.append(
                "Execution mode is required for closed Project issues; choose automated/assisted/manual from the actual work history."
            )

    if snapshot.parent_priority and not snapshot.priority and not snapshot.explicit_priority_override:
        required_updates.append(
            ProjectFieldUpdate(
                "Priority",
                snapshot.parent_priority,
                reason="subissue priority should inherit the parent priority unless explicitly overridden",
            )
        )
    elif (
        not snapshot.priority
        and _looks_like_japanese_learning_or_hobby(snapshot)
        and not _has_priority_leverage_signal(snapshot)
    ):
        required_updates.append(
            ProjectFieldUpdate(
                "Priority",
                "P3 Nice-to-have",
                reason="Japanese-learning/hobby items default to P3 unless urgency, leverage, or recurring-friction rationale is present",
            )
        )

    return MetadataRuleEvaluation(required_updates, review_required, violations)


def updates_for_automated_closure(outcome: ExecutionOutcome) -> list[ProjectFieldUpdate]:
    """Return safe Project updates when Hermes closes an issue end-to-end.

    This deliberately does not set or clear Ryan's deprecated planning field.
    Planning intent comes from the daily log, and legacy field migration/cleanup
    should only happen when Ryan explicitly asks for it.
    """
    if not outcome.issue_closed or outcome.project_owner is None or outcome.project_number is None:
        return []
    return [
        ProjectFieldUpdate("Status", "Done", reason="closed issue should be marked Done"),
        ProjectFieldUpdate("Execution mode", "automated", reason="Hermes completed the issue end-to-end"),
    ]


def _looks_like_japanese_learning_or_hobby(snapshot: IssueMetadataSnapshot) -> bool:
    text = f"{snapshot.title}\n{snapshot.body}".lower()
    hobby_keywords = (
        "japanese",
        "日本語",
        "cover",
        "커버",
        "커버곡",
        "녹음",
        "믹싱",
        "노래",
        "아이유",
        "마리골드",
        "첫사랑",
        "twitch",
        "iriam",
    )
    return any(keyword.lower() in text for keyword in hobby_keywords)


def _has_priority_leverage_signal(snapshot: IssueMetadataSnapshot) -> bool:
    text = f"{snapshot.title}\n{snapshot.body}".lower()
    leverage_keywords = (
        "urgent",
        "blocked",
        "blocker",
        "deadline",
        "recurring friction",
        "recurring-friction",
        "high leverage",
        "materially reduces",
        "p1",
        "p2",
        "interview",
        "job",
    )
    return any(_contains_positive_signal(text, keyword) for keyword in leverage_keywords)


def _contains_positive_signal(text: str, keyword: str) -> bool:
    for match in re.finditer(re.escape(keyword), text):
        prefix = text[max(0, match.start() - 24) : match.start()]
        if re.search(r"\b(no|not|without|lacks?)\b[\w\s/-]{0,18}$", prefix):
            continue
        return True
    return False
