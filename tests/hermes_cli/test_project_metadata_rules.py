from __future__ import annotations

from hermes_cli.project_metadata_rules import (
    ExecutionOutcome,
    IssueMetadataSnapshot,
    ProjectFieldUpdate,
    evaluate_project_metadata,
    updates_for_automated_closure,
)


def test_closed_issue_requires_done_iteration_priority_and_execution_mode():
    snapshot = IssueMetadataSnapshot(
        number=4,
        title="Closed issue missing project metadata",
        state="closed",
        project_item_present=True,
        status="Todo",
        iteration=None,
        priority=None,
        execution_mode=None,
    )

    result = evaluate_project_metadata(snapshot)

    assert result.required_updates == [
        ProjectFieldUpdate("Status", "Done", reason="closed issues must be marked Done on the Project board"),
    ]
    assert result.review_required == [
        "Iteration is required for closed Project issues; infer from closed_at/current workflow before setting.",
        "Priority is required for closed Project issues; use Ryan's priority rubric before setting.",
        "Execution mode is required for closed Project issues; choose automated/assisted/manual from the actual work history.",
    ]


def test_japanese_learning_or_hobby_defaults_to_p3_without_leverage():
    snapshot = IssueMetadataSnapshot(
        number=34,
        title="아이유 마음 녹음하기",
        state="open",
        project_item_present=True,
        status="Todo",
        priority=None,
        body="Japanese/Korean cover practice task with no blocker or recurring-friction rationale.",
    )

    result = evaluate_project_metadata(snapshot)

    assert ProjectFieldUpdate(
        "Priority",
        "P3 Nice-to-have",
        reason="Japanese-learning/hobby items default to P3 unless urgency, leverage, or recurring-friction rationale is present",
    ) in result.required_updates


def test_parent_priority_is_inherited_when_child_priority_is_missing():
    snapshot = IssueMetadataSnapshot(
        number=166,
        title="Create AIE Ch 8 slide deck",
        state="open",
        project_item_present=True,
        status="Todo",
        priority=None,
        parent_priority="P2 Useful",
    )

    result = evaluate_project_metadata(snapshot)

    assert ProjectFieldUpdate(
        "Priority",
        "P2 Useful",
        reason="subissue priority should inherit the parent priority unless explicitly overridden",
    ) in result.required_updates


def test_automated_closure_updates_do_not_touch_deprecated_planning_field():
    updates = updates_for_automated_closure(
        ExecutionOutcome(issue_closed=True, project_owner="ryanleeai", project_number=1)
    )

    assert updates == [
        ProjectFieldUpdate("Status", "Done", reason="closed issue should be marked Done"),
        ProjectFieldUpdate("Execution mode", "automated", reason="Hermes completed the issue end-to-end"),
    ]
    assert all(update.field_name != "Deprecated - Is planned" for update in updates)
