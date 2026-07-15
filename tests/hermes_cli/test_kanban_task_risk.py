from __future__ import annotations

import pytest

from hermes_cli.kanban_task_risk import assess_task_risk


@pytest.mark.parametrize(
    ("title", "body", "expected_level", "expected_dimensions"),
    [
        (
            "Add parser timeout",
            "Implement one timeout option in hermes_cli/parser.py and add a focused test.",
            "low",
            (),
        ),
        (
            "Retained SSE foundation",
            """
            Investigate and define retained cursor semantics, then implement the SSE server endpoint.
            Wire worker journal persistence, generate the OpenAPI contract, and update the dashboard client.
            Acceptance requires restart and tenant-isolation tests, a PR, CI remediation, merge, deployment,
            and runtime validation.
            """,
            "high",
            ("unknown_design", "subsystem_breadth", "delivery_lifecycle"),
        ),
        (
            "Harden Kanban failure paths",
            """
            Diagnose and fix stale dispatcher reclaims, git worktree validation, completion evidence policy,
            and worker bootstrap behavior across hermes_cli/, tools/, plugins/, and gateway/.
            Required:
            - reproduce each bug class
            - implement reclaim protection
            - implement worktree validation
            - implement completion evidence checks
            - implement worker bootstrap normalization
            - add dispatcher tests
            - add tool tests
            - add integration tests
            """,
            "high",
            ("unknown_design", "subsystem_breadth", "acceptance_density"),
        ),
        (
            "Repair live Kanban data",
            """
            Investigate corrupted rows, design the repair, and implement audited reconstruction.
            Back up the live database, apply the migration with production credentials, and manually validate
            the dashboard against the external deployment before rollback is disabled.
            """,
            "high",
            ("unknown_design", "data_ops_risk", "external_gate"),
        ),
    ],
)
def test_assessment_classifies_independent_dimensions(
    title: str,
    body: str,
    expected_level: str,
    expected_dimensions: tuple[str, ...],
):
    assessment = assess_task_risk(
        title=title,
        body=body,
        goal_mode=False,
        max_runtime_seconds=None,
    )

    assert assessment.level == expected_level
    assert set(expected_dimensions) <= set(assessment.dimensions)


def test_long_single_purpose_body_and_delivery_words_do_not_force_high():
    body = "\n".join(
        ["Implement the focused parser fix in one module."]
        + ["Document the exact input and expected output." for _ in range(30)]
        + ["Open a PR, run CI, and merge it."]
    )

    assessment = assess_task_risk(
        title="Focused parser fix",
        body=body,
        goal_mode=False,
        max_runtime_seconds=None,
    )

    assert assessment.level == "low"
    assert assessment.dimensions == ("delivery_lifecycle",)


def test_unrelated_bullet_count_does_not_create_acceptance_density():
    body = "\n".join(
        ["## Background"]
        + [f"- Context note {index}" for index in range(10)]
        + ["## Deliverable", "Implement one parser fix."]
    )

    assessment = assess_task_risk(
        title="Focused parser fix",
        body=body,
        goal_mode=False,
        max_runtime_seconds=None,
    )

    assert "acceptance_density" not in assessment.dimensions


def test_goal_mode_does_not_reduce_high_risk():
    body = """
    Investigate and design the API contract, then implement server, database, CLI, and dashboard changes.
    Repair and migrate live data with credentials and manual device validation.
    """

    normal = assess_task_risk(
        title="Broad task", body=body, goal_mode=False, max_runtime_seconds=None
    )
    goal = assess_task_risk(
        title="Broad task", body=body, goal_mode=True, max_runtime_seconds=None
    )

    assert normal.level == "high"
    assert goal.level == normal.level
    assert goal.dimensions == normal.dimensions


def test_long_runtime_requires_checkpoint_without_raising_risk():
    assessment = assess_task_risk(
        title="Focused parser fix",
        body="Implement one parser fix and its focused regression test.",
        goal_mode=False,
        max_runtime_seconds=1800,
    )

    assert assessment.level == "low"
    assert assessment.checkpoint_required is True
    assert assessment.dimensions == ()


def test_serialization_is_deterministic_bounded_and_redacted():
    secret_body = "Investigate then implement API, database, CLI, and dashboard changes. SECRET-EXCERPT"
    first = assess_task_risk(
        title="Broad change",
        body=secret_body,
        goal_mode=False,
        max_runtime_seconds=None,
    )
    second = assess_task_risk(
        title="Broad change",
        body=secret_body,
        goal_mode=False,
        max_runtime_seconds=None,
    )

    assert first.as_dict() == second.as_dict()
    assert first.version == 1
    serialized = repr(first.as_dict())
    assert "SECRET-EXCERPT" not in serialized
    assert len(first.evidence) <= 8
    assert all(len(item) <= 120 for item in first.evidence)


def test_advisory_message_confirms_creation_and_is_non_blocking():
    assessment = assess_task_risk(
        title="Broad change",
        body="Investigate then implement API, database, CLI, and dashboard changes.",
        goal_mode=False,
        max_runtime_seconds=None,
    )

    assert assessment.level == "medium"
    message = assessment.message()
    assert "created" in message.lower()
    assert "non-blocking" in message.lower()
    assert "unknown_design" in message
    assert "subsystem_breadth" in message
