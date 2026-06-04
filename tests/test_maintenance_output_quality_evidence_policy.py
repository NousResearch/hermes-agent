from __future__ import annotations

from pathlib import Path


DOC_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "maintenance"
    / "output-quality-evidence-without-shadow-review.md"
)


def _doc() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def test_policy_keeps_anti_duplication_boundary() -> None:
    text = _doc()

    assert "## Anti-duplication rule" in text
    assert "Watchdogs must not become subjective reviewers" in text
    assert "must not rerun the responsible profile's full test plan" in text
    assert "shadow-approve deployment work" in text


def test_policy_names_required_completion_evidence() -> None:
    text = _doc()

    for required in [
        "## Required responsible completion evidence",
        "Scope and user-visible effect",
        "Source-control evidence",
        "Verification evidence",
        "Activation evidence for runtime/deployment work",
        "Residual risk and judgment",
        "evidence_version",
        "completion_class",
        "source_control",
        "verification",
        "activation",
        "responsible_judgment",
    ]:
        assert required in text


def test_policy_limits_watchdogs_to_objective_checks() -> None:
    text = _doc()

    for required in [
        "## Objective watchdog checks",
        "Evidence schema presence",
        "PR/merge consistency",
        "Local activation consistency",
        "Restart/reload after change",
        "Surface probe",
        "Task-state activation evidence",
        "## Responsible-owned judgment that watchdogs must not duplicate",
        "must not substitute its own judgment",
    ]:
        assert required in text


def test_policy_keeps_global_watchdog_out_of_project_output_review() -> None:
    text = _doc()

    assert "## Fit with the lightweight global watchdog" in text
    assert "global invariants" in text
    assert "must not expand into project output review" in text
    assert "It should not inspect every project task's output quality" in text


def test_policy_uses_canonical_status_terms() -> None:
    text = _doc()

    assert "Self status: WORKING|WAITING|BLOCKED|DORMANT" in text
    assert "Lineage status: WORKING|WAITING|BLOCKED|DORMANT" in text
    assert "`blocked` is only for concrete human intervention" in text
    assert "missing mechanical evidence are `waiting`" in text
