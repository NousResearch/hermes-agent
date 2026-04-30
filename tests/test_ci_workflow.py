"""Regression tests for CI workflow invariants."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_tests_workflow() -> dict:
    """Return the parsed GitHub Actions tests workflow."""
    workflow_path = ROOT / ".github" / "workflows" / "tests.yml"
    with workflow_path.open(encoding="utf-8") as workflow_file:
        return yaml.safe_load(workflow_file)


def test_ci_test_job_uses_canonical_test_runner() -> None:
    """Lock CI to the same hermetic test runner documented for developers."""
    workflow = _load_tests_workflow()
    steps = workflow["jobs"]["test"]["steps"]
    run_tests_step = next(step for step in steps if step["name"] == "Run tests")
    run_command = run_tests_step["run"]

    assert "scripts/run_tests.sh" in run_command
    assert "python -m pytest" not in run_command
    assert "-n auto" not in run_command


def test_canonical_test_runner_raises_low_nofile_limit() -> None:
    """Prevent macOS shells from exhausting fd 256 during xdist runs."""
    runner = (ROOT / "scripts" / "run_tests.sh").read_text(encoding="utf-8")

    assert "HERMES_TEST_NOFILE_LIMIT" in runner
    assert "ulimit -Sn" in runner


def test_pr_template_points_contributors_to_canonical_test_runner() -> None:
    """Keep contributor checklists aligned with CI's test entry point."""
    template = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")

    assert "scripts/run_tests.sh" in template
    assert "pytest tests/ -q" not in template


def test_contributing_guide_points_to_canonical_test_runner() -> None:
    """Keep contributor docs aligned with CI's test entry point."""
    guide = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "scripts/run_tests.sh" in guide
    assert "pytest tests/ -v" not in guide
