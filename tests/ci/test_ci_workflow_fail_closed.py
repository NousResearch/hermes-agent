"""Regression coverage for the CI orchestrator's fail-closed boundary."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[2]
def _ci_workflow() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(
        (_REPO / ".github/workflows/ci.yaml").read_text(encoding="utf-8")
    )


def _workflow(name: str) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(
        (_REPO / ".github/workflows" / name).read_text(encoding="utf-8")
    )
def test_fork_python_suite_uses_public_runner_without_obsolete_sharding():
    """Fork CI stays on a public runner with bounded concurrency and no stale shard watchdog."""
    workflow = _workflow("tests.yml")
    job = workflow["jobs"]["test"]
    steps = job["steps"]
    run_tests = next(step for step in steps if step.get("name") == "Run tests")

    assert job["runs-on"] == "ubuntu-latest"
    assert int(job["timeout-minutes"]) == 60
    assert "strategy" not in job
    assert run_tests["run"].strip().endswith("scripts/run_tests.sh")
    assert "timebox_process.py" not in run_tests["run"]
    assert str(run_tests["env"]["HERMES_TEST_WORKERS"]) == "4"


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "must_run_after_failed_needs"),
    [
        ("ci.yaml", "detect", False),
        ("ci.yaml", "osv-scanner", False),
        ("ci.yaml", "all-checks-pass", True),
        ("ci.yaml", "ci-timings", True),
        ("nix.yml", "detect", False),
    ],
)
def test_fork_validation_is_opt_in_but_prs_and_upstream_pushes_still_run(
    workflow_name: str, job_name: str, must_run_after_failed_needs: bool
):
    """Allow explicit fork validation without duplicating every merged PR run."""
    condition = str(_workflow(workflow_name)["jobs"][job_name].get("if", ""))
    normalized = re.sub(r"\s+", "", condition)
    fork_guard = (
        "github.event_name=='pull_request'||"
        "github.event_name=='workflow_dispatch'||"
        "github.repository=='NousResearch/hermes-agent'"
    )

    expected = f"always()&&({fork_guard})" if must_run_after_failed_needs else fork_guard
    assert normalized == expected


@pytest.mark.parametrize("job_name, result, expected_exit", [
    ("detect", "cancelled", 1),
    ("infographic-check", "failure", 1),
    ("infographic-check", "success", 0),
])
def test_required_gate_evaluates_validation_results(tmp_path, job_name, result, expected_exit):
    """Both dependency wiring and runtime evaluation must enforce failed checks."""
    gate = _ci_workflow()["jobs"]["all-checks-pass"]
    assert job_name in gate["needs"]
    steps = gate["steps"]
    evaluate = next(
        step for step in steps if step.get("name") == "Evaluate job results"
    )

    shell_command = evaluate["run"]
    python_source = shell_command.split('python3 -c "', 1)[1].rsplit('"', 1)[0]
    python_source = python_source.replace(r"\"", '"').replace(
        "'$GITHUB_OUTPUT'", repr(str(tmp_path / "github-output"))
    )
    completed = subprocess.run(
        [sys.executable, "-c", python_source],
        cwd=_REPO,
        input=json.dumps({job_name: {"result": result}}),
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    assert completed.returncode == expected_exit, completed.stdout + completed.stderr
    assert f"{job_name}: {result}" in completed.stdout
