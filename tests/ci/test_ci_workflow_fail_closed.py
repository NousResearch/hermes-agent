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
def test_fork_main_push_skips_duplicate_validation(
    workflow_name: str, job_name: str, must_run_after_failed_needs: bool
):
    """A fork merge must not repeat the PR validation that just passed."""
    condition = str(_workflow(workflow_name)["jobs"][job_name].get("if", ""))
    normalized = re.sub(r"\s+", "", condition)
    fork_guard = (
        "github.event_name=='pull_request'||"
        "github.repository=='NousResearch/hermes-agent'"
    )

    assert fork_guard in normalized
    if must_run_after_failed_needs:
        assert normalized.startswith("always()&&(")


def test_required_gate_rejects_cancelled_detect_job(tmp_path):
    """A cancelled classifier must block the required gate, not skip every lane green."""
    steps = _ci_workflow()["jobs"]["all-checks-pass"]["steps"]
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
        input=json.dumps({"detect": {"result": "cancelled"}}),
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    assert completed.returncode != 0, completed.stdout
    assert "detect: cancelled" in completed.stdout
