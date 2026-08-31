"""Regression coverage for the CI orchestrator's fail-closed boundary."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[2]
_MINIMUM_FORK_JOB_HEADROOM_SECONDS = 4 * 60


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


def test_detect_timeout_has_checkout_headroom():
    """A slow hosted-runner fetch must not cancel classification after one minute."""
    timeout_minutes = _ci_workflow()["jobs"]["detect"]["timeout-minutes"]

    assert timeout_minutes >= 5


def test_fork_python_suite_uses_twelve_balanced_slices():
    """Fork runners must split the full suite finely enough to avoid long tail jobs."""
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (_REPO / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    )
    slice_expression = workflow["jobs"]["test"]["strategy"]["matrix"]["slice"]

    assert all(f'"{index}/12"' in slice_expression for index in range(1, 13))
    assert '"1/1"' in slice_expression


def test_fork_python_suite_has_a_server_enforced_timeout():
    """GitHub must stop a lost fork runner without relying on its shell."""
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (_REPO / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    )
    timeout_expression = workflow["jobs"]["test"]["timeout-minutes"]
    match = re.fullmatch(
        r"\$\{\{\s*github\.repository\s*==\s*'([^']+)'\s*"
        r"&&\s*(\d+)\s*\|\|\s*(\d+)\s*\}\}",
        str(timeout_expression),
    )

    assert match is not None
    upstream_repository, upstream_timeout, fork_timeout = match.groups()
    assert upstream_repository == "NousResearch/hermes-agent"
    assert int(upstream_timeout) == 60
    assert int(fork_timeout) <= 30


def test_fork_python_suite_has_a_process_tree_watchdog():
    """Fork test descendants must be terminated with their parent process."""
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (_REPO / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["test"]["steps"]
    run_tests = next(step for step in steps if step.get("name") == "Run tests")
    command = shlex.split(re.sub(r"\\\s*\n", " ", run_tests["run"]))
    timeout_index = command.index("--timeout-seconds")
    grace_index = command.index("--grace-seconds")
    watchdog_timeout = int(command[timeout_index + 1])
    shutdown_grace = int(command[grace_index + 1])
    timeout_expression = workflow["jobs"]["test"]["timeout-minutes"]
    fork_timeout = int(re.findall(r"\d+", str(timeout_expression))[-1]) * 60

    assert command[timeout_index - 2 : timeout_index] == [
        "python",
        "scripts/ci/timebox_process.py",
    ]
    assert watchdog_timeout > 0
    assert 0 <= shutdown_grace <= watchdog_timeout
    assert (
        fork_timeout - watchdog_timeout - shutdown_grace
        >= _MINIMUM_FORK_JOB_HEADROOM_SECONDS
    )
    expected_slice_command = [
        "scripts/run_tests.sh",
        "--slice",
        "${{ matrix.slice }}",
    ]
    assert any(
        command[index : index + len(expected_slice_command)] == expected_slice_command
        for index in range(len(command))
    )
    assert "HERMES_TEST_SLICE" not in run_tests.get("env", {})


def test_contributor_attribution_is_upstream_only():
    """Fork syncs must not republish upstream contributor email mappings."""
    condition = str(_ci_workflow()["jobs"]["contributor-check"]["if"])
    normalized = re.sub(r"\s+", "", condition)

    assert "needs.detect.outputs.python=='true'" in normalized
    assert "github.repository=='NousResearch/hermes-agent'" in normalized


@pytest.mark.parametrize(
    ("workflow_name", "job_name"),
    [
        ("js-tests.yml", "check"),
        ("rust-tests.yml", "bootstrap-installer"),
    ],
)
def test_fork_heavy_linux_jobs_use_standard_runners(
    workflow_name: str, job_name: str
):
    """Fork jobs must not wait for runner labels owned by the upstream repo."""
    runner = str(_workflow(workflow_name)["jobs"][job_name]["runs-on"])
    normalized = re.sub(r"\s+", "", runner)

    assert normalized == (
        "${{github.repository=='NousResearch/hermes-agent'"
        "&&'ubuntu-latest-32-core'||'ubuntu-latest'}}"
    )


def test_fork_os_jobs_use_standard_runners():
    """OS lanes must retain coverage when upstream-only labels are unavailable."""
    job = _workflow("tests-os.yml")["jobs"]["os-tests"]
    normalized = re.sub(r"\s+", "", str(job["runs-on"]))
    rows = {row["name"]: row for row in job["strategy"]["matrix"]["include"]}

    assert normalized == (
        "${{github.repository=='NousResearch/hermes-agent'"
        "&&matrix.runner||matrix.fork_runner}}"
    )
    assert rows["macOS-only tests"]["fork_runner"] == "macos-latest"
    assert rows["Windows-only tests"]["runner"] == "windows-latest-32-core"
    assert rows["Windows-only tests"]["fork_runner"] == "windows-latest"


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
