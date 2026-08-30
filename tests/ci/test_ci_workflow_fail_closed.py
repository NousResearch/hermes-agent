"""Regression coverage for the CI orchestrator's fail-closed boundary."""

from __future__ import annotations

import json
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
    )

    assert completed.returncode != 0, completed.stdout
    assert "detect: cancelled" in completed.stdout
