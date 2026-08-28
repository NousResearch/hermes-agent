"""Regression coverage for the CI orchestrator's fail-closed boundary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[2]


def _ci_workflow() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((_REPO / ".github/workflows/ci.yaml").read_text(encoding="utf-8"))


def test_fork_python_suite_uses_twelve_balanced_slices():
    """Fork runners must split the full suite finely enough to avoid long tail jobs."""
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (_REPO / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    )
    slice_expression = workflow["jobs"]["test"]["strategy"]["matrix"]["slice"]

    assert all(f'"{index}/12"' in slice_expression for index in range(1, 13))
    assert '"1/1"' in slice_expression


def test_required_gate_rejects_cancelled_detect_job(tmp_path):
    """A cancelled classifier must block the required gate, not skip every lane green."""
    steps = _ci_workflow()["jobs"]["all-checks-pass"]["steps"]
    evaluate = next(step for step in steps if step.get("name") == "Evaluate job results")

    shell_command = evaluate["run"]
    python_source = shell_command.split('python3 -c "', 1)[1].rsplit('"', 1)[0]
    python_source = python_source.replace(r'\"', '"').replace(
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
