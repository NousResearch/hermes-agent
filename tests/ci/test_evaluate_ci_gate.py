"""Behavioral tests for the CI gate evaluator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "ci" / "evaluate_ci_gate.py"


def _run_gate(tmp_path: Path, needs: dict[str, object]) -> subprocess.CompletedProcess[str]:
    output = tmp_path / "github-output"
    environment = os.environ | {
        "NEEDS": json.dumps(needs),
        "GITHUB_OUTPUT": str(output),
    }
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=_REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    completed.github_output = output.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    return completed


def test_gate_accepts_success_and_skipped_results(tmp_path):
    completed = _run_gate(
        tmp_path,
        {"detect": {"result": "success"}, "desktop": {"result": "skipped"}},
    )

    assert completed.returncode == 0, completed.stdout
    assert 'needs-json={"detect": "success", "desktop": "skipped"}' in completed.github_output


def test_gate_rejects_cancelled_results(tmp_path):
    completed = _run_gate(tmp_path, {"detect": {"result": "cancelled"}})

    assert completed.returncode != 0
    assert "detect: cancelled" in completed.stdout


def test_gate_rejects_missing_or_malformed_job_results(tmp_path):
    completed = _run_gate(tmp_path, {"detect": {}})

    assert completed.returncode != 0
    assert "detect: unknown" in completed.stdout
