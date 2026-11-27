"""The opt-in contract accounting judges only what a run selected: a partial run is not a missing suite, and xdist is refused."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _pytest(tmp_path: Path, *args: str, **env: str) -> tuple[subprocess.CompletedProcess, Path]:
    artifact = tmp_path / "accounting.json"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=300, check=False,
        env={**os.environ, "HERMES_CONTRACT_ACCOUNTING": str(artifact), **env},
    )
    return result, artifact


def test_a_partial_run_with_accounting_passes_when_its_own_cases_pass(tmp_path):
    result, artifact = _pytest(tmp_path, "tests/contracts/core/test_spine_smoke.py")
    # A run that selected no authority suite has not omitted one: no zero_exercised for capabilities it never collected.
    assert result.returncode == 0, result.stdout[-2000:]
    violations = json.loads(artifact.read_text())["authority"]["violations"]
    assert not [v for v in violations if v["kind"] == "zero_exercised"]


def test_a_run_of_an_authority_suite_is_judged_on_that_suite(tmp_path):
    result, artifact = _pytest(tmp_path, "tests/contracts/authority/test_runtime_contract.py")
    # The selected suite's capability is exercised and the other two capabilities were never selected: no violation either way.
    assert result.returncode == 0, result.stdout[-2000:]
    payload = json.loads(artifact.read_text())["authority"]
    assert payload["violations"] == []
    assert payload["capabilities"]["in_memory"]["authority_fences"]["passed"] > 0


def test_accounting_refuses_an_xdist_worker(tmp_path):
    result, artifact = _pytest(tmp_path, "tests/contracts/core/test_spine_smoke.py", PYTEST_XDIST_WORKER="gw0")
    # Each worker would reconcile the whole collection against its own reports and race the others on one artifact.
    assert result.returncode != 0
    assert "xdist" in (result.stdout + result.stderr)
    assert not artifact.exists()
