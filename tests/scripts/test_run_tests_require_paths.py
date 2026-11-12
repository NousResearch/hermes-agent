"""``--require-paths`` refuses a run that names nothing to run.

``just test`` passes it, so ``just test -v`` stops instead of discovering the
whole ``tests/`` tree (about 34k tests), while a named path still runs.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "run_tests_parallel.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "HERMES_TEST_PATHS"}
    return subprocess.run([sys.executable, str(RUNNER), "--require-paths", *args],
                          capture_output=True, text=True, timeout=120, check=False, env=env)


def test_a_lone_pytest_flag_selects_nothing_and_fails():
    result = _run("-v", "--generate-slices", "1")
    assert result.returncode == 2
    assert "name a test path" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("named", [
    ["tests/scripts/test_ruff_runner_cli.py"],
    ["--paths", "tests/scripts/test_ruff_runner_cli.py"],
    ["--files", "tests/scripts/test_ruff_runner_cli.py"],
])
def test_a_named_path_still_selects_its_files(named):
    result = _run(*named, "--generate-slices", "1")
    assert result.returncode == 0, result.stderr
    assert "tests/scripts/test_ruff_runner_cli.py" in result.stdout
