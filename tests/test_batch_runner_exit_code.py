"""Regression tests for batch_runner process exit codes.

Python Fire serializes the return value of the wrapped function but does not
use it as the process exit code.  Before the fix, all of ``main``'s error paths
returned ``0`` because a bare ``return`` or ``return 1`` was treated as the
function result, not a non-zero exit status.
"""

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent
PYTHON = sys.executable


def _run(*args):
    return subprocess.run(
        [PYTHON, "batch_runner.py", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def test_missing_dataset_file_exits_nonzero():
    result = _run("--run_name=test")
    assert result.returncode == 1
    assert "--dataset_file is required" in result.stdout


def test_invalid_batch_size_exits_nonzero():
    result = _run("--dataset_file=/tmp/data.jsonl", "--batch_size=-1", "--run_name=test")
    assert result.returncode == 1
    assert "--batch_size must be a positive integer" in result.stdout


def test_invalid_reasoning_effort_exits_nonzero():
    result = _run(
        "--dataset_file=/tmp/data.jsonl",
        "--batch_size=1",
        "--run_name=test",
        "--reasoning_effort=invalid",
    )
    assert result.returncode == 1
    assert "--reasoning_effort must be one of" in result.stdout


def test_invalid_prefill_messages_file_exits_nonzero(tmp_path):
    bad_prefill = tmp_path / "prefill.json"
    bad_prefill.write_text("not json", encoding="utf-8")
    result = _run(
        "--dataset_file=/tmp/data.jsonl",
        "--batch_size=1",
        "--run_name=test",
        f"--prefill_messages_file={bad_prefill}",
    )
    assert result.returncode == 1
    assert "Error loading prefill messages" in result.stdout


