"""Explicit test inputs must never disappear from an apparently passing run."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("input_mode", ["positional", "--paths", "--files"])
@pytest.mark.parametrize("generate_slices", [False, True])
def test_missing_input_rejects_entire_run_before_any_tests_start(
    tmp_path, input_mode, generate_slices
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner = scripts / "run_tests_parallel.py"
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts" / runner.name, runner
    )
    marker = tmp_path / "test-was-run"
    valid = tmp_path / "test_valid.py"
    valid.write_text(
        "from pathlib import Path\n"
        f"def test_passes():\n    Path({str(marker)!r}).touch()\n"
    )
    missing = tmp_path / "test_mistyped.py"
    paths = [str(valid), str(missing)]
    selection = paths if input_mode == "positional" else [input_mode, os.pathsep.join(paths)]
    args = ["--generate-slices", "1"] if generate_slices else ["-j", "1", "-q"]

    result = subprocess.run(
        [sys.executable, str(runner), *selection, *args],
        cwd=tmp_path,
        capture_output=True,
        encoding="utf-8",
        timeout=30,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert str(missing) in result.stderr
    assert "does not exist" in result.stderr
    assert not marker.exists(), "A partial selection must not execute before input validation"
    assert '"slice":' not in result.stdout, "CI must not publish an incomplete test matrix"
