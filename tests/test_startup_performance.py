"""Startup performance regression guard.

Runs as part of the regular test suite. If import model_tools or
import hermes_cli.main exceeds the thresholds below the test fails,
alerting the team before a slow-import regression ships.

Thresholds are intentionally generous (2x the values observed on a
2026 mid-range Windows dev machine) so the test doesn't flap on CI
runners with variable I/O. Tighten them as imports get leaner.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_IMPORT_MODEL_TOOLS_THRESHOLD_MS = 2500.0
_IMPORT_CLI_MAIN_THRESHOLD_MS = 800.0


def _measure_import(module_name: str) -> float:
    env = dict(os.environ)
    env.pop("HERMES_NO_TOOL_CACHE", None)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["HERMES_HOME"] = tempfile.mkdtemp(prefix="hermes-bench-ci-")
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=30, check=False,
    )
    return (time.perf_counter() - started) * 1000.0


@pytest.mark.skipif(
    os.environ.get("CI") != "true" and os.environ.get("HERMES_BENCH_CI") != "1",
    reason="Startup benchmark only runs in CI or with HERMES_BENCH_CI=1",
)
class TestStartupPerformance:
    def test_import_model_tools_within_threshold(self):
        ms = _measure_import("model_tools")
        assert ms < _IMPORT_MODEL_TOOLS_THRESHOLD_MS, (
            f"import model_tools took {ms:.0f}ms (threshold: {_IMPORT_MODEL_TOOLS_THRESHOLD_MS:.0f}ms)")

    def test_import_cli_main_within_threshold(self):
        ms = _measure_import("hermes_cli.main")
        assert ms < _IMPORT_CLI_MAIN_THRESHOLD_MS, (
            f"import hermes_cli.main took {ms:.0f}ms (threshold: {_IMPORT_CLI_MAIN_THRESHOLD_MS:.0f}ms)")
