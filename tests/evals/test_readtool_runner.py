"""Regression tests for the read-tool evaluation CLI."""

import os
import subprocess
import sys
from pathlib import Path


def test_unsupported_timeout_multiplier_fails_before_provider_setup():
    runner = Path(__file__).parents[2] / "evals" / "readtool" / "runner.py"
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable, str(runner), "--model", "model", "--provider", "provider",
            "--label", "test", "--timeout-mult", "2",
        ],
        capture_output=True, text=True, env=env, check=False,
    )
    assert result.returncode != 0
    assert "--timeout-mult is not supported" in result.stderr
