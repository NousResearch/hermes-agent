"""Native Windows behavior contract for scripts/run_tests.ps1."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.windows_only
def test_powershell_runner_delegates_with_hermetic_environment(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    probe = tmp_path / "test_powershell_runner_probe.py"
    probe.write_text(
        """import os


def test_runner_environment():
    assert os.environ[\"TZ\"] == \"UTC\"
    assert os.environ[\"PYTHONHASHSEED\"] == \"0\"
    assert os.environ[\"PYTHONUTF8\"] == \"1\"
    assert os.environ[\"HERMES_TEST_WORKERS\"] == \"1\"
    assert \"OPENAI_API_KEY\" not in os.environ
    assert \"UNRELATED_RUNNER_PROBE\" not in os.environ
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "must-not-reach-pytest"
    env["UNRELATED_RUNNER_PROBE"] = "must-not-reach-pytest"
    env["HERMES_TEST_WORKERS"] = "1"
    env["HERMES_PYTHON"] = sys.executable

    proc = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(repo_root / "scripts" / "run_tests.ps1"),
            str(probe),
            "-j",
            "1",
            "--file-retries",
            "0",
            "-q",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )

    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "1 tests passed" in output
