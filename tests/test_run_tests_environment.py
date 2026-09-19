"""Behavioral tests for the canonical shell test runner environment."""

import os
import subprocess
import sys
from pathlib import Path


def test_runner_disables_lazy_installs_before_collection(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    probe = tmp_path / "test_collection_environment.py"
    probe.write_text(
        "import os\n"
        "assert os.environ.get('HERMES_DISABLE_LAZY_INSTALLS') == '1'\n\n"
        "def test_probe():\n"
        "    pass\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("HERMES_DISABLE_LAZY_INSTALLS", None)
    env["HERMES_PYTHON"] = sys.executable
    env["HERMES_TEST_FILE_RETRIES"] = "0"
    env["HERMES_TEST_WORKERS"] = "1"

    result = subprocess.run(
        [repo_root / "scripts" / "run_tests.sh", probe, "-q"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
