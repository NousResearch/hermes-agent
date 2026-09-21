"""Regression coverage for the deployed system-health script bundle."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def test_staged_runtime_bundle_resolves_board_compat_at_both_call_sites(tmp_path):
    """The no-agent bundle must work without PYTHONPATH or module injection."""
    hermes_home = tmp_path / "hermes-home"
    staged_scripts = hermes_home / "scripts"
    staged_scripts.mkdir(parents=True)

    for filename in ("system_health_daily.py", "_board_compat.py"):
        shutil.copy2(SCRIPTS / filename, staged_scripts / filename)

    harness = staged_scripts / "exercise_system_health.py"
    harness.write_text(
        "import system_health_daily as health\n"
        "assert health.check_kanban() is None\n"
        "assert health.check_wfa_live() is None\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(harness)],
        cwd=staged_scripts,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
