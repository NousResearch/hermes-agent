"""The canonical shell must run tests and propagate a failing test's status."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.platforms("posix")
def test_shell_runner_executes_tests_and_propagates_failure(tmp_path):
    case = tmp_path / "test_runner_canary.py"
    marker = tmp_path / "executed"
    case.write_text(
        "from pathlib import Path\nimport os\n"
        "def test_canary():\n"
        f"    Path({str(marker)!r}).write_text('executed')\n"
        "    assert os.environ['PATHEXT'] == '.COM;.EXE;.BAT;.CMD'\n"
        "    assert False, 'runner failure propagation canary'\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(root / "scripts/run_tests.sh"), "-j", "1", str(case)],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
        env={**os.environ, "HERMES_PYTHON": sys.executable, "HERMES_TEST_FILE_RETRIES": "0",
             "PATHEXT": ".COM;.EXE;.BAT;.CMD"},
             check=False,
    )
    assert marker.read_text(encoding="utf-8") == "executed", result.stdout + result.stderr
    assert result.returncode != 0, result.stdout + result.stderr
    assert "runner failure propagation canary" in result.stdout + result.stderr


@pytest.mark.platforms("posix")
def test_shell_runner_forwards_the_upgrade_suite_ref_knobs(tmp_path):
    """tests/e2e/core/upgrade reads its start, install and object-store refs inside the runner's
    `env -i`; a dropped one silently runs the default refs instead (#150040)."""
    knobs = {"HERMES_E2E_UPGRADE_BASE": "v0.23.0", "HERMES_E2E_N1_REF": "v0.22.7",
             "HERMES_E2E_INSTALL_REF": "0123abcd", "HERMES_E2E_GIT_OBJECTS": str(tmp_path / "objects"),
             "HERMES_E2E_KEEP_ROOT": "1"}
    case = tmp_path / "test_upgrade_knobs_canary.py"
    case.write_text(
        "import os\n"
        "def test_canary():\n"
        f"    expected = {knobs!r}\n"
        "    assert {k: os.environ.get(k) for k in expected} == expected\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(root / "scripts/run_tests.sh"), "-j", "1", str(case)],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
        env={**os.environ, "HERMES_PYTHON": sys.executable, "HERMES_TEST_FILE_RETRIES": "0", **knobs},
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
