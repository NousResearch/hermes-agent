"""The canonical shell must run tests and propagate a failing test's status."""
import json
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
    )
    assert marker.read_text(encoding="utf-8") == "executed", result.stdout + result.stderr
    assert result.returncode != 0, result.stdout + result.stderr
    assert "runner failure propagation canary" in result.stdout + result.stderr


@pytest.mark.platforms("posix")
def test_shell_runner_forwards_docker_settings_without_credentials(tmp_path):
    case = tmp_path / "test_docker_environment.py"
    marker = tmp_path / "docker-environment.json"
    docker_settings = {
        "DOCKER_HOST": "unix:///private/tmp/runner-docker.sock",
        "DOCKER_CONTEXT": "local-test-context",
        "DOCKER_CONFIG": str(tmp_path / "docker-config"),
        "HERMES_TEST_MATRIX_GATEWAY_IMAGE": "hermes-matrix-live:ci",
        "HERMES_E2E_STRICT_ACCEPTANCE": "1",
    }
    case.write_text(
        "import json, os\nfrom pathlib import Path\n"
        "def test_docker_environment():\n"
        f"    Path({str(marker)!r}).write_text(json.dumps("
        "{key: value for key, value in os.environ.items()"
        " if key.startswith('DOCKER_') or key in {'HERMES_TEST_MATRIX_GATEWAY_IMAGE', 'HERMES_E2E_STRICT_ACCEPTANCE'}}))\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(root / "scripts/run_tests.sh"), "-j", "1", str(case)],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
        env={**os.environ, **docker_settings, "HERMES_PYTHON": sys.executable,
             "HERMES_TEST_FILE_RETRIES": "0", "DOCKER_AUTH_CONFIG": "private-registry-token",
             "DOCKER_CERT_PATH": str(tmp_path / "docker-certificates"), "DOCKER_TLS_VERIFY": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(marker.read_text(encoding="utf-8")) == docker_settings
