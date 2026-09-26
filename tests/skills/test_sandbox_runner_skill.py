"""
Tests for the Sandbox Runner skill (skills/devops/sandbox-runner).

Covers:
- Local isolated sandbox execution
- Execution output capture (stdout, stderr, exit code)
- Timeout enforcement
- Script file execution runner
- Engine status check and CLI subprocess commands
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_SCRIPT = (
    REPO_ROOT
    / "skills"
    / "devops"
    / "sandbox-runner"
    / "scripts"
    / "sandbox.py"
)

# Import module directly
sys.path.insert(0, str(SANDBOX_SCRIPT.parent))
import sandbox


class TestSandboxRunnerCore:
    def test_run_in_local_sandbox_success(self, tmp_path):
        res = sandbox.run_in_local_sandbox("echo 'Hello Sandbox'", mount_dir=tmp_path)
        assert res["exit_code"] == 0
        assert "Hello Sandbox" in res["stdout"]
        assert res["timed_out"] is False
        assert res["engine"] == "local_isolated"

    def test_run_in_local_sandbox_stderr(self, tmp_path):
        res = sandbox.run_in_local_sandbox("python3 -c \"import sys; sys.stderr.write('Test Error\\n'); sys.exit(2)\"", mount_dir=tmp_path)
        assert res["exit_code"] == 2
        assert "Test Error" in res["stderr"]

    def test_run_in_local_sandbox_timeout(self, tmp_path):
        res = sandbox.run_in_local_sandbox("python3 -c \"import time; time.sleep(5)\"", mount_dir=tmp_path, timeout=1)
        assert res["timed_out"] is True
        assert res["exit_code"] == 124
        assert "timed out after 1 seconds" in res["stderr"]

    def test_execute_script_file(self, tmp_path):
        test_script = tmp_path / "calc.py"
        test_script.write_text("print(40 + 2)", encoding="utf-8")

        res = sandbox.execute_script_file(test_script, timeout=10)
        assert res["exit_code"] == 0
        assert "42" in res["stdout"]


class TestSandboxRunnerCLI:
    def test_cli_check_json(self):
        res = subprocess.run(
            [
                sys.executable,
                str(SANDBOX_SCRIPT),
                "check",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(res.stdout)
        assert "engine" in data
        assert "docker_available" in data

    def test_cli_run_json(self):
        res = subprocess.run(
            [
                sys.executable,
                str(SANDBOX_SCRIPT),
                "run",
                "echo 'CLI sandbox test'",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(res.stdout)
        assert data["exit_code"] == 0
        assert "CLI sandbox test" in data["stdout"]
