"""Integration tests for the windows-sandbox terminal backend.

Run with:
    TERMINAL_ENV=windows-sandbox pytest tests/integration/test_windows_sandbox_terminal.py -v
"""

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

if platform.system() != "Windows":
    pytest.skip("windows-sandbox integration tests require Windows", allow_module_level=True)

parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

spec = importlib.util.spec_from_file_location(
    "terminal_tool", parent_dir / "tools" / "terminal_tool.py"
)
terminal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(terminal_module)

terminal_tool = terminal_module.terminal_tool
check_terminal_requirements = terminal_module.check_terminal_requirements
cleanup_vm = terminal_module.cleanup_vm

from hermes_cli.config import get_windows_sandbox_codex_home
from tools.environments import windows_sandbox


@pytest.fixture(autouse=True)
def _force_windows_sandbox(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "windows-sandbox")
    monkeypatch.setenv("TERMINAL_WINDOWS_SANDBOX_SETUP", "explicit")
    monkeypatch.delenv("TERMINAL_WINDOWS_SANDBOX_BIN_DIR", raising=False)


@pytest.fixture()
def task_id(request):
    tid = f"windows_sandbox_test_{request.node.name}"
    yield tid
    cleanup_vm(tid)


@pytest.fixture(scope="module", autouse=True)
def _ensure_windows_sandbox_ready():
    os.environ["TERMINAL_ENV"] = "windows-sandbox"
    os.environ["TERMINAL_WINDOWS_SANDBOX_SETUP"] = "explicit"
    os.environ.pop("TERMINAL_WINDOWS_SANDBOX_BIN_DIR", None)

    if not check_terminal_requirements():
        pytest.skip("windows-sandbox requirements are not met on this machine")

    setup_result = windows_sandbox.run_windows_sandbox_setup(
        cwd=str(parent_dir),
        codex_home=str(get_windows_sandbox_codex_home()),
    )
    if setup_result.get("error"):
        pytest.skip(f"windows-sandbox setup did not complete: {setup_result['error']}")

    status = windows_sandbox.get_windows_sandbox_status(
        cwd=str(parent_dir),
        codex_home=str(get_windows_sandbox_codex_home()),
    )
    diagnostics = status.get("diagnostics", {})
    if status.get("error") or not diagnostics.get("setup_complete"):
        pytest.skip(f"windows-sandbox status not ready: {status}")


def _run(command: str, task_id: str, **kwargs):
    result = terminal_tool(command, task_id=task_id, workdir=str(parent_dir), **kwargs)
    return json.loads(result)


class TestWindowsSandboxBasic:
    def test_powershell_echo(self, task_id):
        result = _run("Write-Output 'Hello from Windows Sandbox!'", task_id)
        assert result["exit_code"] == 0
        assert "Hello from Windows Sandbox!" in result["output"]

    def test_cmd_echo(self, task_id):
        result = _run("cmd /c echo Hello from Windows Sandbox CMD", task_id)
        assert result["exit_code"] == 0
        assert "Hello from Windows Sandbox CMD" in result["output"]


class TestWindowsSandboxFilesystem:
    def test_workspace_write_and_read(self, task_id):
        marker = f"windows-sandbox-{task_id}"
        relative_path = f".windows-sandbox-{task_id}.txt"
        command = (
            f"Set-Content -Path '{relative_path}' -Value '{marker}'; "
            f"Get-Content -Path '{relative_path}'; "
            f"Remove-Item -Path '{relative_path}' -Force"
        )
        result = _run(command, task_id)
        assert result["exit_code"] == 0
        assert marker in result["output"]


class TestWindowsSandboxUnsupported:
    def test_background_rejected(self):
        result = json.loads(terminal_tool("Write-Output hi", background=True))
        assert result["status"] == "error"
        assert "does not support background execution" in result["error"]

    def test_pty_rejected(self):
        result = json.loads(terminal_tool("Write-Output hi", pty=True))
        assert result["status"] == "error"
        assert "does not support PTY mode" in result["error"]


class TestWindowsSandboxPolicies:
    def test_read_only_blocks_workspace_write(self, task_id, monkeypatch):
        relative_path = f".windows-sandbox-readonly-{task_id}.txt"
        monkeypatch.setenv("TERMINAL_WINDOWS_SANDBOX_MODE", "read-only")
        monkeypatch.setenv("TERMINAL_WINDOWS_SANDBOX_WRITABLE_ROOTS", "[]")

        result = _run(
            f"Set-Content -Path '{relative_path}' -Value 'read-only-should-fail'",
            task_id,
        )

        assert result["exit_code"] != 0
        assert "denied" in result["output"].lower()
        assert not (parent_dir / relative_path).exists()
