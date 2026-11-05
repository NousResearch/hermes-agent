"""SSH command composition must never run after a failed directory change."""
import json
import subprocess
from unittest.mock import patch

import pytest
import shlex

from plugins.remote_run import _build_command, remote_run_handler, register, MAX_RESULT_SIZE_CHARS


@pytest.mark.platforms("posix")
def test_workdir_failure_prevents_command_even_with_environment(tmp_path):
    marker = tmp_path / "executed"
    command = _build_command(f"touch {shlex.quote(str(marker))}", str(tmp_path / "missing"), {"VALUE": "hello"}, False)
    result = subprocess.run(["sh", "-c", command], capture_output=True, check=False)
    assert result.returncode != 0
    assert not marker.exists()
    command = _build_command('printf "%s" "$VALUE" # trailing comment', str(tmp_path), {"VALUE": "a;$(false)'b"}, False)
    result = subprocess.run(["sh", "-c", command], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert result.stdout == "a;$(false)'b"


def test_plugin_registration_and_guard_before_transport(monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.registry import registry
    manager = PluginManager()
    register(PluginContext(PluginManifest(name="remote-run"), manager))
    try:
        entry = registry.get_entry("remote_run", scope=manager.scope_key)
        assert entry.max_result_size_chars == MAX_RESULT_SIZE_CHARS
        with patch("tools.terminal_tool.check_command_guards", return_value={"approved": False, "message": "denied"}):
            result = json.loads(entry.handler({"host": "host.example", "command": "rm -rf /"}))
        assert result["stderr"] == "denied"
    finally:
        registry.deregister("remote_run", scope=manager.scope_key)


@pytest.mark.parametrize("args", [{}, {"host": "host", "command": "true", "port": True}, {"host": "host", "command": "true", "timeout": 0}])
def test_invalid_input_rejected_before_importing_transport(args):
    assert json.loads(remote_run_handler(args))["exit_code"] == 2
