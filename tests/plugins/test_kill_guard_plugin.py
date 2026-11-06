"""Tests for the bundled kill-guard plugin."""
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "kill_guard_plugin", Path(__file__).resolve().parents[2] / "plugins" / "kill-guard" / "__init__.py"
)
kg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kg)

_DANGEROUS = [
    'pkill -f "serve.py" -u me --signal TERM',
    "pkill -f serve.py -9",
    "pkill serve.py extra",
    "pkill -u me",
    f"pkill -f {Path.home()}",
    "pkill -f /Users",
    "sudo killall node -9",
    "FOO=1 pkill -f a -TERM && echo ok",
    "echo hi; pkill -f a -9",
]
_SAFE = [
    "pkill -TERM -u me -f 'share/serve.py'",
    "pkill -f serve.py",
    "killall Dock",
    "echo pkill is dangerous",
    "ls -la",
    "",
]


@pytest.mark.parametrize("cmd", _DANGEROUS)
def test_dangerous_commands_are_flagged(cmd):
    assert kg.check_command(cmd)


@pytest.mark.parametrize("cmd", _SAFE)
def test_safe_commands_pass(cmd):
    assert kg.check_command(cmd) is None


def test_hook_blocks_terminal_and_unwraps_tool_call(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("KILL_GUARD_DISABLE", raising=False)
    bad = {"command": "pkill -f x -9"}
    assert kg._pre_tool_call("terminal", bad)["action"] == "block"
    wrapped = {"calls": [{"name": "mcp__terminal", "arguments": bad}]}
    assert kg._pre_tool_call("tool_call", wrapped)["action"] == "block"
    assert kg._pre_tool_call("terminal", {"command": "pkill -f x"}) is None
    assert kg._pre_tool_call("write_file", bad) is None


def test_disable_switch_and_fail_open(monkeypatch):
    monkeypatch.setenv("KILL_GUARD_DISABLE", "1")
    assert kg._pre_tool_call("terminal", {"command": "pkill -f x -9"}) is None
    monkeypatch.delenv("KILL_GUARD_DISABLE")
    assert kg._pre_tool_call("terminal", None) is None
