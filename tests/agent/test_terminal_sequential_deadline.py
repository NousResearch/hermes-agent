"""Terminal owns its bounded foreground wait unless a sequential cap is explicit."""

import json
import shlex
import sys
import threading
import time

import pytest

from agent import tool_executor
from hermes_cli.config import load_config_readonly
from tests.agent.test_sequential_tool_timeout import _make_agent


@pytest.mark.parametrize("concurrent", [None, 1])
def test_terminal_wait_uses_its_own_deadline(tmp_path, monkeypatch, concurrent):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = "terminal:\n  backend: local\n  timeout: 180\n"
    if concurrent is not None:
        config += f"timeouts:\n  tools:\n    concurrent_batch: {concurrent}\n"
    (home / "config.yaml").write_text(config)
    assert load_config_readonly()["terminal"]["timeout"] == 180
    assert tool_executor._resolve_sequential_tool_timeout("terminal") is None
    assert tool_executor._resolve_sequential_tool_timeout() == (concurrent or 420)

    agent = _make_agent(home)
    from tools.terminal_tool import terminal_tool
    from tools.terminal_tool_lifecycle import cleanup_vm
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("TERMINAL_TIMEOUT", "180")
    try:
        result = tool_executor._run_sequential_tool_execution_middleware(
            agent, function_name="terminal", function_args={"command": shlex.quote(sys.executable) + " -c \"import time; time.sleep(2); print(\'deadline-proof\')\"", "timeout": 600},
            effective_task_id="deadline-proof", tool_call_id="terminal-proof",
            execute=lambda args: terminal_tool(**args, task_id="deadline-proof"),
        )
        payload = json.loads(result.result)
        assert payload["exit_code"] == 0
        assert "deadline-proof" in payload["output"]
        timed = tool_executor._run_sequential_tool_execution_middleware(
            agent, function_name="terminal", function_args={"command": shlex.quote(sys.executable) + " -c \"import time; print('partial-output', flush=True); time.sleep(5)\"", "timeout": 1},
            effective_task_id="deadline-proof", tool_call_id="terminal-timeout-proof",
            execute=lambda args: terminal_tool(**args, task_id="deadline-proof"),
        )
        payload = json.loads(timed.result)
        assert payload["exit_code"] == 124
        assert "partial-output" in payload["output"]
        marker = tmp_path / "started"

        def stop_running_command():
            deadline = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            agent.interrupt()

        stopper = threading.Thread(target=stop_running_command, daemon=True)
        stopper.start()
        command = shlex.quote(sys.executable) + " -c " + shlex.quote(
            f"from pathlib import Path; import time; Path({str(marker)!r}).touch(); time.sleep(60)"
        )
        stopped = tool_executor._run_sequential_tool_execution_middleware(
            agent, function_name="terminal", function_args={"command": command, "timeout": 600},
            effective_task_id="deadline-proof", tool_call_id="terminal-stop-proof",
            execute=lambda args: terminal_tool(**args, task_id="deadline-proof"),
        )
        stopper.join(10)
        assert marker.exists()
        assert json.loads(stopped.result)["exit_code"] == 130
        agent.clear_interrupt()
        from tools.tool_search import resolve_underlying_call
        assert resolve_underlying_call({"name": "terminal", "arguments": {"command": "true"}})[2] is not None
    finally:
        cleanup_vm("deadline-proof")


def test_explicit_sequential_cap_still_bounds_terminal(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = home / "config.yaml"
    path.write_text("timeouts:\n  tools:\n    concurrent_batch: 300\n    sequential_call: 90\n")
    assert tool_executor._resolve_sequential_tool_timeout("terminal") == 90
    assert tool_executor._resolve_sequential_tool_timeout() == 90
    path.write_text("timeouts:\n  tools:\n    concurrent_batch: 300\n    sequential_call: 0\n")
    assert tool_executor._resolve_sequential_tool_timeout("terminal") is None
    assert tool_executor._resolve_concurrent_tool_timeout() == 300
