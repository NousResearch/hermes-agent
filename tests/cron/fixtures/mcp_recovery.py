"""Subprocess-only real stdio replay; no model calls or mocked MCP state.

Usage: python mcp_recovery.py SOURCE {recover,delayed-list} FRESH_HOME
The retry ladder is intentionally not shortened (~30s to park).
"""

import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

repo, case, raw_home = sys.argv[1:]
repo = Path(repo).resolve()
home = Path(raw_home).resolve()
home.mkdir(parents=True, exist_ok=False)
os.environ["HERMES_HOME"] = str(home)
os.environ["HERMES_TOOL_PROGRESS"] = "0"
os.environ["HERMES_ENABLE_PROJECT_PLUGINS"] = "0"
sys.path.insert(0, str(repo))

import hermes_yaml as yaml

from tests.waiting import wait_until
from tools import mcp_tool as core
from tools.mcp_tool_discovery import discover_mcp_tools
from tools.mcp_tool_lifecycle import shutdown_mcp_servers
from tools.mcp_tool_loop import reconnect_mcp_server
from tools.mcp_tool_scope import _resolve_server_key
from cron.scheduler import _init_cron_mcp_tools
from agent.agent_init import _load_tools
from run_agent import AIAgent
import model_tools

# Reject an accidental import from an installed/live source tree.
for module in (
    core,
    sys.modules["cron.scheduler"],
    model_tools,
    sys.modules["agent.agent_init"],
):
    assert Path(module.__file__).resolve().is_relative_to(repo)

control = home / "probe"
control.mkdir()
(control / "mode").write_text("healthy")
config = {
    "mcp_servers": {
        "probe": {
            "command": sys.executable,
            "args": [str(Path(__file__).with_name("probe_server.py")), str(control)],
            "connect_timeout": 5,
            "timeout": 5,
            "sampling": {"enabled": False},
        }
    },
    "tools": {"tool_search": {"enabled": "off"}},
}
(home / "config.yaml").write_text(yaml.safe_dump(config))
stop = threading.Event()
threads = []
result = {}
try:
    discover_mcp_tools()
    server = core._servers[_resolve_server_key("probe")]
    assert server.session is not None and server._ever_connected
    assert "hello initial" in model_tools.handle_function_call(
        "mcp__probe__hello", {"name": "initial"}
    )
    (control / "mode").write_text("down")
    # Only our fixture's child PID is terminated, never another MCP process.
    os.kill(int((control / "pid").read_text()), signal.SIGTERM)
    assert reconnect_mcp_server("probe")
    wait_until(
        lambda: server.session is None and server._park_reason,
        timeout=150,
        what="formerly healthy MCP server to exhaust retries and park",
    )
    assert not server._registered_tool_names
    (control / "mode").write_text(
        "delayed-list" if case == "delayed-list" else "recover"
    )

    def restore():
        if not stop.wait(0.5):
            (control / "restore").touch()

    session_before_tools = threading.Event()

    def watch_registration():
        while not stop.is_set():
            if server.session is not None and not server._registered_tool_names:
                session_before_tools.set()
            stop.wait(0.01)

    for target in (restore, watch_registration):
        thread = threading.Thread(target=target)
        threads.append(thread)
        thread.start()
    start = time.monotonic()
    _init_cron_mcp_tools("stdio-regression")
    result["init_seconds"] = time.monotonic() - start
    # Real AIAgent tool-loading phase only: no credentials, provider or model.
    agent = object.__new__(AIAgent)
    agent.quiet_mode = True
    agent.platform = "cron"
    _load_tools(agent, ["mcp-probe"], [])
    names = sorted(name for name in agent.valid_tool_names if name.startswith("mcp__"))
    result["snapshot_mcp_names"] = names
    result["snapshot_present"] = "mcp__probe__hello" in names
    calls = {}
    if result["snapshot_present"]:
        calls["probe"] = model_tools.handle_function_call(
            "mcp__probe__hello",
            {"name": "scheduled"},
            enabled_tools=agent.valid_tool_names,
        )
    result.update(
        calls=calls,
        call_ok="hello scheduled" in calls.get("probe", ""),
        session_before_tools=session_before_tools.is_set(),
        completed=True,
    )
finally:
    stop.set()
    for thread in threads:
        thread.join(timeout=3)
    shutdown_mcp_servers()
    (home / "result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
