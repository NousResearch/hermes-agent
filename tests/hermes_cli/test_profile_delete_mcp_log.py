"""Profile deletion after a real stdio MCP probe must release its log file."""

import sys
from pathlib import Path

import pytest

from hermes_cli import profiles
from hermes_cli.mcp_config import _probe_single_server
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import mcp_tool_config


@pytest.mark.windows_only
def test_delete_profile_after_stdio_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Do not manage host services or scan unrelated developer processes in this test.
    monkeypatch.setattr(profiles, "_cleanup_gateway_service", lambda *_: None)
    monkeypatch.setattr(profiles, "_stop_profile_backends", lambda *_: None)
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda *_: None)
    profile = profiles.create_profile("mcp-probe-delete", no_alias=True)
    server = tmp_path / "stdio_server.py"
    server.write_text('''import json
import sys
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    if request["method"] == "initialize":
        result = {"protocolVersion": request["params"]["protocolVersion"],
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "probe-fixture", "version": "1"}}
    elif request["method"] == "tools/list":
        result = {"tools": [{"name": "echo", "description": "Echo",
                             "inputSchema": {"type": "object"}}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
''', encoding="utf-8")
    token = set_hermes_home_override(profile)
    try:
        assert _probe_single_server("fixture", {"command": sys.executable, "args": [str(server)]}) == [("echo", "Echo")]
    finally:
        reset_hermes_home_override(token)
    try:
        profiles.delete_profile("mcp-probe-delete", yes=True)
        assert not profile.exists()
    finally:
        # Keep the pre-fix failure from leaking a Windows handle into tempdir cleanup.
        for handle in list(mcp_tool_config._mcp_stderr_log_fh.values()):
            if getattr(handle, "name", None) == str(profile / "logs" / "mcp-stderr.log"):
                handle.close()