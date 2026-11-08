"""The gateway's ``reload-plugins`` verb connects an enabled plugin's MCP servers before it answers.

``hermes plugins install`` (a "y" at Enable-now, or ``--enable``), ``plugins enable`` and the Desktop
ask the running gateway to reload plugins, then tell the user the plugin's MCP servers apply "next
session". The forced re-discovery only registered the server configs; the servers connected on the
next housekeeping reconcile tick, up to a minute later, so a session started in between had none of
the plugin's MCP tools. Real plugin dir, real stdio MCP server, real registry: no fakes on that path.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

SERVER = '''\
import json, sys
TOOL = {"name": "ping", "description": "Probe tool.", "inputSchema": {"type": "object", "properties": {}}}
for line in sys.stdin:
    msg = json.loads(line) if line.strip() else {}
    rid = msg.get("id")
    if rid is None:
        continue
    method = msg.get("method")
    result = ({"protocolVersion": msg["params"].get("protocolVersion"), "capabilities": {"tools": {}},
               "serverInfo": {"name": "rpprobe", "version": "1"}} if method == "initialize"
              else {"tools": [TOOL]} if method == "tools/list" else {})
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}) + "\\n")
    sys.stdout.flush()
'''


@pytest.fixture
def gateway_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    plugin = home / "plugins" / "rpprobe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.json").write_text(json.dumps({
        "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
        "name": "rpprobe", "version": "0.1.0", "description": "reload-plugins MCP probe"}), encoding="utf-8")
    server = plugin / "server.py"
    server.write_text(f"#!{sys.executable}\n{SERVER}", encoding="utf-8")
    server.chmod(0o755)
    (plugin / "mcp.json").write_text(json.dumps({
        "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
        "mcpServers": {"rpprobe": {"type": "stdio", "command": "./server.py", "args": []}}}), encoding="utf-8")
    (home / "config.yaml").write_text("plugins:\n  enabled: [rpprobe]\n  disabled: []\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import plugins
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    plugins._reset_plugin_managers_for_tests()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield home, loop
    finally:
        shutdown_mcp_servers()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()
        plugins._reset_plugin_managers_for_tests()


# The fixture server launches as ./server.py via its shebang + exec bit, which Windows lacks.
@pytest.mark.platforms("posix")
def test_reload_plugins_answers_with_the_plugins_mcp_tools_already_registered(gateway_home):
    from gateway.run_plugin_rewire import reload_plugins_verb
    from tools.registry import registry
    _home, loop = gateway_home
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False), adapters={})

    answer = reload_plugins_verb(runner, loop)({})

    assert answer["reloaded"] is True and "rpprobe" in answer["plugins"]
    assert "mcp__rpprobe__ping" in registry.get_all_tool_names()
