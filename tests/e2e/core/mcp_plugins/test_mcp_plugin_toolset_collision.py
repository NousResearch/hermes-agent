"""An MCP server named like a PLUGIN toolset stays reachable through the Tool Search bridge.

Regression: Home Assistant left core for the ``homeassistant`` catalog plugin
(fb9fc8a3dd), so a profile with that plugin AND an MCP server named ``homeassistant`` has two owners
of one toolset name. The MCP layer added the server under its bare name, which selects the PLUGIN
toolset (the static-toolset case is merged by ``toolsets.get_toolset``, 3bc1780f2d). Discovery still
logged every MCP tool as registered and ``hermes mcp test`` still listed them, but no session
selected one: the plugin's tools were gated off (no ``HASS_TOKEN``), the Tool Search catalog shrank
and an exact ``tool_call`` answered "not available in this
session". The server is now layered on as ``mcp-<name>`` (``tools_config.mcp_server_toolset_key``).

One real ``hermes chat -q`` turn per platform selection against a real ``mcp`` 2.x server: the model
finds the server's tool with ``tool_search``, loads it with ``tool_describe`` and calls it with
``tool_call``; the assertions read what the server received, so a raw discovery that never reaches
a session cannot pass. The server is ``trust: untrusted``, proving the bridge keeps the trust gate:
a read-only tool runs, a write-capable one is refused before the RPC.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.mcp_plugins._helpers import (
    FINAL,
    build_home,
    calls_received,
    payload,
    provider,
    run_chat_q,
    script,
    stdio_server,
    tool_name,
    tool_names,
    tool_results,
)
from tests.e2e.core.mcp_plugins._plugin_helpers import reap_tagged

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="orphan sweep uses /proc"),
    pytest.mark.live_system_guard_bypass,  # reap_tagged signals only this run's tagged tree
]

SERVER = "homeassistant"
# The shape of the catalog plugin: tools in the ``homeassistant`` toolset, gated off without a token.
PLUGIN_MANIFEST = "name: homeassistant\nkind: standalone\nversion: 0.0.1\nprovides_tools:\n  - ha_get_state\n"
PLUGIN_INIT = '''
def register(ctx):
    ctx.register_tool(
        name="ha_get_state", toolset="homeassistant",
        schema={"name": "ha_get_state", "description": "Plugin tool, gated off without a token.",
                "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: "{}", check_fn=lambda: False)
'''


def _bridge_turn(root: Path, platform_toolsets: list[str]) -> dict[str, Any]:
    log = root / "inbound.jsonl"
    ro, rw = (tool_name(SERVER, n) for n in ("ro_probe", "rw_probe"))
    calls = [
        ("tool_search", {"queries": ["ro_probe"]}),
        ("tool_describe", {"names": [ro]}),
        ("tool_call", {"calls": [{"name": ro, "arguments": {"nonce": "r"}}]}),
        ("tool_call", {"calls": [{"name": rw, "arguments": {"nonce": "w"}}]}),
    ]
    extra = {"tools": {"tool_search": {"enabled": "on"}},
             "plugins": {"enabled": ["homeassistant"], "disabled": []},
             "platform_toolsets": {"cli": platform_toolsets}}
    with provider(script(*calls)) as srv:
        eh = build_home(root, srv.base_url, extra=extra)
        plugin = eh.hermes_home / "plugins" / "homeassistant"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(PLUGIN_MANIFEST, encoding="utf-8")
        (plugin / "__init__.py").write_text(PLUGIN_INIT, encoding="utf-8")
        cfg = {**stdio_server(SERVER, log, eh.tag, MCPE2E_CANARY=f"CANARY-{root.name}"), "trust": "untrusted"}
        eh.update_config(lambda c: c["mcp_servers"].__setitem__(SERVER, cfg))
        try:
            proc = run_chat_q(eh, "Check the house.")
        finally:
            reap_tagged(eh)
        detail = f"exit {proc.returncode}\nstdout: {proc.stdout[-1500:]}\nstderr: {proc.stderr[-1500:]}"
        assert proc.returncode == 0 and FINAL in proc.stdout, f"turn did not complete:\n{detail}"
        return {"log": log, "results": tool_results(srv), "canary": f"CANARY-{root.name}",
                "offered": tool_names(srv.main_requests()[0]), "ro": ro, "detail": detail}


@pytest.mark.parametrize("platform_toolsets", [["hermes-cli"], ["file", "homeassistant", "terminal", "web"]],
                         ids=["default bundle", "explicit homeassistant"])
def test_mcp_server_named_like_a_plugin_toolset_is_searchable_and_callable(
        tmp_path: Path, platform_toolsets: list[str]) -> None:
    obs = _bridge_turn(tmp_path, platform_toolsets)
    assert {"tool_search", "tool_describe", "tool_call"} <= obs["offered"], sorted(obs["offered"])
    search, describe, ro, rw = obs["results"]
    found = [name for group in payload(search)["results"] for name in group["matches"]]
    assert obs["ro"] in found, f"tool_search did not find the server's tool: {search}\n{obs['detail']}"
    assert obs["ro"] in describe and "nonce" in describe, describe
    assert calls_received(obs["log"], "ro_probe"), f"server never received the read-only call: {ro}"
    assert f"RO:{obs['canary']}:r" in ro, ro
    # The trust gate survives the bridge: a write-capable tool on an untrusted server never reaches it.
    assert not calls_received(obs["log"], "rw_probe"), "write-capable tool ran on an untrusted server"
    refusal = json.dumps(payload(rw))
    assert f"untrusted server '{SERVER}'" in refusal and obs["canary"] not in refusal, rw
