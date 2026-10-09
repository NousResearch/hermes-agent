"""An MCP server named like a plugin toolset: selected by the MCP rules, never shadowed, never leaked.

Regression: Home Assistant left core for the ``homeassistant`` catalog plugin, so a
profile with that plugin and an MCP server named ``homeassistant`` has two owners of one name. The MCP
layer added the server under its bare name, which selects the PLUGIN toolset, so no session carried a
server tool (the Tool Search catalog shrank, ``tool_call`` answered "not available in this session").
The server must be selected under ``mcp-<name>`` wherever MCP servers are layered onto a selection,
and ``no_mcp`` or an allowlist naming another server must still keep it out (the plugin toolset is on
by default, so a name-level union would leak home control onto a platform that excluded it).
"""

import json
from types import SimpleNamespace

import pytest

from cron.scheduler import _merge_mcp_into_per_job_toolsets
from hermes_cli import tools_config
from hermes_cli.tools_config import _get_platform_tools
from model_tools import _select_tool_names
from tools.registry import ToolRegistry

HA_MCP = "mcp__homeassistant__GetLiveContext"
GH_MCP = "mcp__github__search"
CFG = {"mcp_servers": {"homeassistant": {"url": "http://ha"}, "github": {"command": "gh"}}}


def _schema(name):
    return {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    reg = ToolRegistry()
    for name, toolset in (("ha_get_state", "homeassistant"), (HA_MCP, "mcp-homeassistant"), (GH_MCP, "mcp-github")):
        reg.register(name=name, toolset=toolset, schema=_schema(name), handler=lambda args, **kw: "{}")
    reg.register_toolset_alias("homeassistant", "mcp-homeassistant")
    reg.register_toolset_alias("github", "mcp-github")
    monkeypatch.setattr("tools.registry.registry", reg)
    monkeypatch.setattr(tools_config, "_get_plugin_toolset_keys", lambda: {"homeassistant"})
    monkeypatch.setattr(tools_config, "_settled_plugin_toolset_keys", lambda: {"homeassistant"})
    monkeypatch.setattr("hermes_cli.plugins.get_portable_mcp_server_names_nowait", lambda: set())
    return reg


def _session_tools(toolsets):
    return _select_tool_names(sorted(toolsets), None, True)


@pytest.mark.parametrize("platform_toolsets", [
    ["hermes-cli"],
    ["file", "homeassistant", "memory", "terminal", "web"],  # an explicit list naming the plugin
])
def test_default_inclusion_selects_the_colliding_servers_tools(platform_toolsets):
    tools = _session_tools(_get_platform_tools({**CFG, "platform_toolsets": {"cli": platform_toolsets}}, "cli"))
    assert {HA_MCP, GH_MCP, "ha_get_state"} <= tools


@pytest.mark.parametrize("platform_toolsets, present, absent", [
    (["hermes-cli", "no_mcp"], {"ha_get_state"}, {HA_MCP, GH_MCP}),
    (["hermes-cli", "github"], {"ha_get_state", GH_MCP}, {HA_MCP}),
])
def test_opt_out_and_allowlist_still_exclude_the_colliding_server(platform_toolsets, present, absent):
    tools = _session_tools(_get_platform_tools({**CFG, "platform_toolsets": {"cli": platform_toolsets}}, "cli"))
    assert present <= tools and not absent & tools


@pytest.mark.parametrize("per_job, present, absent", [
    (["web"], {HA_MCP, GH_MCP}, set()),
    (["web", "homeassistant"], {HA_MCP, "ha_get_state"}, {GH_MCP}),  # naming it is the allowlist
    (["web", "github"], {GH_MCP}, {HA_MCP}),
    (["web", "no_mcp"], set(), {HA_MCP, GH_MCP}),
])
def test_cron_per_job_lists_follow_the_same_rules(per_job, present, absent):
    tools = _session_tools(_merge_mcp_into_per_job_toolsets(per_job, CFG))
    assert present <= tools and not absent & tools


def test_acp_keeps_the_plugin_toolset_and_the_colliding_server(monkeypatch):
    from acp_adapter.session import SessionManager

    seen: list[dict] = []
    config = {"model": {"default": "m"}, **CFG}
    class FakeAgent:
        def __init__(self, **kwargs):
            seen.append(kwargs)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **_kw: {})
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_kw: None)
    monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)

    SessionManager(db=None)._make_agent(session_id="fresh", cwd=".")

    assert {HA_MCP, GH_MCP, "ha_get_state"} <= _session_tools(seen[0]["enabled_toolsets"])


def test_disabling_the_name_disables_the_colliding_server_too():
    """``disabled_toolsets: [homeassistant]`` must not leave the server on under ``mcp-homeassistant``."""
    cfg = {**CFG, "platform_toolsets": {"cli": ["hermes-cli"]}, "agent": {"disabled_toolsets": ["homeassistant"]}}
    resolved = _get_platform_tools(cfg, "cli")
    assert "mcp-homeassistant" not in resolved and "homeassistant" not in resolved
    tools = _select_tool_names(sorted(resolved), ["homeassistant"], True)
    assert not {HA_MCP, "ha_get_state"} & tools and GH_MCP in tools
    # The runtime subtraction alone, for a selection built before the disable.
    runtime = _select_tool_names(["homeassistant", "mcp-homeassistant", "mcp-github"], ["homeassistant"], True)
    assert not {HA_MCP, "ha_get_state"} & runtime and GH_MCP in runtime


def test_a_server_named_like_a_static_toolset_does_not_enable_it(registry):
    """Adding servers ``web`` and ``terminal`` to a ``[file]`` platform adds the servers, not the built-ins."""
    from toolsets import resolve_toolset

    for name, toolset in (("mcp__web__fetch", "mcp-web"), ("mcp__terminal__run", "mcp-terminal")):
        registry.register(name=name, toolset=toolset, schema=_schema(name), handler=lambda args, **kw: "{}")
    cfg = {"mcp_servers": {"web": {"command": "w"}, "terminal": {"command": "t"}},
           "platform_toolsets": {"cli": ["file"]}}
    resolved = _get_platform_tools(cfg, "cli")
    assert {"mcp-web", "mcp-terminal"} <= resolved and not {"web", "terminal"} & resolved
    tools = _session_tools(resolved)
    assert {"mcp__web__fetch", "mcp__terminal__run"} <= tools
    builtin = set(resolve_toolset("web", include_registry=False)) | set(resolve_toolset("terminal", include_registry=False))
    assert not (builtin - set(resolve_toolset("file", include_registry=False))) & tools


def test_a_call_that_connects_a_colliding_server_scopes_in_its_key():
    from agent.inline_tool_executors import _scope_in_connected_mcp_servers

    agent = SimpleNamespace(enabled_toolsets=["file", "homeassistant"])
    result = json.dumps({"targets": [
        {"kind": "mcp", "name": "homeassistant", "state": "connected", "tools": ["GetLiveContext"]},
        {"kind": "mcp", "name": "github", "state": "connected", "tools": ["search"]},
    ]})
    _scope_in_connected_mcp_servers(agent, result)
    assert agent.enabled_toolsets == ["file", "homeassistant", "mcp-homeassistant", "github"]
    assert {HA_MCP, GH_MCP, "ha_get_state"} <= _session_tools(agent.enabled_toolsets)


def test_a_late_connection_to_a_colliding_server_is_adopted_under_its_key(monkeypatch):
    from tools.connectors import mcp

    class Approved:
        def poll(self):
            return {"status": "approved"}

    key = (mcp.hermes_home_key(), mcp.operation_session_key("s1"))
    monkeypatch.setitem(mcp._LATE_ATTEMPTS, key, {"homeassistant": Approved()})
    monkeypatch.setattr("tools.mcp_tool_config._load_mcp_config", lambda: {"homeassistant": {"url": "http://ha"}})
    monkeypatch.setattr("tools.mcp_tool_discovery.register_mcp_servers", lambda servers: list(servers))
    agent = SimpleNamespace(session_id="s1", enabled_toolsets=["file", "homeassistant"])
    assert mcp.adopt_late_connections(agent) == ["homeassistant"]
    assert agent.enabled_toolsets == ["file", "homeassistant", "mcp-homeassistant"]
    assert HA_MCP in _session_tools(agent.enabled_toolsets)


def test_cron_preflight_judges_a_colliding_server_by_its_own_tools(registry, monkeypatch):
    """A job naming ``homeassistant`` is refused when the SERVER has no tools: the plugin's tools under
    the same name must not count as the server's."""
    from cron.scheduler_preflight import _empty_requested_mcp_toolsets

    monkeypatch.setattr("tools.mcp_tool_discovery.mcp_server_reconnecting", lambda name: False)
    job = {"id": "j", "enabled_toolsets": ["web", "homeassistant"]}
    assert _empty_requested_mcp_toolsets(job, CFG) is None
    registry.deregister(HA_MCP)
    reason = _empty_requested_mcp_toolsets(job, CFG)
    assert reason and "homeassistant" in reason
