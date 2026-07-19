"""Generic plugin tool handlers receive the active parent agent context."""

from __future__ import annotations

import json

from agent.agent_runtime_helpers import invoke_tool
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def test_agent_dispatch_passes_parent_agent_to_plugin_tool(monkeypatch):
    from tools.registry import registry

    seen = {}

    def handler(args, **kwargs):
        seen["args"] = args
        seen["parent_agent"] = kwargs.get("parent_agent")
        return json.dumps({"ok": True})

    manager = PluginManager()
    context = PluginContext(PluginManifest(name="runtime-context-test"), manager)
    context.register_tool(
        name="plugin_runtime_context_probe",
        toolset="plugin_runtime_context_test",
        schema={
            "name": "plugin_runtime_context_probe",
            "description": "probe",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=handler,
    )
    agent = type(
        "FakeAgent",
        (),
        {
            "session_id": "parent-session",
            "valid_tool_names": ["plugin_runtime_context_probe"],
            "enabled_toolsets": ["plugin_runtime_context_test"],
            "disabled_toolsets": [],
            "_memory_manager": None,
            "_current_turn_id": "turn",
            "_current_api_request_id": "request",
        },
    )()

    result = invoke_tool(
        agent,
        "plugin_runtime_context_probe",
        {},
        "task",
        pre_tool_block_checked=True,
        skip_tool_request_middleware=True,
    )

    assert json.loads(result) == {"ok": True}
    assert seen["parent_agent"] is agent
    registry.deregister("plugin_runtime_context_probe")
