"""Framework kwargs parity; a nesting registry handler is not the sandbox.

The built-in execute_code consumes a smaller kwargs subset. These cases pin
forwarding into its registry seam, not support for clarify in the sandbox.
"""
import json

import pytest
import model_tools
from tools.registry import registry
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.mark.parametrize("name", ["clarify_parity_probe", "execute_code"])
@pytest.mark.parametrize("present", [False, True])
def test_registry_framework_callback_and_nested_clarify(monkeypatch, name, present):
    manager = PluginManager()
    manager._cli_ref = None
    ctx = PluginContext(PluginManifest(name="parity-probe", source="user"), manager)
    captured = {}
    calls = []

    def callback(question, choices):
        calls.append(question)
        return "REAL_USER"

    def handler(args, **kwargs):
        captured.update(kwargs)
        return ctx.dispatch_tool("clarify", {
            "question": "Proceed?", "callback": "SPOOF", "clarify_callback": "SPOOF",
        }, **kwargs)

    entry = registry.get_entry(name)
    if entry is None:
        registry.register(name=name, toolset="parityprobe", schema={
            "name": name, "description": "probe", "parameters": {"type": "object", "properties": {}},
        }, handler=handler)
    else:
        monkeypatch.setattr(entry, "handler", handler)
    try:
        cb = callback if present else None
        raw = model_tools._execute_tool(name, {}, {}, model_tools._CallIds(task_id="isolated"),
            user_task="task", enabled_tools=["terminal"],
            clarify_callback=cb, skip_tool_execution_middleware=True)
        assert ("clarify_callback" in captured) is present
        if present:
            assert captured["clarify_callback"] is callback
            assert json.loads(raw)["user_response"] == "REAL_USER"
            assert calls == ["Proceed?"]
        else:
            assert "not available in this execution context" in json.loads(raw)["error"].lower()
            assert calls == []
        if name == "execute_code":
            assert captured["enabled_tools"] == ["terminal"]
            assert "user_task" not in captured
        else:
            assert captured["user_task"] == "task"
    finally:
        if entry is None:
            registry.deregister(name)
