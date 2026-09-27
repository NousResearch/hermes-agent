"""Middleware receives host provenance without accepting a second authority source."""

import pytest

from hermes_cli.middleware import run_tool_execution_middleware
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools.registry import ToolRegistry, bind_host_context, get_current_host_context


def test_closed_policy_and_handler_share_one_binding(monkeypatch):
    manager = PluginManager()
    monkeypatch.setattr("hermes_cli.plugins._delivery_manager", lambda: manager)
    plugin = PluginContext(PluginManifest(name="host-policy", source="user"), manager)
    registry = ToolRegistry()
    provenance = object()
    executions = []

    def handler(args, *, host_context=None):
        executions.append(host_context)
        assert get_current_host_context() is host_context
        return "ok"

    registry.register("probe", "test", {"name": "probe"}, handler)

    def policy(tool_name, args, next_call, host_context=None):
        if host_context is not provenance:
            raise PermissionError("missing host authority")
        return next_call(args)

    plugin.register_middleware("tool_execution", policy, failure_mode="closed")
    with pytest.raises(PermissionError, match="missing host authority"):
        run_tool_execution_middleware("probe", {}, lambda args: registry.dispatch("probe", args))
    assert executions == []

    with bind_host_context(provenance):
        assert run_tool_execution_middleware(
            "probe", {}, lambda args: registry.dispatch("probe", args)) == "ok"
        with pytest.raises(ValueError, match="reserved"):
            run_tool_execution_middleware(
                "probe", {}, lambda args: registry.dispatch("probe", args),
                host_context=object())
    assert executions == [provenance]
    assert get_current_host_context() is None


def test_open_middleware_and_registration_compatibility(monkeypatch):
    manager = PluginManager()
    monkeypatch.setattr("hermes_cli.plugins._delivery_manager", lambda: manager)
    plugin = PluginContext(PluginManifest(name="observer", source="user"), manager)
    seen = []

    def old_callback(tool_name, args, next_call):
        seen.append(tool_name)
        raise RuntimeError("observer failed")

    plugin.register_middleware("tool_execution", old_callback)
    assert run_tool_execution_middleware("ordinary", {}, lambda args: "done") == "done"
    assert seen == ["ordinary"]
    with pytest.raises(ValueError, match="failure_mode"):
        plugin.register_middleware("tool_execution", old_callback, failure_mode="unknown")
    with pytest.raises(ValueError, match="execution middleware"):
        plugin.register_middleware("tool_request", old_callback, failure_mode="closed")
