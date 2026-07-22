"""Tests for validate_required_params — schema-driven missing-param detection.

These tests verify that the function correctly identifies required parameters
that are missing from a tool call's arguments, so the agent gets an actionable
error message instead of a confusing KeyError deep inside a tool handler.
"""

import json
import types

from hermes_cli.plugins import PluginManager
from tools.registry import ToolRegistry


def _make_test_registry():
    """Create a fresh ToolRegistry isolated from the global one."""
    return ToolRegistry()


def _register_test_tool(reg, name, required, properties=None, tool_handler=None):
    """Register a minimal tool schema in the given registry."""
    properties = properties or {}
    schema = {
        "name": name,
        "description": f"Test tool {name}",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }

    if tool_handler is None:
        def _noop_handler(*args, **kwargs):
            return '{"ok": true}'
        tool_handler = _noop_handler

    reg.register(
        name=name,
        toolset="test",
        schema=schema,
        handler=tool_handler,
    )
    return schema


class TestValidateRequiredParams:
    """Core behavior tests using an isolated registry."""

    def test_all_required_present(self):
        """Returns empty list when all required params are present."""
        reg = _make_test_registry()
        _register_test_tool(reg, "test_all_present", ["a", "b"])
        # Monkey-patch the module-level registry
        import model_tools
        orig = model_tools.registry
        model_tools.registry = reg
        try:
            result = model_tools.validate_required_params("test_all_present", {"a": 1, "b": 2})
            assert result == []
        finally:
            model_tools.registry = orig

    def test_one_required_missing(self):
        """Returns the missing param name."""
        reg = _make_test_registry()
        _register_test_tool(reg, "test_one_missing", ["a", "b"])
        import model_tools
        orig = model_tools.registry
        model_tools.registry = reg
        try:
            result = model_tools.validate_required_params("test_one_missing", {"a": 1})
            assert result == ["b"]
        finally:
            model_tools.registry = orig

    def test_multiple_required_missing(self):
        """Returns all missing param names in order."""
        reg = _make_test_registry()
        _register_test_tool(reg, "test_multi_missing", ["a", "b", "c"])
        import model_tools
        orig = model_tools.registry
        model_tools.registry = reg
        try:
            result = model_tools.validate_required_params("test_multi_missing", {})
            assert result == ["a", "b", "c"]
        finally:
            model_tools.registry = orig

    def test_no_required_array(self):
        """Tool without a required array → empty list (all optional)."""
        reg = _make_test_registry()
        schema = {
            "name": "test_no_required",
            "description": "No required params",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
            },
        }

        def _noop(*a, **kw):
            return '{"ok": true}'

        reg.register(name="test_no_required", toolset="test", schema=schema, handler=_noop)
        import model_tools
        orig = model_tools.registry
        model_tools.registry = reg
        try:
            result = model_tools.validate_required_params("test_no_required", {})
            assert result == []
        finally:
            model_tools.registry = orig

    def test_empty_required_array(self):
        """Tool with empty required array → empty list."""
        reg = _make_test_registry()
        _register_test_tool(reg, "test_empty_required", [])
        import model_tools
        orig = model_tools.registry
        model_tools.registry = reg
        try:
            result = model_tools.validate_required_params("test_empty_required", {})
            assert result == []
        finally:
            model_tools.registry = orig

    def test_unknown_tool(self):
        """Unknown tool name → empty list (no schema to check)."""
        import model_tools
        result = model_tools.validate_required_params("nonexistent_tool_xyz_12345", {"a": 1})
        assert result == []

    def test_non_dict_args(self):
        """Non-dict args → empty list (defensive)."""
        reg = _make_test_registry()
        _register_test_tool(reg, "test_nondict", ["a"])
        import model_tools
        orig = model_tools.registry
        model_tools.registry = reg
        try:
            result = model_tools.validate_required_params("test_nondict", None)
            assert result == []
        finally:
            model_tools.registry = orig

    def test_empty_dict_with_required(self):
        """Empty dict with required params → all missing."""
        reg = _make_test_registry()
        _register_test_tool(reg, "test_empty_args", ["x"])
        import model_tools
        orig = model_tools.registry
        model_tools.registry = reg
        try:
            result = model_tools.validate_required_params("test_empty_args", {})
            assert result == ["x"]
        finally:
            model_tools.registry = orig

    def test_extra_unexpected_args(self):
        """Extra args don't affect required validation."""
        reg = _make_test_registry()
        _register_test_tool(reg, "test_extra", ["a"])
        import model_tools
        orig = model_tools.registry
        model_tools.registry = reg
        try:
            result = model_tools.validate_required_params("test_extra", {"a": 1, "extra": True})
            assert result == []
        finally:
            model_tools.registry = orig

    def test_null_value_counts_as_present(self):
        """A param set to None is still 'present' — validation only checks key existence."""
        reg = _make_test_registry()
        _register_test_tool(reg, "test_null_val", ["a"])
        import model_tools
        orig = model_tools.registry
        model_tools.registry = reg
        try:
            result = model_tools.validate_required_params("test_null_val", {"a": None})
            assert result == []
        finally:
            model_tools.registry = orig


class TestRequiredParamDispatchBoundaries:
    """Integration coverage for middleware-adjusted effective arguments."""

    @staticmethod
    def _install_middleware(monkeypatch, **callbacks):
        manager = PluginManager()
        manager._middleware = {
            kind: [callback] for kind, callback in callbacks.items()
        }
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    def test_request_middleware_can_add_required_param(self, monkeypatch):
        import model_tools

        calls = []

        def handler(args, **kwargs):
            calls.append(args)
            return json.dumps({"ok": True, "args": args})

        reg = _make_test_registry()
        _register_test_tool(
            reg,
            "request_adds_required",
            ["command"],
            tool_handler=handler,
        )
        monkeypatch.setattr(model_tools, "registry", reg)

        def add_command(**kwargs):
            return {"args": {**kwargs["args"], "command": "printf ok"}}

        self._install_middleware(monkeypatch, tool_request=add_command)

        result = json.loads(model_tools.handle_function_call("request_adds_required", {}))

        assert result == {"ok": True, "args": {"command": "printf ok"}}
        assert calls == [{"command": "printf ok"}]

    def test_execution_middleware_cannot_remove_required_param(self, monkeypatch):
        import model_tools

        calls = []

        def handler(args, **kwargs):
            calls.append(args)
            return '{"ok": true}'

        reg = _make_test_registry()
        _register_test_tool(
            reg,
            "execution_removes_required",
            ["command"],
            tool_handler=handler,
        )
        monkeypatch.setattr(model_tools, "registry", reg)

        def remove_command(**kwargs):
            return kwargs["next_call"]({})

        self._install_middleware(monkeypatch, tool_execution=remove_command)

        result = json.loads(model_tools.handle_function_call(
            "execution_removes_required", {"command": "printf ok"}
        ))

        assert "Missing required parameter(s): command" in result["error"]
        assert calls == []

    def test_sequential_agent_tool_validates_final_args(self, monkeypatch):
        import model_tools
        from agent.tool_executor import _run_agent_tool_execution_middleware

        calls = []
        reg = _make_test_registry()
        _register_test_tool(reg, "memory", ["target"])
        monkeypatch.setattr(model_tools, "registry", reg)

        def remove_target(**kwargs):
            return kwargs["next_call"]({})

        self._install_middleware(monkeypatch, tool_execution=remove_target)
        agent = types.SimpleNamespace(
            session_id="session",
            _current_turn_id="turn",
            _current_api_request_id="request",
        )

        result, observed_args = _run_agent_tool_execution_middleware(
            agent,
            function_name="memory",
            function_args={"target": "memory"},
            effective_task_id="task",
            tool_call_id="call",
            execute=lambda args: calls.append(args) or '{"ok": true}',
        )

        assert "Missing required parameter(s): target" in json.loads(result)["error"]
        assert observed_args == {}
        assert calls == []

    def test_concurrent_agent_tool_validates_final_args(self, monkeypatch):
        import model_tools
        from agent.agent_runtime_helpers import invoke_tool

        reg = _make_test_registry()
        _register_test_tool(reg, "memory", ["target"])
        monkeypatch.setattr(model_tools, "registry", reg)

        def remove_target(**kwargs):
            return kwargs["next_call"]({})

        self._install_middleware(monkeypatch, tool_execution=remove_target)
        agent = types.SimpleNamespace(
            session_id="session",
            _current_turn_id="turn",
            _current_api_request_id="request",
            _memory_manager=None,
            _memory_store=None,
            valid_tool_names=set(),
        )

        result = invoke_tool(
            agent,
            "memory",
            {"target": "memory"},
            "task",
            tool_call_id="call",
        )

        assert "Missing required parameter(s): target" in json.loads(result)["error"]
