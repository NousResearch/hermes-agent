"""Tests for validate_required_params — schema-driven missing-param detection.

These tests verify that the function correctly identifies required parameters
that are missing from a tool call's arguments, so the agent gets an actionable
error message instead of a confusing KeyError deep inside a tool handler.
"""

import json
import uuid

from tools.registry import registry, ToolRegistry


def _make_test_registry():
    """Create a fresh ToolRegistry isolated from the global one."""
    return ToolRegistry()


def _register_test_tool(reg, name, required, properties=None):
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

    def _noop_handler(*args, **kwargs):
        return '{"ok": true}'

    reg.register(
        name=name,
        toolset="test",
        schema=schema,
        handler=_noop_handler,
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
