"""Tests for tool argument schema validation.

validate_tool_args() checks tool call arguments against their JSON Schema
before dispatch, catching missing required fields, type mismatches, and
constraint violations (enum, minimum/maximum, minLength/maxLength) that
coerce_tool_args() does not cover.
"""

import json

import pytest

from model_tools import validate_tool_args, handle_function_call
from tools.registry import ToolRegistry


# ── Unit tests for validate_tool_args ──────────────────────────────────────


class TestValidateToolArgsRequiredFields:
    """Required field validation."""

    def test_missing_required_field(self):
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "mode": {"type": "string"},
                },
                "required": ["path"],
            }
        }
        errors = validate_tool_args({"mode": "read"}, schema)
        assert any("path" in e for e in errors)

    def test_all_required_present(self):
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            }
        }
        errors = validate_tool_args({"path": "/tmp/file.txt"}, schema)
        assert errors == []

    def test_no_required_in_schema(self):
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
            }
        }
        errors = validate_tool_args({}, schema)
        assert errors == []


class TestValidateToolArgsTypeCheck:
    """Type mismatch detection after coercion."""

    def test_wrong_type_integer(self):
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                },
            }
        }
        errors = validate_tool_args({"count": "not_a_number"}, schema)
        assert any("count" in e and "integer" in e for e in errors)

    def test_wrong_type_boolean(self):
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "verbose": {"type": "boolean"},
                },
            }
        }
        errors = validate_tool_args({"verbose": 42}, schema)
        assert any("verbose" in e and "boolean" in e for e in errors)

    def test_bool_rejected_for_integer_field(self):
        """Python bool is a subclass of int; JSON Schema treats them as distinct.
        True/False must not pass an 'integer' type check."""
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                },
            }
        }
        errors = validate_tool_args({"count": True}, schema)
        assert any("count" in e for e in errors), (
            "bool True should fail integer type check but got no errors"
        )

    def test_none_valid_for_nullable_union(self):
        """JSON Schema allows 'null' as a type; None must pass ['string', 'null']."""
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "tag": {"type": ["string", "null"]},
                },
            }
        }
        errors = validate_tool_args({"tag": None}, schema)
        assert errors == [], f"None should be valid for ['string', 'null'] but got: {errors}"

    def test_type_error_suppresses_enum_error(self):
        """When a value has the wrong type, the enum check should be skipped
        to avoid emitting two confusing errors for a single argument."""
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["read", "write"]},
                },
            }
        }
        # Pass an integer — wrong type AND not in enum
        errors = validate_tool_args({"mode": 42}, schema)
        assert len(errors) == 1, (
            f"Expected exactly one error (type mismatch), got {len(errors)}: {errors}"
        )
        assert "type" in errors[0].lower() or "wrong" in errors[0].lower()

    def test_correct_type_passes(self):
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer"},
                    "rate": {"type": "number"},
                    "flag": {"type": "boolean"},
                },
            }
        }
        args = {"name": "test", "count": 5, "rate": 3.14, "flag": True}
        errors = validate_tool_args(args, schema)
        assert errors == []


class TestValidateToolArgsEnum:
    """Enum constraint validation."""

    def test_value_not_in_enum(self):
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["read", "write"]},
                },
            }
        }
        errors = validate_tool_args({"mode": "delete"}, schema)
        assert any("mode" in e and "enum" in e for e in errors)

    def test_value_in_enum_passes(self):
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["read", "write"]},
                },
            }
        }
        errors = validate_tool_args({"mode": "read"}, schema)
        assert errors == []


class TestValidateToolArgsNumericConstraints:
    """minimum/maximum constraint validation."""

    def test_below_minimum(self):
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1},
                },
            }
        }
        errors = validate_tool_args({"limit": 0}, schema)
        assert any("limit" in e and "minimum" in e for e in errors)

    def test_above_maximum(self):
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "maximum": 100},
                },
            }
        }
        errors = validate_tool_args({"limit": 200}, schema)
        assert any("limit" in e and "maximum" in e for e in errors)

    def test_within_range_passes(self):
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            }
        }
        errors = validate_tool_args({"limit": 50}, schema)
        assert errors == []


class TestValidateToolArgsStringConstraints:
    """minLength/maxLength constraint validation."""

    def test_below_min_length(self):
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                },
            }
        }
        errors = validate_tool_args({"name": ""}, schema)
        assert any("name" in e and "minLength" in e for e in errors)

    def test_above_max_length(self):
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "maxLength": 10},
                },
            }
        }
        errors = validate_tool_args({"code": "a" * 11}, schema)
        assert any("code" in e and "maxLength" in e for e in errors)

    def test_within_length_passes(self):
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 100},
                },
            }
        }
        errors = validate_tool_args({"name": "hello"}, schema)
        assert errors == []


class TestValidateToolArgsIntegration:
    """Integration: validation is wired into handle_function_call."""

    def test_validation_error_returned_for_invalid_args(self):
        """When args fail validation, handle_function_call returns a JSON
        error without dispatching to the tool handler."""
        # Register a test tool with strict schema
        reg = ToolRegistry()
        reg.register(
            name="_test_validate_tool",
            toolset="test",
            schema={
                "name": "_test_validate_tool",
                "description": "Test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                },
            },
            handler=lambda args, **kw: json.dumps({"ok": True}),
        )

        # Temporarily patch the global registry
        from model_tools import registry as global_registry
        global_registry._tools["_test_validate_tool"] = reg._tools["_test_validate_tool"]

        try:
            result = handle_function_call(
                "_test_validate_tool",
                {},  # missing required "path"
            )
            parsed = json.loads(result)
            assert "error" in parsed
            assert "path" in parsed["error"]
        finally:
            global_registry._tools.pop("_test_validate_tool", None)
