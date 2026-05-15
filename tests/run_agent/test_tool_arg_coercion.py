"""Tests for tool argument type coercion.

When LLMs return tool call arguments, they frequently put numbers as strings
("42" instead of 42) and booleans as strings ("true" instead of true).
coerce_tool_args() fixes these type mismatches by comparing argument values
against the tool's JSON Schema before dispatch.
"""

import pytest
from unittest.mock import patch

from model_tools import (
    coerce_tool_args,
    _coerce_value,
    _coerce_number,
    _coerce_boolean,
)


# ── Low-level coercion helpers ────────────────────────────────────────────


class TestCoerceNumber:
    """Unit tests for _coerce_number."""

    def test_integer_string(self):
        assert _coerce_number("42") == 42
        assert isinstance(_coerce_number("42"), int)

    def test_negative_integer(self):
        assert _coerce_number("-7") == -7

    def test_zero(self):
        assert _coerce_number("0") == 0
        assert isinstance(_coerce_number("0"), int)

    def test_float_string(self):
        assert _coerce_number("3.14") == 3.14
        assert isinstance(_coerce_number("3.14"), float)

    def test_float_with_zero_fractional(self):
        """3.0 should become int(3) since there's no fractional part."""
        assert _coerce_number("3.0") == 3
        assert isinstance(_coerce_number("3.0"), int)

    def test_integer_only_rejects_float(self):
        """When integer_only=True, "3.14" should stay as string."""
        result = _coerce_number("3.14", integer_only=True)
        assert result == "3.14"
        assert isinstance(result, str)

    def test_integer_only_accepts_whole(self):
        assert _coerce_number("42", integer_only=True) == 42

    def test_not_a_number(self):
        assert _coerce_number("hello") == "hello"

    def test_empty_string(self):
        assert _coerce_number("") == ""

    def test_large_number(self):
        assert _coerce_number("1000000") == 1000000

    def test_scientific_notation(self):
        assert _coerce_number("1e5") == 100000

    def test_inf_stays_string(self):
        """Infinity is not JSON-serializable, so it should stay as string."""
        result = _coerce_number("inf")
        assert result == "inf"
        assert isinstance(result, str)

    def test_negative_inf_stays_string(self):
        """Negative infinity should also stay as string."""
        result = _coerce_number("-inf")
        assert result == "-inf"
        assert isinstance(result, str)

    def test_nan_stays_string(self):
        """NaN is not JSON-serializable, so it should stay as string."""
        result = _coerce_number("nan")
        assert result == "nan"
        assert isinstance(result, str)

    def test_negative_float(self):
        assert _coerce_number("-2.5") == -2.5


class TestCoerceBoolean:
    """Unit tests for _coerce_boolean."""

    def test_true_lowercase(self):
        assert _coerce_boolean("true") is True

    def test_false_lowercase(self):
        assert _coerce_boolean("false") is False

    def test_true_mixed_case(self):
        assert _coerce_boolean("True") is True

    def test_false_mixed_case(self):
        assert _coerce_boolean("False") is False

    def test_true_with_whitespace(self):
        assert _coerce_boolean("  true  ") is True

    def test_not_a_boolean(self):
        assert _coerce_boolean("yes") == "yes"

    def test_one_zero_not_coerced(self):
        """'1' and '0' are not boolean values."""
        assert _coerce_boolean("1") == "1"
        assert _coerce_boolean("0") == "0"

    def test_empty_string(self):
        assert _coerce_boolean("") == ""


class TestCoerceValue:
    """Unit tests for _coerce_value."""

    def test_integer_type(self):
        assert _coerce_value("5", "integer") == 5

    def test_number_type(self):
        assert _coerce_value("3.14", "number") == 3.14

    def test_boolean_type(self):
        assert _coerce_value("true", "boolean") is True

    def test_string_type_passthrough(self):
        """Strings expected as strings should not be coerced."""
        assert _coerce_value("hello", "string") == "hello"

    def test_unknown_type_passthrough(self):
        assert _coerce_value("stuff", "object") == "stuff"

    def test_union_type_prefers_first_match(self):
        """Union types try each in order."""
        assert _coerce_value("42", ["integer", "string"]) == 42

    def test_union_type_falls_through(self):
        """If no type matches, return original string."""
        assert _coerce_value("hello", ["integer", "boolean"]) == "hello"

    def test_union_with_string_preserves_original(self):
        """A non-numeric string in [number, string] should stay a string."""
        assert _coerce_value("hello", ["number", "string"]) == "hello"

    def test_array_type_parsed_from_json_string(self):
        """Stringified JSON arrays are parsed into native lists."""
        assert _coerce_value('["a", "b"]', "array") == ["a", "b"]
        assert _coerce_value("[1, 2, 3]", "array") == [1, 2, 3]

    def test_object_type_parsed_from_json_string(self):
        """Stringified JSON objects are parsed into native dicts."""
        assert _coerce_value('{"k": "v"}', "object") == {"k": "v"}
        assert _coerce_value('{"n": 1}', "object") == {"n": 1}

    def test_array_invalid_json_preserved(self):
        """Unparseable strings are returned unchanged."""
        assert _coerce_value("not-json", "array") == "not-json"

    def test_object_invalid_json_preserved(self):
        assert _coerce_value("not-json", "object") == "not-json"

    def test_array_type_wrong_shape_preserved(self):
        """A JSON object passed for an 'array' slot is preserved as a string."""
        assert _coerce_value('{"k": "v"}', "array") == '{"k": "v"}'

    def test_object_type_wrong_shape_preserved(self):
        """A JSON array passed for an 'object' slot is preserved as a string."""
        assert _coerce_value('["a"]', "object") == '["a"]'


# ── Full coerce_tool_args with registry ───────────────────────────────────


class TestCoerceToolArgs:
    """Integration tests for coerce_tool_args using the tool registry."""

    def _mock_schema(self, properties):
        """Build a minimal tool schema with the given properties."""
        return {
            "name": "test_tool",
            "description": "test",
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        }

    def test_coerces_integer_arg(self):
        schema = self._mock_schema({"limit": {"type": "integer"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"limit": "10"}
            result = coerce_tool_args("test_tool", args)
            assert result["limit"] == 10
            assert isinstance(result["limit"], int)

    def test_coerces_boolean_arg(self):
        schema = self._mock_schema({"merge": {"type": "boolean"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"merge": "true"}
            result = coerce_tool_args("test_tool", args)
            assert result["merge"] is True

    def test_coerces_number_arg(self):
        schema = self._mock_schema({"temperature": {"type": "number"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"temperature": "0.7"}
            result = coerce_tool_args("test_tool", args)
            assert result["temperature"] == 0.7

    def test_leaves_string_args_alone(self):
        schema = self._mock_schema({"path": {"type": "string"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"path": "/tmp/file.txt"}
            result = coerce_tool_args("test_tool", args)
            assert result["path"] == "/tmp/file.txt"

    def test_leaves_already_correct_types(self):
        schema = self._mock_schema({"limit": {"type": "integer"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"limit": 10}
            result = coerce_tool_args("test_tool", args)
            assert result["limit"] == 10

    def test_unknown_tool_returns_args_unchanged(self):
        with patch("model_tools.registry.get_schema", return_value=None):
            args = {"limit": "10"}
            result = coerce_tool_args("unknown_tool", args)
            assert result["limit"] == "10"

    def test_empty_args(self):
        assert coerce_tool_args("test_tool", {}) == {}

    def test_none_args(self):
        assert coerce_tool_args("test_tool", None) is None

    def test_preserves_non_string_values(self):
        """Lists, dicts, and other non-string values are never touched."""
        schema = self._mock_schema({
            "items": {"type": "array"},
            "config": {"type": "object"},
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"items": [1, 2, 3], "config": {"key": "val"}}
            result = coerce_tool_args("test_tool", args)
            assert result["items"] == [1, 2, 3]
            assert result["config"] == {"key": "val"}

    def test_coerces_stringified_array_arg(self):
        """Regression for #3947 — MCP servers using z.array() expect lists, not strings."""
        schema = self._mock_schema({
            "messageIds": {"type": "array", "items": {"type": "string"}},
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"messageIds": '["abc", "def"]'}
            result = coerce_tool_args("test_tool", args)
            assert result["messageIds"] == ["abc", "def"]

    def test_coerces_stringified_object_arg(self):
        """Stringified JSON objects get parsed into dicts."""
        schema = self._mock_schema({"config": {"type": "object"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"config": '{"max": 50}'}
            result = coerce_tool_args("test_tool", args)
            assert result["config"] == {"max": 50}

    def test_coerces_string_null_for_nullable_object_arg(self):
        """Models often emit literal "null" for optional MCP object args."""
        schema = self._mock_schema({
            "setting": {
                "type": "object",
                "additionalProperties": True,
                "nullable": True,
                "default": None,
            },
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"setting": "null"}
            result = coerce_tool_args("test_tool", args)
            assert result["setting"] is None

    def test_coerces_string_null_for_nullable_array_arg(self):
        schema = self._mock_schema({
            "stages": {
                "type": "array",
                "items": {"type": "object"},
                "nullable": True,
                "default": None,
            },
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"stages": "null"}
            result = coerce_tool_args("test_tool", args)
            assert result["stages"] is None

    def test_invalid_json_array_wrapped_in_single_element_list(self):
        """A bare string gets wrapped into ``[value]`` when the schema says array.

        Open-weight models (DeepSeek, Qwen, GLM) sometimes emit
        ``{"urls": "https://a.com"}`` when the tool expects a list.
        Wrapping produces a valid dispatch rather than a confusing tool
        failure.  This supersedes the earlier "pass the string through"
        behavior — no real tool handles a bare string as an array
        gracefully.
        """
        schema = self._mock_schema({"items": {"type": "array"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"items": "not-json"}
            result = coerce_tool_args("test_tool", args)
            assert result["items"] == ["not-json"]

    def test_bare_string_wrapped_as_array(self):
        """Bare string on array field → single-element list."""
        schema = self._mock_schema({"urls": {"type": "array", "items": {"type": "string"}}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"urls": "https://a.com"}
            result = coerce_tool_args("test_tool", args)
            assert result["urls"] == ["https://a.com"]

    def test_bare_int_wrapped_as_array(self):
        """Bare non-string scalars (int, bool, float) also get wrapped."""
        schema = self._mock_schema({"ids": {"type": "array", "items": {"type": "integer"}}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"ids": 5}
            result = coerce_tool_args("test_tool", args)
            assert result["ids"] == [5]

    def test_bare_dict_wrapped_as_array(self):
        """Bare dict on array field → single-element list."""
        schema = self._mock_schema({"items": {"type": "array"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"items": {"a": 1}}
            result = coerce_tool_args("test_tool", args)
            assert result["items"] == [{"a": 1}]

    def test_none_on_array_field_preserved(self):
        """``None`` is never wrapped — tools with defaults handle it."""
        schema = self._mock_schema({"items": {"type": "array"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"items": None}
            result = coerce_tool_args("test_tool", args)
            assert result["items"] is None

    def test_existing_list_passthrough(self):
        """An already-valid list is not touched."""
        schema = self._mock_schema({"items": {"type": "array"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"items": ["a", "b"]}
            result = coerce_tool_args("test_tool", args)
            assert result["items"] == ["a", "b"]

    def test_json_encoded_array_still_parses(self):
        """JSON-encoded strings still parse (not double-wrapped)."""
        schema = self._mock_schema({"items": {"type": "array"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"items": '["a","b"]'}
            result = coerce_tool_args("test_tool", args)
            assert result["items"] == ["a", "b"]

    def test_extra_args_without_schema_left_alone(self):
        """Args not in the schema properties are not touched."""
        schema = self._mock_schema({"limit": {"type": "integer"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"limit": "10", "extra": "42"}
            result = coerce_tool_args("test_tool", args)
            assert result["limit"] == 10
            assert result["extra"] == "42"  # no schema for extra, stays string

    def test_mixed_coercion(self):
        """Multiple args coerced in the same call."""
        schema = self._mock_schema({
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
            "full": {"type": "boolean"},
            "path": {"type": "string"},
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {
                "offset": "1",
                "limit": "500",
                "full": "false",
                "path": "readme.md",
            }
            result = coerce_tool_args("test_tool", args)
            assert result["offset"] == 1
            assert result["limit"] == 500
            assert result["full"] is False
            assert result["path"] == "readme.md"

    def test_failed_coercion_preserves_original(self):
        """A non-parseable string stays as string even if schema says integer."""
        schema = self._mock_schema({"limit": {"type": "integer"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"limit": "not_a_number"}
            result = coerce_tool_args("test_tool", args)
            assert result["limit"] == "not_a_number"

    def test_real_read_file_schema(self):
        """Test against the actual read_file schema from the registry."""
        # This uses the real registry — read_file should be registered
        args = {"path": "foo.py", "offset": "10", "limit": "100"}
        result = coerce_tool_args("read_file", args)
        assert result["path"] == "foo.py"
        assert result["offset"] == 10
        assert isinstance(result["offset"], int)
        assert result["limit"] == 100
        assert isinstance(result["limit"], int)

    # ── Union schema (anyOf/oneOf) type resolution ─────────────────────────

    def test_schema_expected_type_direct(self):
        """Direct type key is returned as-is."""
        from model_tools import _schema_expected_type
        assert _schema_expected_type({"type": "string"}) == "string"
        assert _schema_expected_type({"type": "integer"}) == "integer"
        assert _schema_expected_type({"type": "array"}) == "array"

    def test_schema_expected_type_anyof_single(self):
        """anyOf with one non-null variant returns a single type string."""
        from model_tools import _schema_expected_type
        result = _schema_expected_type({
            "anyOf": [{"type": "integer"}, {"type": "null"}]
        })
        assert result == "integer"

    def test_schema_expected_type_anyof_multi(self):
        """anyOf with multiple non-null variants returns a list of types."""
        from model_tools import _schema_expected_type
        result = _schema_expected_type({
            "anyOf": [{"type": "string"}, {"type": "array"}]
        })
        assert isinstance(result, list)
        assert set(result) == {"string", "array"}

    def test_schema_expected_type_oneof(self):
        """oneOf union is handled identically to anyOf."""
        from model_tools import _schema_expected_type
        result = _schema_expected_type({
            "oneOf": [{"type": "boolean"}, {"type": "null"}]
        })
        assert result == "boolean"

    def test_schema_expected_type_allof(self):
        """allOf falls back to the first variant with a type."""
        from model_tools import _schema_expected_type
        result = _schema_expected_type({
            "allOf": [{"type": "string"}]
        })
        assert result == "string"

    def test_schema_expected_type_none_for_empty(self):
        """None/invalid schema returns None."""
        from model_tools import _schema_expected_type
        assert _schema_expected_type(None) is None
        assert _schema_expected_type({}) is None
        assert _schema_expected_type("not a dict") is None

    def test_expected_includes_type_direct(self):
        """Direct type match."""
        from model_tools import _expected_includes_type
        assert _expected_includes_type("array", "array") is True
        assert _expected_includes_type("string", "array") is False

    def test_expected_includes_type_list(self):
        """Union type list containing the target type."""
        from model_tools import _expected_includes_type
        assert _expected_includes_type(["array", "null"], "array") is True
        assert _expected_includes_type(["string", "null"], "array") is False

    # ── Union schema coercion integration ──────────────────────────────────

    def test_coerces_integer_arg_with_anyof_union_schema(self):
        """String arg coerced to integer when property uses anyOf with integer variant."""
        schema = self._mock_schema({
            "limit": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "null"},
                ],
            },
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"limit": "42"}
            result = coerce_tool_args("test_tool", args)
            assert result["limit"] == 42
            assert isinstance(result["limit"], int)

    def test_coerces_string_arg_with_oneof_union_schema(self):
        """String arg passes through when property uses oneOf with string variant."""
        schema = self._mock_schema({
            "name": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "null"},
                ],
            },
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"name": "hello"}
            result = coerce_tool_args("test_tool", args)
            assert result["name"] == "hello"

    def test_coerces_array_arg_with_anyof_union_schema(self):
        """JSON array string parsed when property uses anyOf with array variant."""
        schema = self._mock_schema({
            "tags": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "null"},
                ],
            },
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"tags": '["a", "b"]'}
            result = coerce_tool_args("test_tool", args)
            assert result["tags"] == ["a", "b"]

    def test_bare_string_wrapped_as_array_for_union_schema(self):
        """Bare string wrapped in list when anyOf includes array + null."""
        schema = self._mock_schema({
            "urls": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "null"},
                ],
            },
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"urls": "https://a.com"}
            result = coerce_tool_args("test_tool", args)
            assert result["urls"] == ["https://a.com"]

    def test_multitype_union_coercion_tries_each_type(self):
        """Multi-type union (string | array): JSON array string parsed to list."""
        schema = self._mock_schema({
            "url": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
            },
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            # JSON array string should be parsed into an actual list
            args = {"url": '["https://example.com"]'}
            result = coerce_tool_args("test_tool", args)
            assert result["url"] == ["https://example.com"]

    def test_multitype_union_bare_string_wrapped_as_array(self):
        """Multi-type union (string | array): bare non-JSON string wrapped in list."""
        schema = self._mock_schema({
            "url": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
            },
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            # Bare string that's not a JSON array gets wrapped in a list
            # (array variant takes precedence in the union type list)
            args = {"url": "https://example.com"}
            result = coerce_tool_args("test_tool", args)
            assert result["url"] == ["https://example.com"]

    def test_null_value_preserved_with_anyof_nullable(self):
        """null literal is preserved when anyOf includes null variant."""
        schema = self._mock_schema({
            "limit": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "null"},
                ],
            },
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"limit": "null"}
            result = coerce_tool_args("test_tool", args)
            assert result["limit"] is None

    def test_coerces_boolean_arg_with_anyof_union_schema(self):
        """String 'true' coerced to boolean when anyOf includes boolean variant."""
        schema = self._mock_schema({
            "enabled": {
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "null"},
                ],
            },
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"enabled": "true"}
            result = coerce_tool_args("test_tool", args)
            assert result["enabled"] is True

    def test_coerces_number_arg_with_oneof_union_schema(self):
        """String number coerced when oneOf includes number variant."""
        schema = self._mock_schema({
            "threshold": {
                "oneOf": [
                    {"type": "number"},
                    {"type": "null"},
                ],
            },
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"threshold": "3.14"}
            result = coerce_tool_args("test_tool", args)
            assert result["threshold"] == 3.14
