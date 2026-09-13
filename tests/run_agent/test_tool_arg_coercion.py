"""Tests for tool argument type coercion.

When LLMs return tool call arguments, they frequently put numbers as strings
("42" instead of 42) and booleans as strings ("true" instead of true).
coerce_tool_args() fixes these type mismatches by comparing argument values
against the tool's JSON Schema before dispatch.
"""

from unittest.mock import patch

import model_tools  # noqa: F401 — populates the tool registry the "real schema" tests read
from tools.arg_coercion import (
    coerce_tool_args,
    _coerce_value,
    _coerce_number,
    _coerce_boolean,
    _schema_accepts_kind,
    _normalize_json_strings_for_schema,
    _union_schema_types,
)


# ── Low-level coercion helpers ────────────────────────────────────────────


class TestCoerceNumber:
    """Unit tests for _coerce_number."""

    def test_integer_string(self):
        assert _coerce_number("42") == 42
        assert isinstance(_coerce_number("42"), int)

    def test_negative_integer(self):
        assert _coerce_number("-7") == -7




    def test_integer_only_rejects_float(self):
        """When integer_only=True, "3.14" should stay as string."""
        result = _coerce_number("3.14", integer_only=True)
        assert result == "3.14"
        assert isinstance(result, str)











class TestCoerceBoolean:
    """Unit tests for _coerce_boolean."""

    def test_true_lowercase(self):
        assert _coerce_boolean("true") is True






    def test_one_zero_not_coerced(self):
        """'1' and '0' are not boolean values."""
        assert _coerce_boolean("1") == "1"
        assert _coerce_boolean("0") == "0"



class TestCoerceValue:
    """Unit tests for _coerce_value."""

    def test_integer_type(self):
        assert _coerce_value("5", "integer") == 5








    def test_array_type_parsed_from_json_string(self):
        """Stringified JSON arrays are parsed into native lists."""
        assert _coerce_value('["a", "b"]', "array") == ["a", "b"]
        assert _coerce_value("[1, 2, 3]", "array") == [1, 2, 3]







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
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            args = {"limit": "10"}
            result = coerce_tool_args("test_tool", args)
            assert result["limit"] == 10
            assert isinstance(result["limit"], int)




    def test_leaves_already_correct_types(self):
        schema = self._mock_schema({"limit": {"type": "integer"}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            args = {"limit": 10}
            result = coerce_tool_args("test_tool", args)
            assert result["limit"] == 10


    def test_empty_args(self):
        assert coerce_tool_args("test_tool", {}) == {}

















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


# ── Schema-guided nested JSON-string normalization (cline/cline#11803) ─────


class TestSchemaAcceptsKind:
    """Unit tests for _schema_accepts_kind."""

    def test_plain_type(self):
        assert _schema_accepts_kind({"type": "array"}, "array") is True
        assert _schema_accepts_kind({"type": "object"}, "object") is True
        assert _schema_accepts_kind({"type": "string"}, "array") is False



    def test_non_dict(self):
        assert _schema_accepts_kind(None, "array") is False


class TestNormalizeJsonStringsForSchema:
    """Unit tests for _normalize_json_strings_for_schema (the recursive pass)."""

    def test_parses_json_string_array_when_schema_expects_array(self):
        schema = {"type": "array", "items": {"type": "string"}}
        out = _normalize_json_strings_for_schema('["git status", "bun test"]', schema)
        assert out == ["git status", "bun test"]




    def test_native_list_preserved_identity(self):
        schema = {"type": "array", "items": {"type": "object", "properties": {}}}
        value = [{"id": "1"}]
        # Nothing to change — same object back (no-op identity preserved).
        assert _normalize_json_strings_for_schema(value, schema) is value

    def test_non_dict_schema_returns_value(self):
        assert _normalize_json_strings_for_schema("x", None) == "x"


class TestCoerceToolArgsNested:
    """Integration: nested JSON-string elements/fields are normalized via the
    registry schema, while legitimate string fields are preserved."""

    def _array_of_objects_schema(self):
        return {
            "name": "test_tool",
            "description": "test",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }

    def test_array_elements_as_json_strings_are_parsed(self):
        schema = self._array_of_objects_schema()
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            args = {"items": ['{"id": "1", "content": "x"}']}
            result = coerce_tool_args("test_tool", args)
            assert result["items"] == [{"id": "1", "content": "x"}]


    def test_string_subfield_with_json_content_preserved(self):
        """A string-typed sub-field whose value looks like JSON must NOT be parsed."""
        schema = self._array_of_objects_schema()
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            args = {"items": [{"id": "1", "content": '{"not": "parsed"}'}]}
            result = coerce_tool_args("test_tool", args)
            assert result["items"][0]["content"] == '{"not": "parsed"}'



    def test_real_todo_schema_element_strings(self):
        """Against the real todo schema from the registry."""
        import json as _json
        args = {"todos": [_json.dumps({"id": "1", "content": "x", "status": "pending"})]}
        result = coerce_tool_args("todo_list", args)
        assert result["todos"][0] == {"id": "1", "content": "x", "status": "pending"}


# ── Union (anyOf/oneOf) schema resolution ─────────────────────────────────


class TestUnionSchemaTypes:
    """Unit tests for _union_schema_types."""

    def test_single_type_union_returns_string(self):
        """Nullable union collapses to its single non-null type."""
        assert _union_schema_types({"anyOf": [{"type": "integer"}, {"type": "null"}]}) == "integer"

    def test_multitype_union_returns_list_of_types(self):
        """Multi-type union returns the concrete types in schema order."""
        assert _union_schema_types({"anyOf": [{"type": "integer"}, {"type": "string"}]}) == ["integer", "string"]

    def test_oneof_is_supported_too(self):
        assert _union_schema_types({"oneOf": [{"type": "boolean"}, {"type": "array"}]}) == ["boolean", "array"]

    def test_empty_union_returns_none(self):
        """Union variants without type info return None."""
        assert _union_schema_types({"anyOf": [{"minimum": 0}, {"maximum": 100}]}) is None

    def test_empty_variants_list_returns_none(self):
        assert _union_schema_types({"anyOf": []}) is None

    def test_no_union_returns_none(self):
        assert _union_schema_types({"type": "integer"}) is None

    def test_null_only_union_returns_null(self):
        assert _union_schema_types({"anyOf": [{"type": "null"}]}) == "null"


class TestCoerceToolArgsUnions:
    """anyOf/oneOf union properties resolve to their variant types for coercion."""

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

    # ── JSON-encoded containers under a union (the #23129 read_url shape) ──

    def test_json_array_string_parsed_for_string_array_union(self):
        """anyOf [string, array] + JSON-encoded list string → parsed list."""
        schema = self._mock_schema({"url": {"anyOf": [{"type": "string"}, {"type": "array"}]}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"url": '["https://example.com"]'})
            assert result["url"] == ["https://example.com"]

    def test_plain_string_kept_for_string_array_union(self):
        """anyOf [string, array] + plain string stays a string — no wrap."""
        schema = self._mock_schema({"url": {"anyOf": [{"type": "string"}, {"type": "array"}]}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"url": "https://example.com"})
            assert result["url"] == "https://example.com"

    def test_json_object_string_parsed_for_string_object_union(self):
        """anyOf [string, object] + JSON-encoded dict string → parsed dict."""
        schema = self._mock_schema({"data": {"anyOf": [{"type": "string"}, {"type": "object"}]}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"data": '{"key": "value"}'})
            assert result["data"] == {"key": "value"}

    # ── Unions without a string branch: repairs are unambiguous ──

    def test_integer_union_coerces_numeric_string(self):
        """anyOf [integer, null] + "42" → 42 (regression from #26029)."""
        schema = self._mock_schema({"limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"limit": "42"})
            assert result["limit"] == 42
            assert isinstance(result["limit"], int)

    def test_array_union_wraps_bare_scalar(self):
        """anyOf [array, null] + bare string → wrapped single-element list (regression from #25524)."""
        schema = self._mock_schema({
            "urls": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        })
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"urls": "https://a.com"})
            assert result["urls"] == ["https://a.com"]

    def test_array_union_parses_json_string(self):
        """anyOf [array, null] + JSON-encoded string → parsed list (regression from #26029)."""
        schema = self._mock_schema({
            "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        })
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"tags": '["a", "b"]'})
            assert result["tags"] == ["a", "b"]

    # ── Conservative: a string branch makes scalar repairs ambiguous ──

    def test_string_boolean_union_keeps_true_string(self):
        """anyOf [string, boolean] + "true" stays a string: the schema permits it as-is."""
        schema = self._mock_schema({"flag": {"anyOf": [{"type": "string"}, {"type": "boolean"}]}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"flag": "true"})
            assert result["flag"] == "true"

    def test_string_integer_union_keeps_numeric_string(self):
        """oneOf [string, integer] + "42" stays a string: the schema permits it as-is."""
        schema = self._mock_schema({"count": {"oneOf": [{"type": "string"}, {"type": "integer"}]}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"count": "42"})
            assert result["count"] == "42"

    def test_string_null_union_keeps_plain_string(self):
        """oneOf [string, null] + "hello" stays "hello" (regression from #26029)."""
        schema = self._mock_schema({"name": {"oneOf": [{"type": "string"}, {"type": "null"}]}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"name": "hello"})
            assert result["name"] == "hello"

    # ── Fall-through and precedence ──

    def test_union_without_type_variants_skipped(self):
        """anyOf variants without usable type info are left untouched."""
        schema = self._mock_schema({
            "val": {"anyOf": [{"$ref": "#/definitions/Foo"}, {"const": "bar"}]},
        })
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"val": "42"})
            assert result["val"] == "42"

    def test_explicit_type_takes_precedence_over_union(self):
        """A top-level `type` wins over anyOf when both are present."""
        schema = self._mock_schema({"val": {"type": "integer", "anyOf": [{"type": "string"}]}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"val": "42"})
            assert result["val"] == 42

    # ── Native containers under unions ──

    def test_native_list_under_union_normalized(self):
        """Native list value for a union property still gets nested JSON-string repair."""
        schema = self._mock_schema({
            "items": {
                "anyOf": [
                    {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}}}},
                    {"type": "null"},
                ],
            },
        })
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"items": ['{"id": "1"}']})
            assert result["items"] == [{"id": "1"}]

    # ── Real registry: terminal's notify (the #23213 review MRE) ──

    def test_real_terminal_notify_union(self):
        """terminal.notify is anyOf [boolean, array]; both string forms must repair.

        "true" → True and '["PROBE-A"]' → ["PROBE-A"]. Without the union fix,
        dispatch rejected both legal forms ("notify must be true/false or a
        list of strings") and background notifications were silently lost.
        background="true" → True is the single-variable control from the review.
        """
        result = coerce_tool_args("terminal", {
            "command": "echo hi",
            "background": "true",
            "notify": "true",
        })
        assert result["background"] is True
        assert result["notify"] is True

        result = coerce_tool_args("terminal", {
            "command": "echo hi",
            "background": "true",
            "notify": '["PROBE-A"]',
        })
        assert result["notify"] == ["PROBE-A"]

    def test_real_terminal_notify_native_bool_not_wrapped(self):
        """A native bool for the notify union must stay a bool.

        The union-driven array wrap is a repair for *string* arrays the model
        emitted bare; a native ``notify=true`` already satisfies the boolean
        branch. Wrapping it into ``[true]`` made dispatch map it onto
        watch_patterns with notify_on_complete=False, so background completion
        notifications were silently lost (CI regression caught by
        tests/tools/test_completed_process_results.py).
        """
        result = coerce_tool_args("terminal", {
            "command": "echo hi",
            "background": True,
            "notify": True,
        })
        assert result["notify"] is True

        result = coerce_tool_args("terminal", {
            "command": "echo hi",
            "background": True,
            "notify": False,
        })
        assert result["notify"] is False
