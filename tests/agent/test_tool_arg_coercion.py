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


# ── Deep-nesting depth budget (#132016) ────────────────────────────────────


class TestDeepNestingBudget:
    """Deep (schema, args) pairs must not exhaust the Python stack in the
    schema-guided normalizer. json.loads already accepts ~1000 nesting levels,
    so the payload reaches this walker with its depth intact; deep fixtures are
    built by iteration (not json.loads) because the C scanner is the first to
    die on some Python versions, which would mask the walker's own budget."""

    def test_deep_object_chain_does_not_recursion_error(self):
        schema = {"type": "string"}
        for _ in range(5_000):
            schema = {"type": "object", "properties": {"x": schema}}
        value = "leaf"
        for _ in range(5_000):
            value = {"x": value}
        out = _normalize_json_strings_for_schema(value, schema)
        assert isinstance(out, dict) and "x" in out

    def test_within_budget_deep_json_string_still_parsed(self):
        # 99 containers sit inside the budget: the JSON-encoded object string at
        # the deepest position must still be parsed (budget is not a blanket no-op).
        schema = {"type": "object", "properties": {"k": {"type": "integer"}}}
        value = '{"k": 1}'
        for _ in range(99):
            schema = {"type": "object", "properties": {"x": schema}}
            value = {"x": value}
        out = _normalize_json_strings_for_schema(value, schema)
        node = out
        for _ in range(99):
            node = node["x"]
        assert node == {"k": 1}

    def test_past_budget_deep_json_string_left_as_is(self):
        # Past the 100-container budget the walker stops descending: the deep
        # JSON-encoded string survives unparsed while a shallow sibling is still
        # normalized (conservative truncation, not a blanket rejection).
        deep_schema = {"type": "object", "properties": {"k": {"type": "integer"}}}
        deep_value = '{"k": 1}'
        for _ in range(150):
            deep_schema = {"type": "object", "properties": {"x": deep_schema}}
            deep_value = {"x": deep_value}
        schema = {"type": "object", "properties": {
            "x": deep_schema,
            "shallow": {"type": "object", "properties": {"k": {"type": "integer"}}},
        }}
        out = _normalize_json_strings_for_schema(
            {"x": deep_value, "shallow": '{"k": 2}'}, schema)
        assert out["shallow"] == {"k": 2}
        node = out["x"]
        for _ in range(150):
            node = node["x"]
        assert node == '{"k": 1}'

    def test_coerce_tool_args_deep_pair_end_to_end(self):
        schema = {"type": "string"}
        for _ in range(5_000):
            schema = {"type": "object", "properties": {"x": schema}}
        value = "leaf"
        for _ in range(5_000):
            value = {"x": value}
        full = {"parameters": {"type": "object", "properties": {"x": schema}}}
        with patch("tools.arg_coercion.registry.get_schema", return_value=full):
            out = coerce_tool_args("probe_deep", {"x": value})
        assert isinstance(out, dict) and "x" in out
