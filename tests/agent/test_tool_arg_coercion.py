"""Tests for tool argument type coercion.

When LLMs return tool call arguments, they frequently put numbers as strings
("42" instead of 42) and booleans as strings ("true" instead of true).
coerce_tool_args() fixes these type mismatches by comparing argument values
against the tool's JSON Schema before dispatch.
"""

from unittest.mock import patch

import model_tools
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

    def test_recursive_union_refs_keep_noncyclic_alternatives(self):
        for union_key in ("anyOf", "oneOf", "allOf"):
            root = {"$defs": {"Node": {
                union_key: [{"$ref": "#/$defs/Node"}, {"type": "object"}],
            }}}
            schema = {"type": "array", "items": {"$ref": "#/$defs/Node"}}
            assert _schema_accepts_kind(root["$defs"]["Node"], "array", root) is False
            assert _normalize_json_strings_for_schema(
                ['{"street": "Alpha Ave"}'], schema, root,
            ) == [{"street": "Alpha Ave"}]


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

    def test_object_arg_referenced_via_defs_is_parsed(self):
        schema = {
            "name": "test_tool",
            "description": "test",
            "parameters": {
                "type": "object",
                "$defs": {
                    "Address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "zip": {"type": "string"},
                        },
                    },
                },
                "properties": {
                    "address": {"$ref": "#/$defs/Address"},
                },
            },
        }
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            args = {"address": '{"street": "Alpha Ave", "zip": "A-001"}'}
            result = coerce_tool_args("test_tool", args)
            assert result["address"] == {
                "street": "Alpha Ave",
                "zip": "A-001",
            }

    def test_scalar_args_behind_a_ref_are_coerced(self):
        schema = {
            "name": "test_tool",
            "description": "test",
            "parameters": {
                "type": "object",
                "$defs": {"Count": {"type": "integer"}, "Flag": {"type": "boolean"}},
                "properties": {
                    "count": {"$ref": "#/$defs/Count"},
                    "verbose": {"$ref": "#/$defs/Flag", "description": "a keyword beside $ref"},
                },
            },
        }
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"count": "5", "verbose": "true"})
            assert result == {"count": 5, "verbose": True}

    def test_ref_paths_through_arrays_and_percent_escapes_resolve(self):
        schema = {
            "name": "test_tool",
            "description": "test",
            "parameters": {
                "type": "object",
                "$defs": {"My Type": {"type": "integer"}},
                "properties": {
                    "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    # zod-to-json-schema points a reused schema at its first occurrence by path.
                    "offset": {"$ref": "#/properties/limit/anyOf/0"},
                    "page": {"$ref": "#/$defs/My%20Type"},
                },
            },
        }
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"offset": "7", "page": "3"})
            assert result == {"offset": 7, "page": 3}

    def test_malformed_pointer_escape_is_left_unresolved(self):
        schema = {
            "name": "test_tool",
            "description": "test",
            "parameters": {
                "type": "object",
                "$defs": {"a~2b": {"type": "integer"}},
                "properties": {"count": {"$ref": "#/$defs/a~2b"}},
            },
        }
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            assert coerce_tool_args("test_tool", {"count": "5"}) == {"count": "5"}

    def test_sanitized_keys_behind_a_ref_map_back_to_wire_names(self):
        # Providers reject keys like "first name"; the sanitizer renames them in $defs too.
        schema = {
            "name": "test_tool",
            "description": "test",
            "parameters": {
                "type": "object",
                "$defs": {"Person": {"type": "object", "properties": {"first name": {"type": "string"}}}},
                "properties": {"person": {"$ref": "#/$defs/Person"}},
            },
        }
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"person": {"first_name": "Ada"}})
            assert result == {"person": {"first name": "Ada"}}

    def test_array_item_referenced_via_defs_is_parsed(self):
        schema = {
            "name": "test_tool",
            "description": "test",
            "parameters": {
                "type": "object",
                "$defs": {
                    "Address": {
                        "type": "object",
                        "properties": {"street": {"type": "string"}},
                    },
                },
                "properties": {
                    "addresses": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Address"},
                    },
                },
            },
        }
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            args = {"addresses": ['{"street": "Alpha Ave"}']}
            result = coerce_tool_args("test_tool", args)
            assert result["addresses"] == [{"street": "Alpha Ave"}]

    def test_external_and_cyclic_refs_are_preserved(self):
        for ref, definitions in (
            ("https://example.com/address.json", {}),
            ("#/$defs/Address", {"Address": {"$ref": "#/$defs/Address"}}),
        ):
            schema = {
                "name": "test_tool",
                "description": "test",
                "parameters": {
                    "type": "object",
                    "$defs": definitions,
                    "properties": {"address": {"$ref": ref}},
                },
            }
            with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
                raw = '{"street": "Alpha Ave"}'
                result = coerce_tool_args("test_tool", {"address": raw})
                assert result["address"] == raw


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


# ── Parameter-name alias normalization ────────────────────────────────────


class TestParamAliasNormalization:
    """Models trained on other agent ecosystems (e.g. Claude Code's `file_path`)
    emit alias keys for tools whose schema uses a different name. An unknown key
    is silently dropped and the missing required param falls back to its default
    (for read_file: the session cwd — a directory), producing a confusing
    "not a regular file" error. Alias normalization renames known aliases to
    their canonical schema names before dispatch."""

    def _read_file_schema(self):
        return {
            "type": "object",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        }

    def test_file_path_alias_is_renamed_to_path(self):
        with patch("tools.arg_coercion.registry.get_schema", return_value=self._read_file_schema()):
            args = {"file_path": "/tmp/config.yaml", "limit": 5}
            result = coerce_tool_args("read_file", args)
            assert result["path"] == "/tmp/config.yaml"
            assert "file_path" not in result
            assert result["limit"] == 5

    def test_native_file_path_schema_is_untouched(self):
        """A tool whose schema genuinely uses `file_path` must not be rewritten."""
        schema = {
            "type": "object",
            "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
        }
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            args = {"file_path": "x"}
            result = coerce_tool_args("some_tool", args)
            assert result == {"file_path": "x"}

    def test_canonical_key_takes_precedence(self):
        """When both the alias and the canonical key are present, keep both as-is."""
        with patch("tools.arg_coercion.registry.get_schema", return_value=self._read_file_schema()):
            args = {"file_path": "a", "path": "b"}
            result = coerce_tool_args("read_file", args)
            assert result == {"file_path": "a", "path": "b"}

    def test_no_alias_without_canonical_in_schema(self):
        """No rewrite when the schema has neither the alias nor the canonical name."""
        schema = {
            "type": "object",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
        }
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            args = {"file_path": "x"}
            result = coerce_tool_args("some_tool", args)
            assert result == {"file_path": "x"}
