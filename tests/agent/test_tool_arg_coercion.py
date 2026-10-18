"""Tests for tool argument type coercion.

When LLMs return tool call arguments, they frequently put numbers as strings
("42" instead of 42) and booleans as strings ("true" instead of true).
coerce_tool_args() fixes these type mismatches by comparing argument values
against the tool's JSON Schema before dispatch.
"""

import json
from unittest.mock import patch

import pytest

import model_tools
from tools.arg_coercion import (
    coerce_tool_args,
    ToolArgUnparseableError,
    unparseable_container_message,
    _coerce_value,
    _coerce_number,
    _coerce_boolean,
    _schema_accepts_kind,
    _normalize_json_strings_for_schema,
)
from tools.registry import registry


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

    def test_truncated_array_string_is_refused(self):
        """A JSON-encoded array string that fails to parse raises instead of wrapping.

        Regression: a truncated ``operations`` value was previously wrapped into a
        one-element list, so the tool failed with a misleading inner error and the
        model burned turns recovering.
        """
        schema = self._mock_schema({"operations": {"type": "array"}})
        truncated = '["patch", {"action": "add", "content": "cut off mid-str'
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            with pytest.raises(ToolArgUnparseableError) as exc_info:
                coerce_tool_args("test_tool", {"operations": truncated})
        exc = exc_info.value
        assert exc.tool_name == "test_tool"
        assert exc.param == "operations"
        assert exc.expected_kind == "array"
        assert "Unterminated string" in exc.detail
        assert exc.preview == truncated[:120]

    def test_truncated_object_string_is_refused(self):
        """The object path refuses too (it used to hand the raw string to the tool)."""
        schema = self._mock_schema({"payload": {"type": "object"}})
        corrupt = '{"path": "C:\\new", "mode": "add'
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            with pytest.raises(ToolArgUnparseableError) as exc_info:
                coerce_tool_args("test_tool", {"payload": corrupt})
        exc = exc_info.value
        assert exc.param == "payload"
        assert exc.expected_kind == "object"
        assert "Unterminated string" in exc.detail

    def test_invalid_escape_in_array_string_is_refused(self):
        schema = self._mock_schema({"operations": {"type": "array"}})
        corrupt = r'["read_file \qhome", "x"]'
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            with pytest.raises(ToolArgUnparseableError) as exc_info:
                coerce_tool_args("test_tool", {"operations": corrupt})
        assert "Invalid \\escape" in exc_info.value.detail

    def test_bare_plain_string_still_wrapped(self):
        """Regression guard: plain-text (non-bracket) value for an array param wraps."""
        schema = self._mock_schema({"tags": {"type": "array"}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            assert coerce_tool_args("test_tool", {"tags": "urgent"}) == {"tags": ["urgent"]}

    def test_wrong_container_type_not_consumed(self):
        """Regression guard: valid JSON of the WRONG container type is not coerced.

        Object param holding a JSON array string: unchanged pass-through (the
        original behaviour). Array param holding a JSON object string: the array
        branch keeps its pre-existing bare-value wrapping (``[value]``) — only
        bracket-prefixed *unparseable* strings are refused. See report note.
        """
        with patch("tools.arg_coercion.registry.get_schema",
                   return_value=self._mock_schema({"payload": {"type": "object"}})):
            assert coerce_tool_args("test_tool", {"payload": '[1, 2]'}) == {"payload": '[1, 2]'}
        with patch("tools.arg_coercion.registry.get_schema",
                   return_value=self._mock_schema({"items": {"type": "array"}})):
            assert coerce_tool_args("test_tool", {"items": '{"a": 1}'}) == {"items": ['{"a": 1}']}

    def test_union_array_null_truncated_string_is_refused(self):
        """Union coverage: anyOf [array, null] refuses a truncated array string."""
        schema = self._mock_schema({"items": {"anyOf": [{"type": "array"}, {"type": "null"}]}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            with pytest.raises(ToolArgUnparseableError) as exc_info:
                coerce_tool_args("test_tool", {"items": '["a", "b'})
        assert exc_info.value.expected_kind == "array"

    def test_union_array_string_keeps_non_json_text(self):
        """Union coverage: anyOf [array, string] keeps raw bracket text — the string
        branch is schema-valid, so refusing it would break legitimate calls."""
        schema = self._mock_schema({"tags": {"anyOf": [{"type": "array"}, {"type": "string"}]}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"tags": "[INFO] something"})
        assert result["tags"] == "[INFO] something"

    def test_non_container_param_untouched(self):
        """Regression guard: a ``string`` param keeps a bracket-prefixed value verbatim."""
        schema = self._mock_schema({"command": {"type": "string"}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            args = {"command": "[INFO] something"}
            assert coerce_tool_args("test_tool", args) == args

    def test_no_null_anyof_union_refused(self):
        """Review gap 1: a type-less ``anyOf`` (array|object) must not bypass refusal.

        Pydantic model fields serialize as bare ``anyOf`` with no top-level
        ``type``; the ``not expected`` early return previously let both bracket
        prefixes dispatch unchanged.
        """
        schema = self._mock_schema({"payload": {"anyOf": [{"type": "array"}, {"type": "object"}]}})
        for corrupt in ('["cut off', '{"cut off'):
            with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
                with pytest.raises(ToolArgUnparseableError):
                    coerce_tool_args("test_tool", {"payload": corrupt})

    def test_string_accepting_union_still_passes_through(self):
        """anyOf(array|string) with corrupt bracket text stays a valid string member."""
        schema = self._mock_schema({"opts": {"anyOf": [{"type": "array"}, {"type": "string"}]}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            assert coerce_tool_args("test_tool", {"opts": '["cut off'}) == {"opts": '["cut off'}

    def test_native_array_malformed_element_refused(self):
        """Review gap 2: corrupt JSON-encoded element inside a native list is refused."""
        schema = self._mock_schema({"items": {"type": "array", "items": {"type": "object"}}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            with pytest.raises(ToolArgUnparseableError) as exc_info:
                coerce_tool_args("test_tool", {"items": ['{"note": "cut off']})
        assert exc_info.value.param == "items[0]"
        assert exc_info.value.expected_kind == "object"

    def test_nested_dict_field_refused_at_dotted_path(self):
        """Corrupt values in nested objects are refused with the dotted path."""
        schema = self._mock_schema({
            "outer": {"type": "object", "properties": {"inner": {"type": "object"}}},
        })
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            with pytest.raises(ToolArgUnparseableError) as exc_info:
                coerce_tool_args("test_tool", {"outer": {"inner": '{"a": '}})
        assert exc_info.value.param == "outer.inner"

    def test_outer_encoded_array_recursively_normalized(self):
        """Review gap 2 (second half): an outer JSON-encoded array string is parsed
        AND its JSON-encoded elements normalized, not left as strings."""
        schema = self._mock_schema({"items": {"type": "array", "items": {"type": "object"}}})
        outer = json.dumps([json.dumps({"note": "ok"})])  # '["{\\"note\\": \\"ok\\"}"]'
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", {"items": outer})
        assert result == {"items": [{"note": "ok"}]}

    def test_nested_corrupt_string_is_refused(self):
        """Corruption nested inside a native container IS refused, at its path.

        (Formerly a documented limitation — a corrupt nested string passed
        through untouched; review flagged it, the normalizer now refuses.)
        """
        schema = self._mock_schema({"items": {"type": "array", "items": {"type": "object",
                                                                        "properties": {"note": {"type": "string"}}}}})
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            with pytest.raises(ToolArgUnparseableError) as exc_info:
                coerce_tool_args("test_tool", {"items": ['{"note": "cut off']})
        assert exc_info.value.param == "items[0]"

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


# ── Dispatch-level refusal of unparseable container args ───────────────────

_REFUSAL_SCHEMA = {
    "name": "probe_arg_refusal",
    "description": "test probe",
    "parameters": {
        "type": "object",
        "properties": {"operations": {"type": "array", "items": {"type": "object"}}},
    },
}
_TRUNCATED_OPERATIONS = '[{"action": "add", "content": "the stream was cut he'


class TestDispatchRefusal:
    """ToolArgUnparseableError is caught per call site and returned as a
    model-actionable tool error; the handler is never invoked."""

    def test_handle_function_call_refuses_and_fires_post_tool_call(self):
        invocations = []
        hook_calls = []

        def handler(args, **ctx):
            invocations.append(args)
            return json.dumps({"ok": True})

        registry.register("probe_arg_refusal", "test", _REFUSAL_SCHEMA, handler)
        try:
            with patch("hermes_cli.lifecycle.has_hook", return_value=True), \
                 patch("hermes_cli.lifecycle.invoke_hook",
                       side_effect=lambda name, **kw: hook_calls.append((name, kw)) or []):
                result = model_tools.handle_function_call(
                    "probe_arg_refusal",
                    {"operations": _TRUNCATED_OPERATIONS},
                    tool_call_id="call-refusal",
                    skip_pre_tool_call_hook=True,
                    skip_tool_request_middleware=True,
                    skip_tool_execution_middleware=True,
                )
            # Control: the same stub with VALID args must dispatch — otherwise the
            # "handler not invoked" assertion below would be vacuous.
            with patch("hermes_cli.lifecycle.has_hook", return_value=False):
                ok_result = model_tools.handle_function_call(
                    "probe_arg_refusal",
                    {"operations": [{"action": "add"}]},
                    skip_pre_tool_call_hook=True,
                    skip_tool_request_middleware=True,
                    skip_tool_execution_middleware=True,
                )
        finally:
            registry.deregister("probe_arg_refusal")

        assert invocations == [{"operations": [{"action": "add"}]}], \
            "handler must not run for the refused call but must run for the control"
        assert json.loads(ok_result) == {"ok": True}

        error = json.loads(result)["error"]
        assert "Re-issue" in error
        assert "'operations'" in error
        assert "truncated or corrupt JSON-encoded array" in error

        post = [kw for name, kw in hook_calls if name == "post_tool_call"]
        assert post, "post_tool_call must still fire on the coercion-failure path"
        assert post[-1]["tool_name"] == "probe_arg_refusal"
        assert post[-1]["status"] == "error"
        assert "Re-issue" in post[-1]["result"]
        assert post[-1]["tool_call_id"] == "call-refusal"

    def test_inline_executor_wrapper_surfaces_error_string(self):
        from agent.inline_tool_executors import InlineToolContext, _coerced

        invocations = []

        def fake_executor(agent, args, ctx):
            invocations.append(args)
            return json.dumps({"ok": True})

        wrapped = _coerced("probe_arg_refusal", fake_executor)
        with patch("tools.arg_coercion.registry.get_schema", return_value=_REFUSAL_SCHEMA):
            result = wrapped(None, {"operations": _TRUNCATED_OPERATIONS},
                             InlineToolContext(effective_task_id="task-1"))
        assert invocations == []
        assert "Re-issue" in json.loads(result)["error"]

    def test_tool_search_validation_marks_candidate_invalid(self):
        from tools.tool_search_validation import validate_deferred_call_args

        with patch("tools.arg_coercion.registry.get_schema", return_value=_REFUSAL_SCHEMA):
            error = validate_deferred_call_args(
                "probe_arg_refusal", {"operations": _TRUNCATED_OPERATIONS})
        assert error is not None, "corrupt container string must invalidate the candidate"
        parsed = json.loads(error)
        assert "Re-issue" in parsed["error"]
        assert "The tool was NOT invoked." in parsed["error"]
        assert parsed["constraint"] == "unparseable-json-string"
        assert parsed["path"] == "operations"

    def test_valid_container_string_still_validates_clean(self):
        """Control for the validator: a parseable JSON-encoded array passes through
        coercion and is not flagged by the refusal path."""
        from tools.tool_search_validation import validate_deferred_call_args

        with patch("tools.arg_coercion.registry.get_schema", return_value=_REFUSAL_SCHEMA):
            error = validate_deferred_call_args(
                "probe_arg_refusal", {"operations": '[{"action": "add"}]'})
        assert error is None or "unparseable-json-string" not in error
