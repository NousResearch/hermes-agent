"""Tests for tool argument type coercion.

When LLMs return tool call arguments, they frequently put numbers as strings
("42" instead of 42) and booleans as strings ("true" instead of true).
coerce_tool_args() fixes these type mismatches by comparing argument values
against the tool's JSON Schema before dispatch.
"""

from unittest.mock import patch

from model_tools import (
    coerce_tool_args,
    project_tool_args,
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
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"limit": "10"}
            result = coerce_tool_args("test_tool", args)
            assert result["limit"] == 10
            assert isinstance(result["limit"], int)




    def test_leaves_already_correct_types(self):
        schema = self._mock_schema({"limit": {"type": "integer"}})
        with patch("model_tools.registry.get_schema", return_value=schema):
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
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"items": ['{"id": "1", "content": "x"}']}
            result = coerce_tool_args("test_tool", args)
            assert result["items"] == [{"id": "1", "content": "x"}]


    def test_string_subfield_with_json_content_preserved(self):
        """A string-typed sub-field whose value looks like JSON must NOT be parsed."""
        schema = self._array_of_objects_schema()
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"items": [{"id": "1", "content": '{"not": "parsed"}'}]}
            result = coerce_tool_args("test_tool", args)
            assert result["items"][0]["content"] == '{"not": "parsed"}'



    def test_real_todo_schema_element_strings(self):
        """Against the real todo schema from the registry."""
        import json as _json
        args = {"todos": [_json.dumps({"id": "1", "content": "x", "status": "pending"})]}
        result = coerce_tool_args("todo", args)
        assert result["todos"][0] == {"id": "1", "content": "x", "status": "pending"}


# ── project_tool_args: unknown-argument stripping (SECURITY-CLASS-227117ae016de6a5) ─


class TestProjectToolArgs:
    """Tests for project_tool_args — strips arguments not in the tool's schema."""

    def _mock_schema(self, properties, additional_properties=None):
        """Build a minimal tool schema with the given properties."""
        schema = {
            "name": "test_tool",
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        }
        if additional_properties is not None:
            schema["parameters"]["additionalProperties"] = additional_properties
        return schema

    def test_strips_unknown_arg(self):
        """Unknown arguments are removed."""
        schema = self._mock_schema({"command": {"type": "string"}}, additional_properties=False)
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"command": "ls", "force": True}
            result = project_tool_args("test_tool", args)
        assert "force" not in result
        assert result["command"] == "ls"

    def test_preserves_declared_args(self):
        """All declared arguments are kept."""
        schema = self._mock_schema({
            "command": {"type": "string"},
            "timeout": {"type": "integer"},
        })
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"command": "ls", "timeout": 30}
            result = project_tool_args("test_tool", args)
        assert result == {"command": "ls", "timeout": 30}

    def test_additional_properties_true_keeps_unknown(self):
        """Schemas with additionalProperties: true preserve unknown args."""
        schema = self._mock_schema(
            {"method": {"type": "string"}},
            additional_properties=True,
        )
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"method": "Page.navigate", "extra": "kept"}
            result = project_tool_args("test_tool", args)
        assert result == {"method": "Page.navigate", "extra": "kept"}

    def test_additional_properties_false_strips_unknown(self):
        """Schemas with additionalProperties: false strip unknown args."""
        schema = self._mock_schema(
            {"prompt": {"type": "string"}},
            additional_properties=False,
        )
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"prompt": "hello", "sneaky": True}
            result = project_tool_args("test_tool", args)
        assert "sneaky" not in result
        assert result == {"prompt": "hello"}

    def test_empty_args(self):
        assert project_tool_args("test_tool", {}) == {}

    def test_no_schema_returns_args(self):
        """When no schema is registered, args are returned unchanged."""
        with patch("model_tools.registry.get_schema", return_value=None):
            args = {"command": "ls", "force": True}
            result = project_tool_args("unknown_tool", args)
        assert result == args

    def test_no_properties_returns_args(self):
        """When schema has no properties dict, args are returned unchanged."""
        schema = {"name": "test_tool", "parameters": {"type": "object"}}
        with patch("model_tools.registry.get_schema", return_value=schema):
            args = {"command": "ls"}
            result = project_tool_args("test_tool", args)
        assert result == args

    def test_real_terminal_force_is_stripped(self):
        """The terminal tool's hidden ``force`` parameter is stripped.

        Regression test for SECURITY-CLASS-227117ae016de6a5: the terminal
        schema does not declare ``force``, so a model-injected ``force=true``
        must not survive projection.
        """
        args = {"command": "rm -rf /", "force": True}
        result = project_tool_args("terminal", args)
        assert "force" not in result
        assert result["command"] == "rm -rf /"

    def test_real_terminal_declared_args_preserved(self):
        """All declared terminal arguments survive projection."""
        args = {
            "command": "echo hi",
            "background": True,
            "timeout": 60,
            "workdir": "/tmp",
            "pty": False,
            "notify_on_complete": True,
        }
        result = project_tool_args("terminal", args)
        assert result == args


class TestForceBypassIntegration:
    """End-to-end: ``force=true`` cannot bypass dangerous-command approval.

    Exercises the full dispatch path through ``handle_function_call`` to
    verify that ``force`` is projected away before the terminal handler runs.
    """

    def test_force_stripped_before_handler(self):
        """A model-injected ``force=true`` must not reach the terminal handler.

        The handler receives the projected args dict (without ``force``), so
        the dangerous-command check runs normally and the command is blocked
        or sent to approval — never silently executed.
        """
        from unittest.mock import MagicMock
        from model_tools import handle_function_call

        captured_args = {}

        def _capture_handler(args, **kwargs):
            captured_args.update(args)
            return '{"output": "", "exit_code": 0, "error": ""}'

        mock_entry = MagicMock()
        mock_entry.handler = _capture_handler
        mock_entry.is_async = False
        mock_entry.schema = {
            "name": "terminal",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "background": {"type": "boolean"},
                    "timeout": {"type": "integer"},
                    "workdir": {"type": "string"},
                    "pty": {"type": "boolean"},
                    "notify_on_complete": {"type": "boolean"},
                    "watch_patterns": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        }
        mock_entry.max_result_size_chars = None
        mock_entry.name = "terminal"
        mock_entry.toolset = "terminal"

        with patch("model_tools.registry.get_entry", return_value=mock_entry), \
             patch("model_tools.registry.get_schema", return_value=mock_entry.schema), \
             patch("model_tools._emit_post_tool_call_hook"), \
             patch("hermes_cli.lifecycle.has_hook", return_value=False), \
             patch("hermes_cli.middleware.run_tool_execution_middleware") as _mw_mock, \
             patch("tools.file_tools.notify_other_tool_call"):
            _mw_mock.side_effect = lambda name, args, dispatch, **kw: dispatch(args)
            handle_function_call(
                function_name="terminal",
                function_args={"command": "rm -rf /", "force": True},
                skip_pre_tool_call_hook=True,
                skip_tool_request_middleware=True,
            )

        assert "force" not in captured_args, (
            "force=true survived projection and reached the handler — "
            "dangerous-command bypass is possible"
        )
        assert captured_args.get("command") == "rm -rf /"
