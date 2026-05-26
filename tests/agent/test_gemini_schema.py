"""Tests for agent.gemini_schema — OpenAI→Gemini tool parameter translation."""

from agent.gemini_schema import (
    sanitize_gemini_schema,
    sanitize_gemini_tool_parameters,
    sanitize_gemini_tools,
)


class TestSanitizeGeminiSchema:
    def test_strips_unknown_top_level_keys(self):
        """$schema / additionalProperties etc. must not reach Gemini."""
        schema = {
            "type": "object",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "additionalProperties": False,
            "properties": {"foo": {"type": "string"}},
        }
        cleaned = sanitize_gemini_schema(schema)
        assert "$schema" not in cleaned
        assert "additionalProperties" not in cleaned
        assert cleaned["type"] == "object"
        assert cleaned["properties"] == {"foo": {"type": "string"}}

    def test_preserves_string_enums(self):
        """String-valued enums are valid for Gemini and must pass through."""
        schema = {"type": "string", "enum": ["pending", "done", "cancelled"]}
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "string"
        assert cleaned["enum"] == ["pending", "done", "cancelled"]

    def test_drops_integer_enum_to_satisfy_gemini(self):
        """Gemini rejects int-typed enums; the sanitizer must drop the enum.

        Regression for the Discord tool's ``auto_archive_duration``:
        ``{type: integer, enum: [60, 1440, 4320, 10080]}`` caused
        Gemini HTTP 400 INVALID_ARGUMENT
        "Invalid value ... (TYPE_STRING), 60" on every request that
        shipped the full tool catalog to generativelanguage.googleapis.com.
        """
        schema = {
            "type": "integer",
            "enum": [60, 1440, 4320, 10080],
            "description": "Minutes (60, 1440, 4320, 10080).",
        }
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "integer"
        assert "enum" not in cleaned
        # description must survive so the model still sees the allowed values
        assert cleaned["description"].startswith("Minutes")

    def test_drops_number_enum(self):
        """Same rule applies to ``type: number``."""
        schema = {"type": "number", "enum": [0.5, 1.0, 2.0]}
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "number"
        assert "enum" not in cleaned

    def test_drops_boolean_enum(self):
        """And to ``type: boolean`` (Gemini rejects non-string entries)."""
        schema = {"type": "boolean", "enum": [True, False]}
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "boolean"
        assert "enum" not in cleaned

    def test_keeps_string_enum_even_when_numeric_values_coexist_as_strings(self):
        """Stringified-numeric enums ARE valid for Gemini; don't drop them."""
        schema = {"type": "string", "enum": ["60", "1440", "4320", "10080"]}
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["enum"] == ["60", "1440", "4320", "10080"]

    def test_drops_nested_integer_enum_inside_properties(self):
        """The fix must apply recursively — the Discord case is nested."""
        schema = {
            "type": "object",
            "properties": {
                "auto_archive_duration": {
                    "type": "integer",
                    "enum": [60, 1440, 4320, 10080],
                    "description": "Thread archive duration in minutes.",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "archived"],
                },
            },
        }
        cleaned = sanitize_gemini_schema(schema)
        props = cleaned["properties"]
        # Integer enum is dropped...
        assert props["auto_archive_duration"]["type"] == "integer"
        assert "enum" not in props["auto_archive_duration"]
        # ...but the sibling string enum is preserved.
        assert props["status"]["enum"] == ["active", "archived"]

    def test_drops_integer_enum_inside_array_items(self):
        """Array item schemas recurse through ``items``."""
        schema = {
            "type": "array",
            "items": {"type": "integer", "enum": [1, 2, 3]},
        }
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["items"]["type"] == "integer"
        assert "enum" not in cleaned["items"]

    def test_non_dict_input_returns_empty(self):
        assert sanitize_gemini_schema(None) == {}
        assert sanitize_gemini_schema("not a schema") == {}
        assert sanitize_gemini_schema([1, 2, 3]) == {}

    # --- union type array collapsing ---

    def test_nullable_string_shorthand_collapses_to_string_nullable(self):
        """["string", "null"] → type: string, nullable: true."""
        schema = {"type": ["string", "null"]}
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "string"
        assert cleaned["nullable"] is True

    def test_nullable_integer_collapses_correctly(self):
        """["integer", "null"] → type: integer, nullable: true."""
        schema = {"type": ["integer", "null"]}
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "integer"
        assert cleaned["nullable"] is True

    def test_union_without_null_picks_string_when_present(self):
        """["number", "string"] → type: string (most permissive)."""
        schema = {"type": ["number", "string"]}
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "string"
        assert "nullable" not in cleaned

    def test_union_without_null_picks_first_when_no_string(self):
        """["integer", "number"] → type: integer (first in list)."""
        schema = {"type": ["integer", "number"]}
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "integer"
        assert "nullable" not in cleaned

    def test_null_only_type_falls_back_to_string(self):
        """["null"] → type: string (degenerate case)."""
        schema = {"type": ["null"]}
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "string"

    def test_array_type_with_integer_enum_does_not_raise_and_drops_enum(self):
        """An array type paired with a non-string enum must not raise.

        The enum rule does a ``set`` membership test on ``type``; a list type
        there raised ``TypeError: unhashable type: 'list'`` before the array
        collapse was reordered ahead of it. After collapsing to ``integer`` the
        non-string enum is dropped per Gemini's rule. (repro from #55643)
        """
        schema = {"type": ["integer", "null"], "enum": [1, 2, 3]}
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "integer"
        assert cleaned["nullable"] is True
        assert "enum" not in cleaned

    def test_array_type_with_string_enum_preserves_enum(self):
        """A nullable string union keeps a valid string enum after collapsing."""
        schema = {"type": ["string", "null"], "enum": ["a", "b"]}
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "string"
        assert cleaned["nullable"] is True
        assert cleaned["enum"] == ["a", "b"]

    def test_union_type_array_inside_properties(self):
        """The fix recurses into properties — the lcm_grep repro case."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "time_from": {"type": ["number", "string"]},
            },
            "required": ["query"],
        }
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["properties"]["query"]["type"] == "string"
        assert cleaned["properties"]["time_from"]["type"] == "string"

    def test_scalar_type_string_unchanged(self):
        """A plain string ``type`` must not be touched."""
        schema = {"type": "string"}
        assert sanitize_gemini_schema(schema)["type"] == "string"


class TestRequiredPropertyPruning:
    """Gemini rejects ``required`` names missing from the node's ``properties``.

    Regression for the Kilo-Org/kilocode#11955 bug class: MCP servers (e.g.
    the GitHub remote MCP) emit array item schemas whose ``required`` lists
    reference properties that don't exist in the same node — Google fails the
    entire GenerateContentRequest with HTTP 400 "property is not defined".
    """

    def test_drops_required_when_node_has_no_properties(self):
        schema = {"type": "object", "required": ["a", "b"]}
        cleaned = sanitize_gemini_schema(schema)
        assert "required" not in cleaned

    def test_filters_ghost_required_entries(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x", "ghost"],
        }
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["required"] == ["x"]

    def test_prunes_inside_array_items(self):
        """The exact shape from the GitHub MCP report — nested in items."""
        schema = {
            "type": "object",
            "properties": {
                "issue_fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["field_id", "value"],
                    },
                },
            },
            "required": ["issue_fields"],
        }
        cleaned = sanitize_gemini_schema(schema)
        items = cleaned["properties"]["issue_fields"]["items"]
        assert "required" not in items
        # Top-level required is valid and survives.
        assert cleaned["required"] == ["issue_fields"]

    def test_prunes_node_without_explicit_type(self):
        """Nodes carrying properties+required but no ``type`` key still prune."""
        schema = {
            "properties": {"x": {"type": "string"}},
            "required": ["x", "ghost"],
        }
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["required"] == ["x"]

    def test_valid_required_untouched(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["required"] == ["a", "b"]

    def test_drops_non_string_required_entries(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a", 42, None],
        }
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["required"] == ["a"]

    def test_prunes_inside_anyof_branches(self):
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x", "ghost"],
                },
                {"type": "object", "required": ["orphan"]},
            ]
        }
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["anyOf"][0]["required"] == ["x"]
        assert "required" not in cleaned["anyOf"][1]


class TestSanitizeGeminiToolParameters:
    def test_empty_parameters_return_valid_object_schema(self):
        """Gemini requires ``parameters`` to be a valid object schema."""
        cleaned = sanitize_gemini_tool_parameters({})
        assert cleaned == {"type": "object", "properties": {}}

    def test_discord_create_thread_parameters_no_longer_trip_gemini(self):
        """End-to-end regression: the exact shape that was rejected in prod."""
        params = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create_thread"]},
                "auto_archive_duration": {
                    "type": "integer",
                    "enum": [60, 1440, 4320, 10080],
                    "description": "Thread archive duration in minutes "
                    "(create_thread, default 1440).",
                },
            },
            "required": ["action"],
        }
        cleaned = sanitize_gemini_tool_parameters(params)
        aad = cleaned["properties"]["auto_archive_duration"]
        # The field that triggered the Gemini 400 is gone.
        assert "enum" not in aad
        # Type + description survive so the model still knows what to send.
        assert aad["type"] == "integer"
        assert "1440" in aad["description"]
        # And the string-enum sibling is untouched.
        assert cleaned["properties"]["action"]["enum"] == ["create_thread"]


class TestSanitizeGeminiTools:
    """Tests for the OpenAI-format tool-list wrapper used by chat_completions."""

    def test_lcm_grep_union_type_fixed(self):
        """End-to-end: the exact schema reported in #30676 repro 2."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lcm_grep",
                    "description": "d",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "time_from": {"type": ["number", "string"]},
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
        result = sanitize_gemini_tools(tools)
        props = result[0]["function"]["parameters"]["properties"]
        assert props["query"]["type"] == "string"
        assert props["time_from"]["type"] == "string"
        assert "nullable" not in props["time_from"]

    def test_discord_integer_enum_fixed(self):
        """End-to-end: the exact schema reported in #30676 repro 1."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "discord",
                    "description": "d",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["create_thread"]},
                            "auto_archive_duration": {
                                "type": "integer",
                                "enum": [60, 1440, 4320, 10080],
                            },
                        },
                        "required": ["action"],
                    },
                },
            }
        ]
        result = sanitize_gemini_tools(tools)
        props = result[0]["function"]["parameters"]["properties"]
        assert "enum" not in props["auto_archive_duration"]
        assert props["action"]["enum"] == ["create_thread"]

    def test_non_dict_tool_passed_through_unchanged(self):
        tools = ["not-a-tool", 42]
        assert sanitize_gemini_tools(tools) == ["not-a-tool", 42]

    def test_tool_without_function_key_passed_through(self):
        tool = {"type": "function"}
        assert sanitize_gemini_tools([tool]) == [tool]

    def test_original_tool_list_not_mutated(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {"type": "object", "properties": {"x": {"type": ["string", "null"]}}},
                },
            }
        ]
        original_type = tools[0]["function"]["parameters"]["properties"]["x"]["type"]
        sanitize_gemini_tools(tools)
        assert tools[0]["function"]["parameters"]["properties"]["x"]["type"] == original_type
