"""Tests for agent.gemini_schema — OpenAI→Gemini tool parameter translation."""

from agent.gemini_schema import (
    sanitize_gemini_schema,
    sanitize_gemini_tool_parameters,
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


    def test_stringifies_integer_enum_to_satisfy_gemini(self):
        """Gemini rejects numeric enum metadata unless values are strings.

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
        assert cleaned["enum"] == ["60", "1440", "4320", "10080"]
        # Description remains useful model guidance.
        assert cleaned["description"].startswith("Minutes")





    def test_stringifies_nested_integer_enum_inside_properties(self):
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
        # Integer enum is retained as Gemini-compatible string metadata...
        assert props["auto_archive_duration"]["type"] == "integer"
        assert props["auto_archive_duration"]["enum"] == ["60", "1440", "4320", "10080"]
        # ...but the sibling string enum is preserved.
        assert props["status"]["enum"] == ["active", "archived"]



    def test_non_dict_input_returns_empty(self):
        assert sanitize_gemini_schema(None) == {}
        assert sanitize_gemini_schema("not a schema") == {}
        assert sanitize_gemini_schema([1, 2, 3]) == {}


class TestRequiredPropertyPruning:
    """Gemini rejects ``required`` names missing from the node's ``properties``.

    Regression for the Kilo-Org/kilocode#11955 bug class: MCP servers (e.g.
    the GitHub remote MCP) emit array item schemas whose ``required`` lists
    reference properties that don't exist in the same node — Google fails the
    entire GenerateContentRequest with HTTP 400 "property is not defined".
    """



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


    def test_valid_required_untouched(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["required"] == ["a", "b"]


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
        # The field that triggered the Gemini 400 is now string metadata.
        assert aad["enum"] == ["60", "1440", "4320", "10080"]
        # Type + description survive so the model still knows what to send.
        assert aad["type"] == "integer"
        assert "1440" in aad["description"]
        # And the string-enum sibling is untouched.
        assert cleaned["properties"]["action"]["enum"] == ["create_thread"]


class TestNoTypelessNodes:
    """Gemini's FunctionDeclaration validator rejects a Schema with no type.

    A property whose entire definition is an unsupported keyword ($ref to a
    $defs entry, a const-only node) lost every key to the allow-list and
    collapsed to {}. One such property failed the whole GenerateContentRequest
    with HTTP 400 before any token was produced.
    """

    def test_ref_property_degrades_to_a_typed_node(self):
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {"cfg": {"$ref": "#/$defs/Cfg"}},
                "required": ["cfg"],
            }
        )
        cfg = out["properties"]["cfg"]
        assert cfg != {}
        assert cfg["type"] == "object"
        assert cfg["properties"] == {}

    def test_const_only_property_degrades_to_a_typed_node(self):
        out = sanitize_gemini_schema(
            {"type": "object", "properties": {"k": {"const": "fixed"}}}
        )
        assert "type" in out["properties"]["k"]

    def test_oneof_is_translated_to_anyof(self):
        """oneOf is not in Gemini's subset; its branches must not be dropped."""
        out = sanitize_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "v": {"oneOf": [{"type": "string"}, {"type": "integer"}]}
                },
                "required": ["v"],
            }
        )
        assert out["properties"]["v"]["anyOf"] == [
            {"type": "string"},
            {"type": "integer"},
        ]

    def test_allof_is_translated_to_anyof(self):
        out = sanitize_gemini_schema(
            {"type": "object", "properties": {"v": {"allOf": [{"type": "string"}]}}}
        )
        assert out["properties"]["v"]["anyOf"] == [{"type": "string"}]

    def test_existing_anyof_wins_over_oneof(self):
        out = sanitize_gemini_schema(
            {
                "anyOf": [{"type": "string"}],
                "oneOf": [{"type": "integer"}],
            }
        )
        assert out["anyOf"] == [{"type": "string"}]

    def test_ordinary_schema_is_unchanged(self):
        schema = {
            "type": "object",
            "properties": {"p": {"type": "string", "description": "d"}},
            "required": ["p"],
        }
        assert sanitize_gemini_schema(schema) == schema

    def test_array_items_unchanged(self):
        schema = {"type": "array", "items": {"type": "string"}}
        assert sanitize_gemini_schema(schema) == schema
