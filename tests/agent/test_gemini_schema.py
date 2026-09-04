"""Tests for agent.gemini_schema — OpenAI→Gemini tool parameter translation."""

import pytest

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


class TestArrayTypeNormalization:
    """JSON Schema allows an array ``type``; Gemini's ``Schema`` accepts one string.

    The enum-compatibility check evaluated ``[...] in {...}`` on that list and raised
    ``TypeError: unhashable type: 'list'``, aborting translation of the WHOLE tool
    catalog, not just the offending tool. #55645
    """

    def test_nullable_form_collapses_and_keeps_the_enum(self):
        cleaned = sanitize_gemini_schema({"type": ["string", "null"], "enum": ["low", "high"]})
        assert cleaned["type"] == "string"
        assert cleaned["nullable"] is True
        assert cleaned["enum"] == ["low", "high"]

    def test_nullable_form_without_enum(self):
        cleaned = sanitize_gemini_schema({"type": ["integer", "null"]})
        assert cleaned["type"] == "integer"
        assert cleaned["nullable"] is True

    def test_real_union_becomes_anyof_with_every_branch_kept(self):
        """No branch may be dropped. ``tools/schema_sanitizer._normalize_type_array``
        is the canonical behaviour and is reused here rather than reimplemented."""
        cleaned = sanitize_gemini_schema({"type": ["string", "integer"]})
        assert "type" not in cleaned
        assert cleaned["anyOf"] == [{"type": "string"}, {"type": "integer"}]
        assert "nullable" not in cleaned

    def test_three_way_union_keeps_all_three(self):
        cleaned = sanitize_gemini_schema({"type": ["string", "integer", "boolean"]})
        assert cleaned["anyOf"] == [
            {"type": "string"}, {"type": "integer"}, {"type": "boolean"}]

    @pytest.mark.parametrize("schema", [
        {"nullable": False, "type": ["string", "null"]},   # nullable emitted first
        {"type": ["string", "null"], "nullable": False},   # nullable emitted last
    ])
    def test_derived_nullable_beats_an_input_nullable_either_order(self, schema):
        """The array says null is permitted, so an input ``nullable: false`` is wrong.

        Deriving inside the key loop made this order-dependent: whichever key the
        producer emitted last won.
        """
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["nullable"] is True
        assert cleaned["type"] == "string"

    def test_all_null_array_becomes_the_null_type(self):
        assert sanitize_gemini_schema({"type": ["null"]})["type"] == "null"

    def test_empty_type_array_falls_back_to_object(self):
        assert sanitize_gemini_schema({"type": []})["type"] == "object"

    def test_integer_union_still_stringifies_its_enum(self):
        cleaned = sanitize_gemini_schema({"type": ["integer", "null"], "enum": [1, 2, 3]})
        assert cleaned["type"] == "integer"
        assert cleaned["enum"] == ["1", "2", "3"]

    def test_nested_property_is_normalized_too(self):
        cleaned = sanitize_gemini_schema({
            "type": "object",
            "properties": {"color": {"type": ["string", "null"], "enum": ["red", "green"]}},
        })
        color = cleaned["properties"]["color"]
        assert color["type"] == "string" and color["nullable"] is True
        assert color["enum"] == ["red", "green"]

    def test_plain_string_type_is_untouched(self):
        """Control: the non-array path must be unchanged."""
        cleaned = sanitize_gemini_schema({"type": "string", "enum": ["a"]})
        assert cleaned == {"type": "string", "enum": ["a"]}
