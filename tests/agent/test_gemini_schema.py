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

    def test_drops_items_on_non_array_scalar(self):
        """Gemini: ``items`` is legal only when ``type == array``.

        Regression for ClickUp's ``clickup_filter_tasks``: its ``value`` field
        ships ``{type: string, items: {type: string}, description: ...}`` and
        triggered Gemini HTTP 400 INVALID_ARGUMENT
        "field predicate failed: $type == Type.ARRAY", rejecting the whole
        tool catalog on every request.
        """
        schema = {
            "type": "string",
            "items": {"type": "string"},
            "description": "Value; for RANGE provide an array of two strings.",
        }
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "string"
        assert "items" not in cleaned
        # description survives so the model still learns about the array case
        assert "RANGE" in cleaned["description"]

    def test_drops_array_only_constraints_on_non_array(self):
        """minItems / maxItems share the array-only predicate."""
        schema = {
            "type": "string",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 2,
        }
        cleaned = sanitize_gemini_schema(schema)
        assert "items" not in cleaned
        assert "minItems" not in cleaned
        assert "maxItems" not in cleaned

    def test_keeps_items_on_real_array(self):
        """A legitimate array schema must keep its ``items``."""
        schema = {"type": "array", "items": {"type": "string"}}
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "array"
        assert cleaned["items"] == {"type": "string"}

    def test_pins_array_type_when_items_present_without_type(self):
        """``items`` with no ``type`` is an array that forgot to say so."""
        schema = {"items": {"type": "string"}, "description": "tags"}
        cleaned = sanitize_gemini_schema(schema)
        assert cleaned["type"] == "array"
        assert cleaned["items"] == {"type": "string"}

    def test_clickup_filter_tasks_value_no_longer_trips_gemini(self):
        """End-to-end: the exact ``custom_fields`` shape rejected in prod."""
        params = {
            "type": "object",
            "properties": {
                "custom_fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_id": {"type": "string"},
                            "operator": {"type": "string"},
                            "value": {
                                "type": "string",
                                "items": {"type": "string"},
                                "description": "Value; RANGE takes two strings.",
                            },
                        },
                        "required": ["field_id", "operator"],
                    },
                },
            },
        }
        cleaned = sanitize_gemini_tool_parameters(params)
        value = cleaned["properties"]["custom_fields"]["items"]["properties"][
            "value"
        ]
        assert value["type"] == "string"
        assert "items" not in value
        # The outer custom_fields array keeps its items (it really is an array).
        assert cleaned["properties"]["custom_fields"]["type"] == "array"
        assert "items" in cleaned["properties"]["custom_fields"]


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
