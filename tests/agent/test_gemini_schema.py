"""Tests for agent.gemini_schema.sanitize_gemini_schema."""

from agent.gemini_schema import (
    sanitize_gemini_schema,
    sanitize_gemini_tool_parameters,
)


def test_integer_enum_is_stringified():
    """Gemini's Schema.enum only accepts TYPE_STRING values.

    Tools like discord_server's auto_archive_duration declare integer enums
    ([60, 1440, 4320, 10080]). Those would raise Gemini HTTP 400
    'Invalid value at enum[0] (TYPE_STRING)' if passed through unchanged.
    """
    schema = {
        "type": "integer",
        "enum": [60, 1440, 4320, 10080],
        "description": "Thread archive duration in minutes.",
    }
    cleaned = sanitize_gemini_schema(schema)
    assert cleaned["enum"] == ["60", "1440", "4320", "10080"]
    assert cleaned["type"] == "integer"


def test_string_enum_is_preserved():
    schema = {"type": "string", "enum": ["alpha", "beta", "gamma"]}
    cleaned = sanitize_gemini_schema(schema)
    assert cleaned["enum"] == ["alpha", "beta", "gamma"]


def test_mixed_enum_is_stringified():
    schema = {"enum": ["a", 1, 2.5, True]}
    cleaned = sanitize_gemini_schema(schema)
    assert cleaned["enum"] == ["a", "1", "2.5", "True"]


def test_nested_enum_inside_properties_is_stringified():
    schema = {
        "type": "object",
        "properties": {
            "auto_archive_duration": {
                "type": "integer",
                "enum": [60, 1440],
            },
            "name": {"type": "string"},
        },
    }
    cleaned = sanitize_gemini_tool_parameters(schema)
    assert cleaned["properties"]["auto_archive_duration"]["enum"] == ["60", "1440"]
    assert "enum" not in cleaned["properties"]["name"]


def test_enum_in_items_is_stringified():
    schema = {
        "type": "array",
        "items": {"type": "integer", "enum": [1, 2, 3]},
    }
    cleaned = sanitize_gemini_schema(schema)
    assert cleaned["items"]["enum"] == ["1", "2", "3"]


def test_non_list_enum_is_dropped():
    schema = {"enum": "not-a-list"}
    cleaned = sanitize_gemini_schema(schema)
    assert "enum" not in cleaned
