"""Tests for _sanitize_anthropic_schema_property_keys.

Anthropic's tool schema validator only accepts property keys matching
``^[a-zA-Z0-9_.-]{1,64}$``. Tools (typically MCP servers) that ship keys
like ``filters[]`` or ``user name`` fail request validation with an HTTP
400, which disables native tool-use for the entire request — the model
then emits its tool calls as plain text into the conversation instead of
structured ``tool_use`` blocks.

The sanitizer rewrites offending keys to validator-safe names inside
``_normalize_tool_input_schema`` (and thus ``convert_tools_to_anthropic``),
keeping ``required`` in sync and disambiguating collisions.
"""

from agent.anthropic_adapter import (
    _sanitize_anthropic_schema_property_keys,
    convert_tools_to_anthropic,
)


def _schema(properties, required=None):
    schema = {"type": "object", "properties": properties}
    if required is not None:
        schema["required"] = required
    return schema


class TestSanitizeSchemaPropertyKeys:
    def test_valid_keys_pass_through_unchanged(self):
        schema = _schema(
            {"query": {"type": "string"}, "max.results-2": {"type": "integer"}},
            required=["query"],
        )
        out = _sanitize_anthropic_schema_property_keys(schema)
        assert out == schema

    def test_array_bracket_suffix_is_stripped(self):
        # Rails/PHP-style `filters[]` keys are the most common offender in
        # MCP server schemas.
        schema = _schema({"filters[]": {"type": "array"}}, required=["filters[]"])
        out = _sanitize_anthropic_schema_property_keys(schema)
        assert set(out["properties"]) == {"filters"}
        assert out["required"] == ["filters"]

    def test_invalid_characters_become_underscores(self):
        schema = _schema({"user name (full)": {"type": "string"}})
        out = _sanitize_anthropic_schema_property_keys(schema)
        assert set(out["properties"]) == {"user_name_full"}

    def test_empty_key_gets_placeholder_name(self):
        schema = _schema({"": {"type": "string"}})
        out = _sanitize_anthropic_schema_property_keys(schema)
        assert set(out["properties"]) == {"property"}

    def test_key_longer_than_64_chars_is_truncated(self):
        long_key = "k" * 80
        schema = _schema({long_key: {"type": "string"}})
        out = _sanitize_anthropic_schema_property_keys(schema)
        (key,) = out["properties"]
        assert key == "k" * 64

    def test_colliding_sanitized_keys_are_disambiguated(self):
        # Both sanitize to "param"; the second gets a deterministic
        # hash suffix instead of silently overwriting the first.
        schema = _schema(
            {"param[]": {"type": "array"}, "param()": {"type": "object"}},
            required=["param[]", "param()"],
        )
        out = _sanitize_anthropic_schema_property_keys(schema)
        keys = list(out["properties"])
        assert len(keys) == 2
        assert "param" in keys
        others = [k for k in keys if k != "param"]
        assert others[0].startswith("param_")
        assert set(out["required"]) == set(keys)
        # Deterministic: same input, same output.
        again = _sanitize_anthropic_schema_property_keys(schema)
        assert list(again["properties"]) == keys

    def test_required_entries_without_matching_property_are_dropped(self):
        schema = _schema({"a": {"type": "string"}}, required=["a", "ghost"])
        out = _sanitize_anthropic_schema_property_keys(schema)
        assert out["required"] == ["a"]

    def test_required_removed_when_no_entries_survive(self):
        schema = _schema({"a": {"type": "string"}}, required=["ghost"])
        out = _sanitize_anthropic_schema_property_keys(schema)
        assert "required" not in out

    def test_nested_properties_are_sanitized(self):
        schema = _schema(
            {
                "outer": {
                    "type": "object",
                    "properties": {"inner[]": {"type": "string"}},
                    "required": ["inner[]"],
                },
                "items_holder": {
                    "type": "array",
                    "items": _schema({"deep key": {"type": "string"}}),
                },
            }
        )
        out = _sanitize_anthropic_schema_property_keys(schema)
        outer = out["properties"]["outer"]
        assert set(outer["properties"]) == {"inner"}
        assert outer["required"] == ["inner"]
        items = out["properties"]["items_holder"]["items"]
        assert set(items["properties"]) == {"deep_key"}

    def test_non_dict_nodes_are_returned_unchanged(self):
        assert _sanitize_anthropic_schema_property_keys("string") == "string"
        assert _sanitize_anthropic_schema_property_keys(None) is None
        assert _sanitize_anthropic_schema_property_keys([1, "a"]) == [1, "a"]


class TestConvertToolsSanitizesPropertyKeys:
    def test_invalid_keys_rewritten_end_to_end(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search with filters",
                    "parameters": _schema(
                        {
                            "query": {"type": "string"},
                            "filters[]": {"type": "array"},
                        },
                        required=["query", "filters[]"],
                    ),
                },
            }
        ]
        result = convert_tools_to_anthropic(tools)
        input_schema = result[0]["input_schema"]
        assert set(input_schema["properties"]) == {"query", "filters"}
        assert set(input_schema["required"]) == {"query", "filters"}

    def test_valid_schema_unchanged_end_to_end(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run",
                    "description": "Run",
                    "parameters": _schema(
                        {"command": {"type": "string"}}, required=["command"]
                    ),
                },
            }
        ]
        result = convert_tools_to_anthropic(tools)
        input_schema = result[0]["input_schema"]
        assert input_schema["properties"] == {"command": {"type": "string"}}
        assert input_schema["required"] == ["command"]
