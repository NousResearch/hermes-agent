"""Tests for Moonshot/Kimi flavored-JSON-Schema sanitizer.

Moonshot's tool-parameter validator rejects several shapes that the rest of
the JSON Schema ecosystem accepts:

1. Properties without ``type`` — Moonshot requires ``type`` on every node.
2. ``type`` at the parent of ``anyOf`` — Moonshot requires it only inside
   ``anyOf`` children.

These tests cover the repairs applied by ``agent/moonshot_schema.py``.
"""

from __future__ import annotations

import pytest

from agent.moonshot_schema import (
    is_moonshot_model,
    sanitize_moonshot_tool_parameters,
    sanitize_moonshot_tools,
)


class TestMoonshotModelDetection:
    """is_moonshot_model() must match across aggregator prefixes."""

    @pytest.mark.parametrize(
        "model",
        [
            "kimi-k2.6",
            "kimi-k2-thinking",
            "moonshotai/Kimi-K2.6",
            "moonshotai/kimi-k2.6",
            "nous/moonshotai/kimi-k2.6",
            "openrouter/moonshotai/kimi-k2-thinking",
            "MOONSHOTAI/KIMI-K2.6",
        ],
    )
    def test_positive_matches(self, model):
        assert is_moonshot_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "",
            None,
            "anthropic/claude-sonnet-4.6",
            "openai/gpt-5.4",
            "google/gemini-3-flash-preview",
            "deepseek-chat",
        ],
    )
    def test_negative_matches(self, model):
        assert is_moonshot_model(model) is False


class TestMissingTypeFilled:
    """Rule 1: every property must carry a type."""

    def test_property_without_type_gets_string(self):
        params = {
            "type": "object",
            "properties": {"query": {"description": "a bare property"}},
        }
        out = sanitize_moonshot_tool_parameters(params)
        assert out["properties"]["query"]["type"] == "string"

    def test_property_with_enum_infers_type_from_first_value(self):
        params = {
            "type": "object",
            "properties": {"flag": {"enum": [True, False]}},
        }
        out = sanitize_moonshot_tool_parameters(params)
        assert out["properties"]["flag"]["type"] == "boolean"

    def test_nested_properties_are_repaired(self):
        params = {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "object",
                    "properties": {
                        "field": {"description": "no type"},
                    },
                },
            },
        }
        out = sanitize_moonshot_tool_parameters(params)
        assert out["properties"]["filter"]["properties"]["field"]["type"] == "string"

    def test_array_items_without_type_get_repaired(self):
        params = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"description": "tag entry"},
                },
            },
        }
        out = sanitize_moonshot_tool_parameters(params)
        assert out["properties"]["tags"]["items"]["type"] == "string"

    def test_ref_node_is_not_given_synthetic_type(self):
        """$ref nodes should NOT get a synthetic type — the referenced
        definition supplies it, and Moonshot would reject the conflict."""
        params = {
            "type": "object",
            "properties": {"payload": {"$ref": "#/$defs/Payload"}},
            "$defs": {"Payload": {"type": "object", "properties": {}}},
        }
        out = sanitize_moonshot_tool_parameters(params)
        assert "type" not in out["properties"]["payload"]
        assert out["properties"]["payload"]["$ref"] == "#/$defs/Payload"


class TestAnyOfParentType:
    """Rule 2: type must not appear at the anyOf parent level."""

    def test_parent_type_stripped_when_anyof_present(self):
        params = {
            "type": "object",
            "properties": {
                "from_format": {
                    "type": "string",
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ],
                },
            },
        }
        out = sanitize_moonshot_tool_parameters(params)
        from_format = out["properties"]["from_format"]
        assert "type" not in from_format
        assert "anyOf" in from_format

    def test_anyof_children_missing_type_get_filled(self):
        params = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"description": "A typeless option"},
                    ],
                },
            },
        }
        out = sanitize_moonshot_tool_parameters(params)
        children = out["properties"]["value"]["anyOf"]
        assert children[0]["type"] == "string"
        assert "type" in children[1]


class TestTopLevelGuarantees:
    """The returned top-level schema is always a well-formed object."""

    def test_non_dict_input_returns_empty_object(self):
        assert sanitize_moonshot_tool_parameters(None) == {"type": "object", "properties": {}}
        assert sanitize_moonshot_tool_parameters("garbage") == {"type": "object", "properties": {}}
        assert sanitize_moonshot_tool_parameters([]) == {"type": "object", "properties": {}}

    def test_non_object_top_level_coerced(self):
        params = {"type": "string"}
        out = sanitize_moonshot_tool_parameters(params)
        assert out["type"] == "object"
        assert "properties" in out

    def test_does_not_mutate_input(self):
        params = {
            "type": "object",
            "properties": {"q": {"description": "no type"}},
        }
        snapshot = {
            "type": params["type"],
            "properties": {"q": dict(params["properties"]["q"])},
        }
        sanitize_moonshot_tool_parameters(params)
        assert params["type"] == snapshot["type"]
        assert "type" not in params["properties"]["q"]


class TestToolListSanitizer:
    """sanitize_moonshot_tools() walks an OpenAI-format tool list."""

    def test_applies_per_tool(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"description": "query"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "description": "Does nothing",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        out = sanitize_moonshot_tools(tools)
        assert out[0]["function"]["parameters"]["properties"]["q"]["type"] == "string"
        # Second tool already clean — should be structurally equivalent
        assert out[1]["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_empty_list_is_passthrough(self):
        assert sanitize_moonshot_tools([]) == []
        assert sanitize_moonshot_tools(None) is None

    def test_skips_malformed_entries(self):
        """Entries without a function dict are passed through untouched."""
        tools = [{"type": "function"}, {"not": "a tool"}]
        out = sanitize_moonshot_tools(tools)
        assert out == tools


class TestRealWorldMCPShape:
    """End-to-end: a realistic MCP-style schema that used to 400 on Moonshot."""

    def test_combined_rewrites(self):
        # Shape: missing type on a property, anyOf with parent type, array
        # items without type — all in one tool.
        params = {
            "type": "object",
            "properties": {
                "query": {"description": "search text"},
                "filter": {
                    "type": "string",
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ],
                },
                "tags": {
                    "type": "array",
                    "items": {"description": "tag"},
                },
            },
            "required": ["query"],
        }
        out = sanitize_moonshot_tool_parameters(params)
        assert out["properties"]["query"]["type"] == "string"
        assert "type" not in out["properties"]["filter"]
        assert out["properties"]["filter"]["anyOf"][0]["type"] == "string"
        assert out["properties"]["tags"]["items"]["type"] == "string"
        assert out["required"] == ["query"]
