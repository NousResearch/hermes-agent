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
    """Rule 2: type must not appear at the anyOf parent level.

    When an anyOf contains a null-type branch, Moonshot rejects it.
    The sanitizer collapses the anyOf: single non-null branch is promoted,
    multiple non-null branches have null removed from the list.
    """

    def test_anyof_null_branch_collapsed_to_single_type(self):
        """anyOf [string, null] → plain string (anyOf removed)."""
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
        # null branch removed, anyOf collapsed to the single non-null type
        assert "anyOf" not in from_format
        assert from_format["type"] == "string"

    def test_anyof_multiple_non_null_preserved(self):
        """anyOf [string, integer] (no null) → kept as-is with parent type stripped."""
        params = {
            "type": "object",
            "properties": {
                "mode": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                    ],
                },
            },
        }
        out = sanitize_moonshot_tool_parameters(params)
        mode = out["properties"]["mode"]
        assert "anyOf" in mode
        assert "type" not in mode  # parent type stripped

    def test_anyof_enum_with_null_collapsed(self):
        """anyOf [{enum: [...], type: string}, {type: null}] → enum + type only."""
        params = {
            "type": "object",
            "properties": {
                "db_type": {
                    "anyOf": [
                        {"enum": ["mysql", "postgresql", ""]},
                        {"type": "null"},
                    ],
                },
            },
        }
        out = sanitize_moonshot_tool_parameters(params)
        db_type = out["properties"]["db_type"]
        assert "anyOf" not in db_type
        assert db_type["type"] == "string"
        assert db_type["enum"] == ["mysql", "postgresql"]  # "" stripped by enum cleanup


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
        # Shape: missing type on a property, anyOf with parent type + null, array
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
        # anyOf with null collapsed to plain type
        assert "anyOf" not in out["properties"]["filter"]
        assert out["properties"]["filter"]["type"] == "string"
        assert out["properties"]["tags"]["items"]["type"] == "string"
        assert out["required"] == ["query"]


class TestEnumNullStripping:
    """Rule 3: Moonshot rejects null/empty-string inside enum arrays."""

    def test_enum_null_value_stripped(self):
        """enum containing Python None must have it removed for Moonshot."""
        params = {
            "type": "object",
            "properties": {
                "db_type": {
                    "type": "string",
                    "enum": ["mysql", "postgresql", None],
                },
            },
        }
        out = sanitize_moonshot_tool_parameters(params)
        db_type = out["properties"]["db_type"]
        assert None not in db_type["enum"]
        assert "mysql" in db_type["enum"]
        assert "postgresql" in db_type["enum"]

    def test_enum_empty_string_stripped(self):
        """enum containing empty string '' must have it removed for Moonshot."""
        params = {
            "type": "object",
            "properties": {
                "db_type": {
                    "type": "string",
                    "enum": ["mysql", "postgresql", ""],
                },
            },
        }
        out = sanitize_moonshot_tool_parameters(params)
        db_type = out["properties"]["db_type"]
        assert "" not in db_type["enum"]
        assert db_type["enum"] == ["mysql", "postgresql"]

    def test_enum_all_null_becomes_no_enum(self):
        """enum that only had null/empty values is dropped entirely."""
        params = {
            "type": "object",
            "properties": {
                "val": {
                    "type": "string",
                    "enum": [None, ""],
                },
            },
        }
        out = sanitize_moonshot_tool_parameters(params)
        assert "enum" not in out["properties"]["val"]

    def test_dataslayer_db_type_after_mcp_normalize(self):
        """Real-world: dataslayer db_type anyOf+enum after MCP normalization."""
        # This is the exact shape after _normalize_mcp_input_schema runs:
        # anyOf collapsed, but enum still has null + empty string
        params = {
            "type": "object",
            "properties": {
                "datasource": {"type": "string"},
                "db_type": {
                    "enum": ["mysql", "mariadb", "postgresql", "sqlserver", "oracle", "", None],
                    "type": "string",
                    "nullable": True,
                    "default": None,
                },
            },
            "required": ["datasource"],
        }
        out = sanitize_moonshot_tool_parameters(params)
        db_type = out["properties"]["db_type"]
        assert "nullable" not in db_type, "nullable keyword must be stripped"
        assert None not in db_type["enum"]
        assert "" not in db_type["enum"]
        assert db_type["enum"] == ["mysql", "mariadb", "postgresql", "sqlserver", "oracle"]
        assert db_type["type"] == "string"

    def test_enum_on_object_type_not_stripped(self):
        """enum on non-scalar types (object) should NOT be touched."""
        params = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {},
                    "enum": [{}, None],
                },
            },
        }
        out = sanitize_moonshot_tool_parameters(params)
        # object-typed enum should pass through unchanged
        assert "enum" in out["properties"]["config"]

    def test_anyof_collapse_still_runs_nullable_and_enum_cleanup(self):
        """After anyOf collapses to a single non-null branch, the merged
        node must still have ``nullable`` stripped and null/empty-string
        values removed from enum — not skipped by the early anyOf return.
        """
        params = {
            "type": "object",
            "properties": {
                "db_type": {
                    "anyOf": [
                        {"enum": ["mysql", "postgresql", "", None]},
                        {"type": "null"},
                    ],
                    "nullable": True,
                },
            },
        }
        out = sanitize_moonshot_tool_parameters(params)
        db_type = out["properties"]["db_type"]
        assert "anyOf" not in db_type
        assert "nullable" not in db_type, "nullable must be stripped after anyOf collapse"
        assert db_type["type"] == "string"
        assert db_type["enum"] == ["mysql", "postgresql"], \
            "null/empty enum values must be stripped after anyOf collapse"


class TestUnionTypeList:
    """Regression for #28291: JSON Schema union type lists must not crash."""

    def test_fill_missing_type_preserves_list_type(self):
        """_fill_missing_type returns nodes unchanged when type is a list."""
        from agent.moonshot_schema import _fill_missing_type

        node = {"type": ["number", "string"], "description": "timestamp or string"}
        result = _fill_missing_type(node)

        assert result is node
        assert result["type"] == ["number", "string"]

    def test_fill_missing_type_still_infers_for_missing_type(self):
        """_fill_missing_type still infers type when absent."""
        from agent.moonshot_schema import _fill_missing_type

        node = {"properties": {"x": {"type": "integer"}}}
        result = _fill_missing_type(node)

        assert result["type"] == "object"

    def test_repair_schema_does_not_crash_on_ref_list_type_with_enum(self):
        """_repair_schema enum cleanup skips list-typed $ref schemas."""
        params = {
            "type": "object",
            "properties": {
                "payload": {
                    "$ref": "#/$defs/Payload",
                    "type": ["string", "integer"],
                    "enum": ["a", None, ""],
                },
            },
            "$defs": {"Payload": {"type": "object", "properties": {}}},
        }

        out = sanitize_moonshot_tool_parameters(params)
        payload = out["properties"]["payload"]

        assert payload["$ref"] == "#/$defs/Payload"
        assert payload["type"] == ["string", "integer"]
        assert payload["enum"] == ["a", None, ""]

    def test_end_to_end_union_type_tool_parameter(self):
        """Full sanitize pipeline does not crash on union-type parameters."""
        params = {
            "type": "object",
            "properties": {
                "time_from": {
                    "type": ["number", "string"],
                    "description": "Optional timestamp or string",
                },
                "name": {
                    "type": "string",
                    "description": "A name",
                },
            },
        }

        out = sanitize_moonshot_tool_parameters(params)

        assert out["properties"]["time_from"]["type"] == ["number", "string"]
        assert out["properties"]["name"]["type"] == "string"
