"""Tests for MCP tool input schema normalization."""

import pytest
from tools.mcp_tool import _normalize_mcp_input_schema


class TestNormalizeArrayItems:
    """OpenAI rejects JSON Schema arrays without 'items' — ensure we patch them."""

    def test_bare_array_gets_empty_items(self):
        result = _normalize_mcp_input_schema({
            "type": "object",
            "properties": {"tags": {"type": "array"}},
        })
        assert result["properties"]["tags"]["items"] == {}

    def test_typed_array_preserved(self):
        result = _normalize_mcp_input_schema({
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        })
        assert result["properties"]["tags"]["items"] == {"type": "string"}

    def test_nested_bare_array_fixed(self):
        result = _normalize_mcp_input_schema({
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"values": {"type": "array"}},
                }
            },
        })
        assert result["properties"]["nested"]["properties"]["values"]["items"] == {}

    def test_non_array_untouched(self):
        result = _normalize_mcp_input_schema({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        assert "items" not in result["properties"]["name"]

    def test_none_returns_default_object(self):
        result = _normalize_mcp_input_schema(None)
        assert result == {"type": "object", "properties": {}}

    def test_top_level_array_fixed(self):
        result = _normalize_mcp_input_schema({"type": "array"})
        assert result["items"] == {}
