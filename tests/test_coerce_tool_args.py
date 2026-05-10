"""Tests for model_tools.coerce_tool_args — specifically anyOf/oneOf union coercion."""

from unittest.mock import patch

import pytest

from model_tools import coerce_tool_args


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_schema(properties: dict) -> dict:
    """Wrap *properties* into a minimal tool schema dict."""
    return {
        "parameters": {
            "type": "object",
            "properties": properties,
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCoerceToolArgsAnyOf:
    """anyOf/oneOf union schemas should be resolved and coerced."""

    @patch("model_tools.registry")
    def test_anyof_string_array_coerces_json_string_to_list(self, mock_registry):
        """Reproduces #23129: Jina read_url schema uses anyOf:[string, array].

        The LLM sends `'["https://example.com"]'` (a JSON-encoded string).
        The coercion should parse it into an actual list ``["https://example.com"]``.
        """
        mock_registry.get_schema.return_value = _make_schema({
            "url": {
                "anyOf": [{"type": "string"}, {"type": "array"}],
            },
        })

        result = coerce_tool_args("jina_read_url", {
            "url": '["https://example.com"]',
        })

        assert result["url"] == ["https://example.com"]

    @patch("model_tools.registry")
    def test_anyof_string_array_keeps_plain_string(self, mock_registry):
        """When the value is a plain URL string (not JSON), it should be kept as-is."""
        mock_registry.get_schema.return_value = _make_schema({
            "url": {
                "anyOf": [{"type": "string"}, {"type": "array"}],
            },
        })

        result = coerce_tool_args("jina_read_url", {
            "url": "https://example.com",
        })

        assert result["url"] == "https://example.com"

    @patch("model_tools.registry")
    def test_oneof_string_integer_coerces_number(self, mock_registry):
        """oneOf with string/integer should coerce numeric strings."""
        mock_registry.get_schema.return_value = _make_schema({
            "count": {
                "oneOf": [{"type": "string"}, {"type": "integer"}],
            },
        })

        result = coerce_tool_args("test_tool", {"count": "42"})
        assert result["count"] == 42

    @patch("model_tools.registry")
    def test_oneof_string_integer_keeps_non_numeric(self, mock_registry):
        """oneOf with string/integer should keep non-numeric strings."""
        mock_registry.get_schema.return_value = _make_schema({
            "count": {
                "oneOf": [{"type": "string"}, {"type": "integer"}],
            },
        })

        result = coerce_tool_args("test_tool", {"count": "hello"})
        assert result["count"] == "hello"

    @patch("model_tools.registry")
    def test_anyof_with_empty_variants_falls_through(self, mock_registry):
        """anyOf with empty variants list should skip coercion."""
        mock_registry.get_schema.return_value = _make_schema({
            "val": {"anyOf": []},
        })

        result = coerce_tool_args("test_tool", {"val": "42"})
        assert result["val"] == "42"

    @patch("model_tools.registry")
    def test_anyof_with_no_type_variants_falls_through(self, mock_registry):
        """anyOf variants without 'type' keys should skip coercion."""
        mock_registry.get_schema.return_value = _make_schema({
            "val": {
                "anyOf": [
                    {"$ref": "#/definitions/Foo"},
                    {"const": "bar"},
                ],
            },
        })

        result = coerce_tool_args("test_tool", {"val": "42"})
        assert result["val"] == "42"

    @patch("model_tools.registry")
    def test_explicit_type_takes_precedence_over_anyof(self, mock_registry):
        """When both 'type' and 'anyOf' are present, 'type' should be used."""
        mock_registry.get_schema.return_value = _make_schema({
            "val": {
                "type": "integer",
                "anyOf": [{"type": "string"}],
            },
        })

        result = coerce_tool_args("test_tool", {"val": "42"})
        assert result["val"] == 42

    @patch("model_tools.registry")
    def test_direct_type_still_works(self, mock_registry):
        """Regression: direct type coercion must still work after the anyOf change."""
        mock_registry.get_schema.return_value = _make_schema({
            "count": {"type": "integer"},
            "flag": {"type": "boolean"},
        })

        result = coerce_tool_args("test_tool", {
            "count": "7",
            "flag": "true",
        })
        assert result["count"] == 7
        assert result["flag"] is True

    @patch("model_tools.registry")
    def test_non_string_values_skipped(self, mock_registry):
        """Non-string values should not be coerced."""
        mock_registry.get_schema.return_value = _make_schema({
            "url": {
                "anyOf": [{"type": "string"}, {"type": "array"}],
            },
        })

        result = coerce_tool_args("test_tool", {"url": ["already", "a", "list"]})
        assert result["url"] == ["already", "a", "list"]

    @patch("model_tools.registry")
    def test_anyof_array_object_coerces_json_to_dict(self, mock_registry):
        """anyOf with array/object should coerce JSON string to dict."""
        mock_registry.get_schema.return_value = _make_schema({
            "data": {
                "anyOf": [{"type": "array"}, {"type": "object"}],
            },
        })

        result = coerce_tool_args("test_tool", {"data": '{"key": "value"}'})
        assert result["data"] == {"key": "value"}

    @patch("model_tools.registry")
    def test_anyof_mixed_types_with_array_first(self, mock_registry):
        """anyOf [array, string] — JSON array string should be parsed as list."""
        mock_registry.get_schema.return_value = _make_schema({
            "items": {
                "anyOf": [{"type": "array"}, {"type": "string"}],
            },
        })

        result = coerce_tool_args("test_tool", {"items": '["a", "b"]'})
        assert result["items"] == ["a", "b"]
