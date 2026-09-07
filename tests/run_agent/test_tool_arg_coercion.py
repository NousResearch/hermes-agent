"""Regression tests for tools/arg_coercion.py issue #104803 (GLM artifact).

GLM/XML-shaped providers emit array parameters as {"item": [...]}, where the
dict containing a single "item" key is an artifact of XML→JSON conversion. When
a parameter's schema says type: array but the incoming value is {"item": X},
coerce_tool_args must unwrap it BEFORE the bare-value wrap, so validation sees
[X] instead of [{"item": X}].
"""

import pytest
from unittest.mock import patch
from tools.arg_coercion import coerce_tool_args, _normalize_json_strings_for_schema


def test_normalize_json_strings_recursively():
    """Recursive string→object normalization within nested structures."""
    schema = {
        "type": "object",
        "properties": {
            "config": {"type": "object"},
            "items": {
                "type": "array",
                "items": {"type": "object"},
            },
        },
    }
    args = {
        "config": '{"key": "value"}',
        "items": ['{"id": 1}', '{"id": 2}'],
    }
    result = _normalize_json_strings_for_schema(args, schema)
    assert result == {
        "config": {"key": "value"},
        "items": [{"id": 1}, {"id": 2}],
    }


class TestGLMItemArtifactUnwrap:
    """Regression: #104803 — GLM XML `{"item": X}` artifact for arrays."""

    def test_item_dict_with_array_inside_unwrapped(self):
        """{"item": ["306", "6"]} for an integer-array param becomes ["306", "6"].

        The dict's "item" key is unwrapped → the bare list is kept. Element
        coercion (string→int) is NOT performed by arg_coercion (only JSON-string
        parsing); validators will receive strings when they are present.
        """
        schema = {
            "parameters": {
                "type": "object",
                "properties": {"categories": {"type": "array", "items": {"type": "integer"}}},
            }
        }
        args = {"categories": {"item": ["306", "6"]}}
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", args)
        # After unwrap, the list is preserved; element coercion from string→int is
        # outside arg_coercion's scope (validators/handlers receive strings).
        assert result == {"categories": ["306", "6"]}, "Should unwrap item-dict → list"

    def test_item_dict_with_single_string_unwrapped(self):
        """{"item": "tag_name"} for an array param becomes ["tag_name"].

        After unwrap, "tag_name" is a bare value for the array param → wrapped once.
        """
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "terms": {
                        "type": "array",
                        "items": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                    }
                },
            }
        }
        args = {"terms": {"item": "tag_name"}}
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", args)
        assert result == {"terms": ["tag_name"]}, "Should unwrap + wrap bare string"

    def test_multi_key_dict_for_array_untouched(self):
        """A dict with >1 key for an array param wraps bare into [dict].

        The unwrap only applies to single-key {"item": ...} artifacts. Multi-key
        dicts don't trigger the unwrap → treated as bare non-container → wrapped.
        """
        schema = {
            "parameters": {
                "type": "object",
                "properties": {"filters": {"type": "array", "items": {"type": "object"}}},
            }
        }
        args = {"filters": {"item": [1, 2], "other": 3}}
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", args)
        # Multi-key → no unwrap → bare dict for array param → wrapped: [dict].
        assert result == {"filters": [{"item": [1, 2], "other": 3}]}, \
            "Multi-key dict for array param wrapped into [dict]"

    def test_legit_item_key_object_schema_untouched(self):
        """An object schema with an "item" property is left alone.

        The unwrap only triggers when the param's schema is type: array AND the
        value is a single-key {"item": ...} dict. For object schemas, the dict is
        the expected shape and must not be touched.
        """
        schema = {
            "parameters": {
                "type": "object",
                "properties": {
                    "metadata": {
                        "type": "object",
                        "properties": {"item": {"type": "string"}},
                    }
                },
            }
        }
        args = {"metadata": {"item": "actual-value"}}
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", args)
        assert result == {"metadata": {"item": "actual-value"}}, "Object schema, no unwrap"

    def test_none_for_array_param_untouched(self):
        """None for an array param is preserved (tool decides default vs empty)."""
        schema = {
            "parameters": {
                "type": "object",
                "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
            }
        }
        args = {"tags": None}
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", args)
        assert result == {"tags": None}, "None preserved for array param"

    def test_already_list_array_param_untouched(self):
        """When the value is already a list for an array param, no wrapping."""
        schema = {
            "parameters": {
                "type": "object",
                "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
            }
        }
        args = {"ids": [1, 2, 3]}
        with patch("tools.arg_coercion.registry.get_schema", return_value=schema):
            result = coerce_tool_args("test_tool", args)
        assert result == {"ids": [1, 2, 3]}, "List for array param, no changes"
