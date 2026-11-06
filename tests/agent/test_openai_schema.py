"""Tests for the OpenAI JSON-Schema nullability restore.

``tools.schema_sanitizer`` turns ``type: [X, "null"]`` into ``type: X`` plus
``nullable: true``. OpenAI ignores ``nullable``, so its models fill optional
fields with placeholders (#38366). These tests cover the restore applied by
``agent/openai_schema.py``.
"""

from __future__ import annotations

import pytest

from agent.openai_schema import (
    is_openai_model,
    sanitize_openai_tool_parameters,
    sanitize_openai_tools,
)


class TestOpenAIModelDetection:
    """is_openai_model() must match across aggregator prefixes."""

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-6-luna",
            "openai/gpt-6-luna",
            "openai/gpt-6.1-sol",
            "openrouter/openai/gpt-5.6-sol",
            "o3-mini",
            "o4-mini",
            "GPT-6-SOL",
        ],
    )
    def test_positive_matches(self, model):
        assert is_openai_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "",
            None,
            "openai/gpt-oss-120b",
            "anthropic/claude-sonnet-4.6",
            "moonshotai/kimi-k2.6",
            "google/gemini-3-flash-preview",
        ],
    )
    def test_negative_matches(self, model):
        assert is_openai_model(model) is False


class TestNullableRestore:
    def test_nullable_hint_becomes_type_array(self):
        params = {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "nullable": True, "minimum": 1},
                "query": {"type": "string"},
            },
            "required": ["query"],
        }
        out = sanitize_openai_tool_parameters(params)
        assert out["properties"]["limit"] == {"type": ["integer", "null"], "minimum": 1}
        assert out["properties"]["query"] == {"type": "string"}
        assert params["properties"]["limit"]["nullable"] is True

    def test_tools_list_is_returned_as_is_when_nothing_is_nullable(self):
        tools = [{"type": "function", "function": {
            "name": "search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }}]
        assert sanitize_openai_tools(tools) is tools
