"""Tests for agent.minimax_schema — MiniMax tool schema sanitization."""

from __future__ import annotations

import copy
import json

import pytest

from agent.minimax_schema import (
    is_minimax_model,
    sanitize_minimax_tool_parameters,
    sanitize_minimax_tools,
)


# ---------------------------------------------------------------------------
# is_minimax_model
# ---------------------------------------------------------------------------

class TestIsMiniMaxModel:
    """Model-name detection for MiniMax family."""

    @pytest.mark.parametrize("model", [
        "MiniMax3",
        "minimax3",
        "MiniMax-M3",
        "minimax-m3",
        "MiniMax-M2.7",
        "minimax-m2.7",
        "minimax-m2.5",
        "minimax-m2.7-free",
        "minimax/MiniMax-M3",
        "openrouter/minimax/minimax-m3",
        "nous/minimax/minimax-m2.7",
    ])
    def test_minimax_models_detected(self, model):
        assert is_minimax_model(model) is True

    @pytest.mark.parametrize("model", [
        "gpt-4",
        "gpt-5",
        "claude-3-opus",
        "deepseek-v3",
        "qwen-max",
        "moonshot-v1",
        "kimi-k2.6",
        "gemini-2.0-flash",
        "",
        None,
    ])
    def test_non_minimax_not_detected(self, model):
        assert is_minimax_model(model) is False

    def test_case_insensitive(self):
        assert is_minimax_model("MINIMAX-M3") is True
        assert is_minimax_model("MINIMAX3") is True


# ---------------------------------------------------------------------------
# sanitize_minimax_tool_parameters
# ---------------------------------------------------------------------------

class TestSanitizeMiniMaxToolParameters:
    """Schema-level repairs for MiniMax compatibility."""

    def test_boolean_converted_to_integer(self):
        """Boolean type → integer with enum [0, 1]."""
        params = {
            "type": "object",
            "properties": {
                "full": {
                    "type": "boolean",
                    "description": "If true, return complete content.",
                }
            },
            "required": [],
        }
        result = sanitize_minimax_tool_parameters(params)
        full = result["properties"]["full"]
        assert full["type"] == "integer"
        assert full["enum"] == [0, 1]
        assert "0=false" in full["description"]

    def test_boolean_with_default_strips_default(self):
        """Default keyword is stripped and folded into description."""
        params = {
            "type": "object",
            "properties": {
                "verbose": {
                    "type": "boolean",
                    "description": "Enable verbose output.",
                    "default": False,
                }
            },
            "required": [],
        }
        result = sanitize_minimax_tool_parameters(params)
        verbose = result["properties"]["verbose"]
        assert "default" not in verbose
        assert "default:" in verbose["description"]
        assert verbose["type"] == "integer"

    def test_nullable_stripped(self):
        """Nullable keyword is removed."""
        params = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "nullable": True}
            },
            "required": [],
        }
        result = sanitize_minimax_tool_parameters(params)
        assert "nullable" not in result["properties"]["name"]

    def test_anyOf_null_collapsed(self):
        """anyOf with null branch collapses to the non-null branch."""
        params = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "null"},
                        {"type": "string"},
                    ]
                }
            },
            "required": [],
        }
        result = sanitize_minimax_tool_parameters(params)
        value = result["properties"]["value"]
        assert "anyOf" not in value
        assert value["type"] == "string"

    def test_multibranch_null_union_strips_outer_keywords(self):
        """A retained multi-branch union still loses outer nullable/default.

        Regression for the review finding: the union-handling path returned
        early, so ``nullable`` and ``default`` survived on
        ``anyOf: [string, integer, null]`` schemas and MiniMax would still
        reject the emitted tool schema.
        """
        params = {
            "type": "object",
            "properties": {
                "mode": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                        {"type": "null"},
                    ],
                    "nullable": True,
                    "default": "auto",
                    "description": "Mode",
                }
            },
            "required": [],
        }
        result = sanitize_minimax_tool_parameters(params)
        mode = result["properties"]["mode"]
        assert "nullable" not in mode
        assert "default" not in mode
        # Union kept (two non-null branches) without a synthetic type.
        assert mode["anyOf"] == [{"type": "string"}, {"type": "integer"}]
        assert "type" not in mode
        # Common cleanup ran: default folded into the description.
        assert "default:" in mode["description"]

    def test_clean_union_strips_outer_keywords(self):
        """Even a union with no null branches gets outer keyword cleanup."""
        params = {
            "type": "object",
            "properties": {
                "mode": {
                    "anyOf": [{"type": "string"}, {"type": "integer"}],
                    "nullable": True,
                    "default": "auto",
                }
            },
            "required": [],
        }
        result = sanitize_minimax_tool_parameters(params)
        mode = result["properties"]["mode"]
        assert "nullable" not in mode
        assert "default" not in mode
        assert mode["anyOf"] == [{"type": "string"}, {"type": "integer"}]

    def test_missing_type_filled(self):
        """Missing type on property → inferred type."""
        params = {
            "type": "object",
            "properties": {
                "name": {"description": "A name"},
                "items": {"items": {"type": "string"}, "description": "A list"},
            },
            "required": [],
        }
        result = sanitize_minimax_tool_parameters(params)
        assert result["properties"]["name"]["type"] == "string"
        assert result["properties"]["items"]["type"] == "array"

    def test_object_without_required_gets_empty_array(self):
        """Object schemas must have a required array."""
        params = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            # no "required" key
        }
        result = sanitize_minimax_tool_parameters(params)
        assert "required" in result
        assert isinstance(result["required"], list)

    def test_top_level_always_object(self):
        """Top-level parameters must be type: object."""
        params = {"type": "string"}
        result = sanitize_minimax_tool_parameters(params)
        assert result["type"] == "object"

    def test_non_dict_input_returns_default(self):
        """Non-dict input → default empty object schema."""
        result = sanitize_minimax_tool_parameters(None)
        assert result == {"type": "object", "properties": {}, "required": []}

        result = sanitize_minimax_tool_parameters("invalid")
        assert result["type"] == "object"

    def test_preserves_string_enum(self):
        """String enums are preserved."""
        params = {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                }
            },
            "required": ["direction"],
        }
        result = sanitize_minimax_tool_parameters(params)
        assert result["properties"]["direction"]["enum"] == ["up", "down"]

    def test_input_not_mutated(self):
        """Original input is not mutated."""
        params = {
            "type": "object",
            "properties": {
                "flag": {"type": "boolean", "default": False}
            },
        }
        original = copy.deepcopy(params)
        sanitize_minimax_tool_parameters(params)
        assert params == original


# ---------------------------------------------------------------------------
# sanitize_minimax_tools
# ---------------------------------------------------------------------------

class TestSanitizeMiniMaxTools:
    """Full tool-list sanitization."""

    def _make_tool(self, name, params):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Tool {name}",
                "parameters": params,
            }
        }

    def test_sanitizes_openai_style(self):
        """OpenAI-style tools (type=function, function.parameters)."""
        tool = self._make_tool("test_tool", {
            "type": "object",
            "properties": {
                "flag": {"type": "boolean", "description": "A flag"}
            },
            "required": [],
        })
        result = sanitize_minimax_tools([tool])
        flag = result[0]["function"]["parameters"]["properties"]["flag"]
        assert flag["type"] == "integer"
        assert flag["enum"] == [0, 1]

    def test_sanitizes_anthropic_style(self):
        """Anthropic-style tools (input_schema)."""
        tool = {
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "flag": {"type": "boolean", "description": "A flag"}
                },
                "required": [],
            },
        }
        result = sanitize_minimax_tools([tool])
        flag = result[0]["input_schema"]["properties"]["flag"]
        assert flag["type"] == "integer"
        assert flag["enum"] == [0, 1]

    def test_empty_tools_unchanged(self):
        """Empty/None tools list returns as-is."""
        assert sanitize_minimax_tools([]) == []
        assert sanitize_minimax_tools(None) is None

    def test_browser_snapshot_tool(self):
        """Real browser_snapshot schema is sanitized correctly."""
        tool = {
            "type": "function",
            "function": {
                "name": "browser_snapshot",
                "description": "Get a text-based snapshot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "full": {
                            "type": "boolean",
                            "description": "If true, returns complete page content.",
                            "default": False,
                        }
                    },
                    "required": [],
                },
            },
        }
        result = sanitize_minimax_tools([tool])
        full = result[0]["function"]["parameters"]["properties"]["full"]
        assert full["type"] == "integer"
        assert full["enum"] == [0, 1]
        assert "default" not in full

    def test_mixed_tools(self):
        """Mix of boolean and string tools all sanitized."""
        tools = [
            self._make_tool("a", {
                "type": "object",
                "properties": {"x": {"type": "boolean"}},
                "required": [],
            }),
            self._make_tool("b", {
                "type": "object",
                "properties": {"y": {"type": "string"}},
                "required": ["y"],
            }),
        ]
        result = sanitize_minimax_tools(tools)
        assert result[0]["function"]["parameters"]["properties"]["x"]["type"] == "integer"
        assert result[1]["function"]["parameters"]["properties"]["y"]["type"] == "string"

    def test_input_not_mutated(self):
        """Original tools list is not mutated."""
        tools = [self._make_tool("test", {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
            "required": [],
        })]
        original = copy.deepcopy(tools)
        sanitize_minimax_tools(tools)
        assert tools == original

    def test_real_browser_back_tool(self):
        """browser_back with empty properties passes through cleanly."""
        tool = {
            "type": "function",
            "function": {
                "name": "browser_back",
                "description": "Navigate back.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
        result = sanitize_minimax_tools([tool])
        params = result[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert params["required"] == []

    def test_real_browser_scroll_tool(self):
        """browser_scroll with string enum preserved."""
        tool = {
            "type": "function",
            "function": {
                "name": "browser_scroll",
                "description": "Scroll the page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down"],
                        }
                    },
                    "required": ["direction"],
                },
            },
        }
        result = sanitize_minimax_tools([tool])
        direction = result[0]["function"]["parameters"]["properties"]["direction"]
        assert direction["type"] == "string"
        assert direction["enum"] == ["up", "down"]

    def test_browser_console_tool(self):
        """browser_console with boolean clear + string expression."""
        tool = {
            "type": "function",
            "function": {
                "name": "browser_console",
                "description": "Get console output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "clear": {
                            "type": "boolean",
                            "default": False,
                            "description": "If true, clear buffers",
                        },
                        "expression": {
                            "type": "string",
                            "description": "JS expression to evaluate",
                        },
                    },
                    "required": [],
                },
            },
        }
        result = sanitize_minimax_tools([tool])
        props = result[0]["function"]["parameters"]["properties"]
        assert props["clear"]["type"] == "integer"
        assert props["clear"]["enum"] == [0, 1]
        assert "default" not in props["clear"]
        assert props["expression"]["type"] == "string"

    def test_no_changes_preserves_content(self):
        """Content is preserved when nothing needed repair."""
        tools = [self._make_tool("test", {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        })]
        result = sanitize_minimax_tools(tools)
        # No boolean, no missing types, no nullable → content unchanged
        assert result[0]["function"]["parameters"] == tools[0]["function"]["parameters"]
