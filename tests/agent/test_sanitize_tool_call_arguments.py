"""Tests for sanitize_tool_call_arguments() type validation (issue #58057).

When a model emits function.arguments as a JSON array instead of an object,
strict OpenAI-compatible providers reject with HTTP 400. The sanitizer must
detect non-dict JSON and either unwrap single-element arrays or fall back
to "{}" with a corruption marker.
"""
import json

import pytest

from agent.agent_runtime_helpers import sanitize_tool_call_arguments


def _make_msg(tc_id: str, arguments: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [{"id": tc_id, "function": {"name": "fn", "arguments": arguments}}],
    }


class TestSanitizeToolCallArgumentTypes:
    """validate that non-dict JSON arguments are caught and repaired."""

    def test_valid_dict_passes_through(self):
        msgs = [_make_msg("tc1", '{"key": "value"}')]
        assert sanitize_tool_call_arguments(msgs) == 0
        assert json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"]) == {"key": "value"}

    def test_single_element_array_unwrapped(self):
        """A single-element array [{...}] should be unwrapped to its dict."""
        msgs = [_make_msg("tc2", '[{"mode": "replace", "path": "config.yaml"}]')]
        assert sanitize_tool_call_arguments(msgs) == 0
        args = json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"])
        assert isinstance(args, dict)
        assert args == {"mode": "replace", "path": "config.yaml"}

    def test_multi_element_array_repaired(self):
        """Multi-element arrays cannot be safely unwrapped — repair to {}."""
        msgs = [_make_msg("tc3", '[{"a": 1}, {"b": 2}]')]
        assert sanitize_tool_call_arguments(msgs) == 1
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_array_of_non_dicts_repaired(self):
        """Array of primitives is not a valid arguments dict."""
        msgs = [_make_msg("tc4", '["foo", "bar"]')]
        assert sanitize_tool_call_arguments(msgs) == 1
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_bare_string_repaired(self):
        msgs = [_make_msg("tc5", '"just a string"')]
        assert sanitize_tool_call_arguments(msgs) == 1
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_bare_number_repaired(self):
        msgs = [_make_msg("tc6", "42")]
        assert sanitize_tool_call_arguments(msgs) == 1
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_bare_null_repaired(self):
        msgs = [_make_msg("tc7", "null")]
        assert sanitize_tool_call_arguments(msgs) == 1
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_invalid_json_repaired(self):
        msgs = [_make_msg("tc8", "not json at all")]
        assert sanitize_tool_call_arguments(msgs) == 1
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_empty_string_repaired(self):
        msgs = [_make_msg("tc9", "")]
        assert sanitize_tool_call_arguments(msgs) == 0  # handled before JSON parse
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_single_element_array_with_non_dict_repaired(self):
        """Single-element array with a non-dict element should be repaired."""
        msgs = [_make_msg("tc10", '["not a dict"]')]
        assert sanitize_tool_call_arguments(msgs) == 1
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] == "{}"
