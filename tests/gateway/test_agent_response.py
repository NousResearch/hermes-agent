"""Tests for ``gateway.agent_response`` — gateway-level agent prefix."""

from __future__ import annotations

import pytest

from gateway.agent_response import (
    apply_agent_prefix_to_result,
    format_agent_response,
    show_agent_name_enabled,
)


class TestShowAgentNameEnabled:
    def test_default_true(self):
        assert show_agent_name_enabled(None) is True
        assert show_agent_name_enabled({}) is True

    def test_explicit_true(self):
        assert show_agent_name_enabled({"gateway": {"show_agent_name": True}})
        assert show_agent_name_enabled({"gateway": {"show_agent_name": "yes"}})
        assert show_agent_name_enabled({"gateway": {"show_agent_name": 1}})

    def test_explicit_false(self):
        assert not show_agent_name_enabled({"gateway": {"show_agent_name": False}})
        assert not show_agent_name_enabled({"gateway": {"show_agent_name": "no"}})
        assert not show_agent_name_enabled({"gateway": {"show_agent_name": 0}})

    def test_non_dict_gateway_section_falls_back_to_default(self):
        assert show_agent_name_enabled({"gateway": "weird"})


class TestFormatAgentResponse:
    def test_prefixes(self):
        assert format_agent_response("hello", "coder") == "[coder] hello"

    def test_disabled_returns_content(self):
        assert format_agent_response("hello", "coder", enabled=False) == "hello"

    def test_empty_name_returns_content(self):
        assert format_agent_response("hello", None) == "hello"
        assert format_agent_response("hello", "") == "hello"
        assert format_agent_response("hello", "   ") == "hello"

    def test_empty_content_returns_content(self):
        # Never surface a bare prefix
        assert format_agent_response("", "coder") == ""
        assert format_agent_response("   ", "coder") == "   "

    def test_idempotent(self):
        once = format_agent_response("hi", "coder")
        twice = format_agent_response(once, "coder")
        assert once == twice == "[coder] hi"

    def test_does_not_strip_user_content(self):
        # Leading whitespace and newlines preserved (just prepend)
        original = "first line\nsecond line"
        assert format_agent_response(original, "coder") == "[coder] " + original

    def test_strips_name_whitespace(self):
        assert format_agent_response("hi", "  coder  ") == "[coder] hi"

    def test_non_string_content_returns_content(self):
        # Defensive — caller might pass a non-str final_response
        assert format_agent_response(None, "coder") is None  # type: ignore[arg-type]


class TestApplyAgentPrefixToResult:
    def test_mutates_final_response(self):
        result = {"final_response": "hi"}
        out = apply_agent_prefix_to_result(result, "coder", {})
        assert out is result  # mutates in place
        assert result["final_response"] == "[coder] hi"

    def test_disabled_does_not_mutate(self):
        result = {"final_response": "hi"}
        apply_agent_prefix_to_result(
            result, "coder", {"gateway": {"show_agent_name": False}}
        )
        assert result["final_response"] == "hi"

    def test_no_final_response_key(self):
        result = {"messages": []}
        out = apply_agent_prefix_to_result(result, "coder", {})
        assert out is result
        assert "final_response" not in result

    def test_none_result(self):
        assert apply_agent_prefix_to_result(None, "coder", {}) is None

    def test_idempotent_via_dict(self):
        result = {"final_response": "[coder] hi"}
        apply_agent_prefix_to_result(result, "coder", {})
        assert result["final_response"] == "[coder] hi"

    def test_empty_final_response_not_prefixed(self):
        result = {"final_response": ""}
        apply_agent_prefix_to_result(result, "coder", {})
        assert result["final_response"] == ""

    def test_default_config_prefixes(self):
        """When config is absent, default behaviour is to prefix."""
        result = {"final_response": "hi"}
        apply_agent_prefix_to_result(result, "coder", None)
        assert result["final_response"] == "[coder] hi"

    def test_non_string_final_response_unchanged(self):
        result = {"final_response": 42}  # type: ignore[dict-item]
        apply_agent_prefix_to_result(result, "coder", {})
        assert result["final_response"] == 42
