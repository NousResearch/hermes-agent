"""Tests for _extract_tool_calls_from_reasoning — recovery of tool calls
that models (GLM-5.x, Qwen3.5) emit inside reasoning/thinking blocks."""

import json
import pytest
from types import SimpleNamespace

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(tools=None):
    """Create a minimal AIAgent with optional tool definitions."""
    agent = AIAgent.__new__(AIAgent)
    agent.valid_tool_names = set()
    if tools:
        agent.valid_tool_names = {t["function"]["name"] for t in tools}
    return agent


# ---------------------------------------------------------------------------
# Pattern 1: JSON objects with "name" + "arguments"
# ---------------------------------------------------------------------------

class TestJsonPattern:
    """Pattern 1: bare JSON tool calls inside reasoning."""

    def test_simple_json_tool_call(self):
        agent = _make_agent(tools=[
            {"function": {"name": "web_search"}},
            {"function": {"name": "terminal"}},
        ])
        reasoning = (
            'The user wants to search. I should use the search tool.\n'
            '{"name": "web_search", "arguments": {"query": "python flask"}}'
        )
        result = agent._extract_tool_calls_from_reasoning(
            reasoning, valid_tool_names=agent.valid_tool_names,
        )
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "python flask"}

    def test_multiple_json_tool_calls(self):
        agent = _make_agent(tools=[
            {"function": {"name": "read_file"}},
            {"function": {"name": "write_file"}},
        ])
        reasoning = (
            'First I will read, then write.\n'
            '{"name": "read_file", "arguments": {"path": "/tmp/a.txt"}}\n'
            'After reading, I write:\n'
            '{"name": "write_file", "arguments": {"path": "/tmp/b.txt", "content": "hello"}}'
        )
        result = agent._extract_tool_calls_from_reasoning(
            reasoning, valid_tool_names=agent.valid_tool_names,
        )
        assert len(result) == 2
        assert result[0]["function"]["name"] == "read_file"
        assert result[1]["function"]["name"] == "write_file"

    def test_json_with_parameters_key(self):
        """Some models use 'parameters' instead of 'arguments'."""
        agent = _make_agent(tools=[
            {"function": {"name": "terminal"}},
        ])
        reasoning = '{"name": "terminal", "parameters": {"command": "ls"}}'
        result = agent._extract_tool_calls_from_reasoning(
            reasoning, valid_tool_names=agent.valid_tool_names,
        )
        assert len(result) == 1
        assert result[0]["function"]["name"] == "terminal"

    def test_invalid_tool_name_filtered(self):
        """Unknown tool names are filtered when valid_tool_names is provided."""
        agent = _make_agent(tools=[
            {"function": {"name": "web_search"}},
        ])
        reasoning = '{"name": "nonexistent_tool", "arguments": {"x": 1}}'
        result = agent._extract_tool_calls_from_reasoning(
            reasoning, valid_tool_names=agent.valid_tool_names,
        )
        assert len(result) == 0

    def test_no_filter_when_names_none(self):
        """Without valid_tool_names, all extractions are returned."""
        agent = _make_agent()
        reasoning = '{"name": "any_tool", "arguments": {"x": 1}}'
        result = agent._extract_tool_calls_from_reasoning(
            reasoning, valid_tool_names=None,
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Pattern 2: XML <tool_call name="..."> tags
# ---------------------------------------------------------------------------

class TestXmlPattern:
    """Pattern 2: XML-style tool call tags inside reasoning."""

    def test_tool_call_xml_tag(self):
        agent = _make_agent(tools=[
            {"function": {"name": "read_file"}},
        ])
        reasoning = (
            'I need to read the file.\n'
            '<tool_call name="read_file">{"path": "/etc/hosts"}</tool_call'
            '>'
        )
        result = agent._extract_tool_calls_from_reasoning(
            reasoning, valid_tool_names=agent.valid_tool_names,
        )
        assert len(result) == 1
        assert result[0]["function"]["name"] == "read_file"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"path": "/etc/hosts"}

    def test_tool_call_xml_empty_body(self):
        agent = _make_agent(tools=[
            {"function": {"name": "web_search"}},
        ])
        reasoning = '<tool_call name="web_search"></tool_call'
        reasoning += '>'
        result = agent._extract_tool_calls_from_reasoning(
            reasoning, valid_tool_names=agent.valid_tool_names,
        )
        assert len(result) == 1
        assert result[0]["function"]["arguments"] == "{}"

    def test_tool_call_xml_case_insensitive(self):
        agent = _make_agent(tools=[
            {"function": {"name": "terminal"}},
        ])
        reasoning = '<TOOL_CALL name="terminal">{"command": "pwd"}</TOOL_CALL>'
        result = agent._extract_tool_calls_from_reasoning(
            reasoning, valid_tool_names=agent.valid_tool_names,
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Pattern 3: <function_call> JSON wrapper
# ---------------------------------------------------------------------------

class TestFunctionCallPattern:
    """Pattern 3: <function_call>{JSON}</function_call> tags."""

    def test_function_call_tag(self):
        agent = _make_agent(tools=[
            {"function": {"name": "write_file"}},
        ])
        reasoning = (
            '<function_call>{"name": "write_file", "arguments": '
            '{"path": "/tmp/x", "content": "test"}}</function_call>'
        )
        result = agent._extract_tool_calls_from_reasoning(
            reasoning, valid_tool_names=agent.valid_tool_names,
        )
        assert len(result) == 1
        assert result[0]["function"]["name"] == "write_file"

    def test_function_call_invalid_json_ignored(self):
        reasoning = '<function_call>{not valid json}</function_call>'
        agent = _make_agent()
        result = agent._extract_tool_calls_from_reasoning(reasoning)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_reasoning(self):
        agent = _make_agent()
        assert agent._extract_tool_calls_from_reasoning("") == []
        assert agent._extract_tool_calls_from_reasoning(None) == []

    def test_whitespace_only(self):
        agent = _make_agent()
        assert agent._extract_tool_calls_from_reasoning("   \n  ") == []

    def test_no_tool_calls_in_reasoning(self):
        agent = _make_agent()
        reasoning = "Just thinking about the problem. No tool calls needed."
        assert agent._extract_tool_calls_from_reasoning(reasoning) == []

    def test_dedup_by_id(self):
        """Each extracted call gets a unique ID."""
        agent = _make_agent(tools=[
            {"function": {"name": "web_search"}},
        ])
        reasoning = (
            '{"name": "web_search", "arguments": {"query": "a"}}\n'
            '{"name": "web_search", "arguments": {"query": "b"}}'
        )
        result = agent._extract_tool_calls_from_reasoning(
            reasoning, valid_tool_names=agent.valid_tool_names,
        )
        assert len(result) == 2
        ids = [r["id"] for r in result]
        assert len(set(ids)) == 2  # all unique

    def test_mixed_formats(self):
        """All three patterns can be extracted from the same reasoning."""
        agent = _make_agent(tools=[
            {"function": {"name": "web_search"}},
            {"function": {"name": "read_file"}},
            {"function": {"name": "terminal"}},
        ])
        reasoning = (
            'JSON: {"name": "web_search", "arguments": {"query": "x"}}\n'
            'XML: <tool_call name="read_file">{"path": "/a"}</tool_call'
            '>\n'
            'FC: <function_call>{"name": "terminal", "arguments": '
            '{"command": "ls"}}</function_call>'
        )
        result = agent._extract_tool_calls_from_reasoning(
            reasoning, valid_tool_names=agent.valid_tool_names,
        )
        assert len(result) == 3
        names = {r["function"]["name"] for r in result}
        assert names == {"web_search", "read_file", "terminal"}
