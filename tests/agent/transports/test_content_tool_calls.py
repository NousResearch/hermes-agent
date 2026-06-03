"""Tests for agent.transports.content_tool_calls — content-channel tool-call
extraction for models that put tool calls in `content` instead of the
structured `tool_calls` field.  Ported from upstream Hermes PR #35129.
"""

import os

import pytest


# Fixture tools used by the extraction tests
VALID_TOOLS = {"read_file", "shell", "web_search"}


@pytest.fixture(autouse=True)
def _enable_promotion(monkeypatch):
    """Ensure the HERMES_PROMOTE_TOOLCALLS env gate is on for every test."""
    monkeypatch.setenv("HERMES_PROMOTE_TOOLCALLS", "true")
    yield


def _get_extractor():
    """Lazy import — module reads env vars at call time, not import time."""
    from agent.transports.content_tool_calls import extract_content_tool_calls
    return extract_content_tool_calls


class TestToolCallJsonParser:
    """<tool_call>{json}</tool_call> — Ollama qwen2.5-coder, GLM."""

    def test_line_start(self):
        extract = _get_extractor()
        text = '\n<tool_call>\n{"name": "read_file", "arguments": {"path": "/x"}}\n</tool_call>\n'
        calls, residual = extract(text, VALID_TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert '"path"' in calls[0].arguments
        assert "<tool_call>" not in residual

    def test_after_closing_tag(self):
        """Reasoning-block close then tool call is a common pattern."""
        extract = _get_extractor()
        text = '</think>\n<tool_call>\n{"name": "shell", "arguments": {"cmd": "ls"}}\n</tool_call>\n'
        calls, _ = extract(text, VALID_TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "shell"

    def test_mid_prose_does_not_match(self):
        """Narrated framing must not promote."""
        extract = _get_extractor()
        text = "you'd emit <tool_call>{...}</tool_call> for that"
        calls, _ = extract(text, VALID_TOOLS)
        assert calls == []


class TestBareJsonObjectParser:
    """Whole-content {name, arguments} — opt-in via env var."""

    def test_whole_content(self, monkeypatch):
        monkeypatch.setenv("HERMES_PROMOTE_BARE_JSON_TOOLCALL", "true")
        extract = _get_extractor()
        text = '{"name": "shell", "arguments": {"command": "ls"}}'
        calls, residual = extract(text, VALID_TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "shell"
        assert residual == ""

    def test_can_be_disabled_via_env(self, monkeypatch):
        """Setting HERMES_PROMOTE_BARE_JSON_TOOLCALL=false disables the parser."""
        monkeypatch.setenv("HERMES_PROMOTE_BARE_JSON_TOOLCALL", "false")
        extract = _get_extractor()
        text = '{"name": "shell", "arguments": {"command": "ls"}}'
        calls, _ = extract(text, VALID_TOOLS)
        assert calls == []

    def test_extra_keys_rejected(self, monkeypatch):
        """A bare JSON with unknown keys is treated as prose, not a tool call."""
        monkeypatch.setenv("HERMES_PROMOTE_BARE_JSON_TOOLCALL", "true")
        extract = _get_extractor()
        text = '{"name": "shell", "arguments": {}, "extra": "stuff"}'
        calls, _ = extract(text, VALID_TOOLS)
        assert calls == []


class TestMinimaxInvokeParser:
    """<invoke name="..."><parameter name="...">value</parameter></invoke>
    — M3 / MiniMax."""

    def test_basic(self):
        extract = _get_extractor()
        text = (
            '\n<minimax:tool_call>\n'
            '<invoke name="web_search">\n'
            '<parameter name="query">hermes tool calling</parameter>\n'
            '<parameter name="limit">5</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>\n'
        )
        calls, residual = extract(text, VALID_TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "web_search"
        import json
        args = json.loads(calls[0].arguments)
        assert args["query"] == "hermes tool calling"
        assert args["limit"] == "5"
        # The wrapper tag remains in residual (only the inner <invoke> is consumed)
        assert "<minimax:tool_call>" in residual

    def test_unknown_tool_name_fail_closed(self):
        extract = _get_extractor()
        text = (
            '\n<invoke name="nonexistent_tool">\n'
            '<parameter name="x">1</parameter>\n'
            '</invoke>\n'
        )
        calls, _ = extract(text, VALID_TOOLS)
        assert calls == []


class TestKimiK2Parser:
    """Kimi K2 native tool-call tokens."""

    def test_section(self):
        extract = _get_extractor()
        text = (
            '<|tool_calls_section_begin|>\n'
            '<|tool_call_begin|>functions.read_file<|tool_call_argument_begin|>'
            '{"path": "/k.py"}'
            '<|tool_call_end|>\n'
            '<|tool_calls_section_end|>\n'
        )
        calls, _ = extract(text, VALID_TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert '"path"' in calls[0].arguments

    def test_strips_functions_prefix(self):
        extract = _get_extractor()
        text = (
            '<|tool_calls_section_begin|>\n'
            '<|tool_call_begin|>functions.web_search<|tool_call_argument_begin|>'
            '{"q": "x"}'
            '<|tool_call_end|>\n'
            '<|tool_calls_section_end|>\n'
        )
        calls, _ = extract(text, VALID_TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "web_search"


class TestGemmaFunctionParser:
    """<function name="...">{json}</function> — Gemma."""

    def test_basic(self):
        extract = _get_extractor()
        text = '<function name="shell">\n{"command": "echo hi"}\n</function>'
        calls, _ = extract(text, VALID_TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "shell"


class TestPythonicFunctionParser:
    """<function=NAME>{json}</function> — Ollama-cloud pythonic form."""

    def test_basic(self):
        extract = _get_extractor()
        text = '<function=web_search>\n{"query": "hermes"}\n</function>'
        calls, _ = extract(text, VALID_TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "web_search"


class TestOverlappingDedup:
    """Two parsers matching the same span must not double-fire."""

    def test_dedup(self, monkeypatch):
        monkeypatch.setenv("HERMES_PROMOTE_BARE_JSON_TOOLCALL", "true")
        extract = _get_extractor()
        # If both bare_json_object and tool_call_json fire on the same bytes,
        # only one ToolCall should be returned.
        text = (
            '<tool_call>\n'
            '{"name": "shell", "arguments": {"command": "ls"}}\n'
            '</tool_call>\n'
        )
        calls, _ = extract(text, VALID_TOOLS)
        assert len(calls) == 1


class TestEmptyAndInvalid:
    def test_empty_content(self):
        extract = _get_extractor()
        calls, residual = extract("", VALID_TOOLS)
        assert calls == []
        assert residual == ""

    def test_no_matches(self):
        extract = _get_extractor()
        calls, residual = extract("just some normal assistant text", VALID_TOOLS)
        assert calls == []
        assert residual == "just some normal assistant text"

    def test_tool_call_id_is_deterministic(self):
        """Same content + position = same call id (prompt-cache friendly)."""
        extract = _get_extractor()
        text1 = '\n<tool_call>\n{"name": "shell", "arguments": {"x": 1}}\n</tool_call>\n'
        text2 = '\n<tool_call>\n{"name": "shell", "arguments": {"x": 1}}\n</tool_call>\n'
        calls1, _ = extract(text1, VALID_TOOLS)
        calls2, _ = extract(text2, VALID_TOOLS)
        assert calls1[0].id == calls2[0].id

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_PROMOTE_TOOLCALLS", "false")
        extract = _get_extractor()
        text = '\n<invoke name="web_search"><parameter name="q">x</parameter></invoke>\n'
        calls, _ = extract(text, VALID_TOOLS)
        assert calls == []
