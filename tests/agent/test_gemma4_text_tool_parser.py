"""Tests for agent.gemma4_text_tool_parser."""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agent.gemma4_text_tool_parser import (
    extract_gemma4_thinking,
    extract_gemma4_tool_calls,
    is_gemma4_model,
    _find_balanced_calls,
    _naive_split,
    _schema_aware_split,
)


# ── Model detection ──────────────────────────────────────────────


class TestIsGemma4Model:
    def test_matches(self):
        assert is_gemma4_model("coder3101/gemma-4-31B-it-heretic")
        assert is_gemma4_model("google/gemma-4-31b-it")
        assert is_gemma4_model("gemma4-27b")
        assert is_gemma4_model("gemma_4_31b")

    def test_rejects(self):
        assert not is_gemma4_model("gemma-3-27b-it")
        assert not is_gemma4_model("gpt-4o")
        assert not is_gemma4_model(None)
        assert not is_gemma4_model("")


# ── Balanced brace finder ────────────────────────────────────────


class TestFindBalancedCalls:
    def test_simple(self):
        calls = _find_balanced_calls("call:get_weather{city:Paris}")
        assert len(calls) == 1
        assert calls[0][0] == "get_weather"
        assert calls[0][1] == "city:Paris"

    def test_nested_braces(self):
        calls = _find_balanced_calls("call:fn{a:{b:1}}")
        assert len(calls) == 1
        assert calls[0][1] == "a:{b:1}"

    def test_multiple(self):
        calls = _find_balanced_calls("call:a{x:1}\ncall:b{y:2}")
        assert len(calls) == 2
        assert calls[0][0] == "a"
        assert calls[1][0] == "b"

    def test_no_match(self):
        assert _find_balanced_calls("hello world") == []

    def test_unbalanced(self):
        calls = _find_balanced_calls("call:fn{a:{b:1}")
        assert calls == []

    def test_quoted_closing_brace(self):
        calls = _find_balanced_calls('call:execute_code{code:print("}")}')
        assert len(calls) == 1
        assert calls[0][0] == "execute_code"
        assert calls[0][1] == 'code:print("}")'

    def test_quoted_opening_brace(self):
        calls = _find_balanced_calls("call:fn{code:print('{')}")
        assert len(calls) == 1
        assert calls[0][1] == "code:print('{')"

    def test_escaped_quote_in_string(self):
        calls = _find_balanced_calls(r'call:fn{code:print("he said \"}\"")}')
        assert len(calls) == 1
        assert calls[0][1] == r'code:print("he said \"}\"")'

    def test_apostrophe_in_bare_value(self):
        calls = _find_balanced_calls("call:web_search{query:what's new in Python,limit:5}")
        assert len(calls) == 1
        assert calls[0][1] == "query:what's new in Python,limit:5"

    def test_apostrophe_in_code_comment(self):
        calls = _find_balanced_calls("call:execute_code{code:x = {1: 2}  # don't}")
        assert len(calls) == 1
        assert calls[0][1] == "code:x = {1: 2}  # don't"

    def test_unterminated_quote_is_literal(self):
        calls = _find_balanced_calls("call:fn{q:'tis the season}")
        assert len(calls) == 1
        assert calls[0][1] == "q:'tis the season"

    def test_apostrophes_do_not_swallow_next_call(self):
        calls = _find_balanced_calls("call:a{q:what's up}\ncall:b{q:it's}")
        assert [c[0] for c in calls] == ["a", "b"]

    def test_string_prefix_quote(self):
        calls = _find_balanced_calls('call:fn{code:print(f"{x}}")}')
        assert len(calls) == 1
        assert calls[0][1] == 'code:print(f"{x}}")'

    def test_truncated_call_with_quoted_brace_rejected(self):
        assert _find_balanced_calls('call:fn{code:print("}")') == []


# ── Schema-aware splitting ───────────────────────────────────────


class TestSchemaAwareSplit:
    def test_single_param(self):
        result = _schema_aware_split("code:print('hello')", ["code"])
        assert result == {"code": "print('hello')"}

    def test_single_param_with_commas(self):
        result = _schema_aware_split("query:restaurants in Paris, France", ["query"])
        assert result == {"query": "restaurants in Paris, France"}

    def test_multi_param(self):
        result = _schema_aware_split("query:test,limit:10", ["query", "limit"])
        assert result == {"query": "test", "limit": "10"}

    def test_multi_param_comma_in_value(self):
        result = _schema_aware_split(
            "query:restaurants in Paris, France,limit:5", ["query", "limit"]
        )
        assert result == {"query": "restaurants in Paris, France", "limit": "5"}

    def test_empty(self):
        assert _schema_aware_split("", ["a"]) == {}

    def test_code_with_braces(self):
        code = 'import json\nresult = {"pi": 3.14}\nprint(result)'
        result = _schema_aware_split(f"code:{code}", ["code"])
        assert result["code"] == code


# ── Naive splitting ──────────────────────────────────────────────


class TestNaiveSplit:
    def test_simple(self):
        assert _naive_split("city:Paris") == {"city": "Paris"}

    def test_multi(self):
        assert _naive_split("a:1,b:2") == {"a": "1", "b": "2"}

    def test_nested_braces_preserved(self):
        result = _naive_split("q:test,f:{lang:en}")
        assert result["q"] == "test"
        assert "lang:en" in result["f"]


# ── Full extraction ──────────────────────────────────────────────


class TestExtractGemma4ToolCalls:
    def test_simple_call(self):
        tcs, cleaned = extract_gemma4_tool_calls("call:get_weather{city:Paris}")
        assert len(tcs) == 1
        assert tcs[0].name == "get_weather"
        assert json.loads(tcs[0].arguments) == {"city": "Paris"}
        assert cleaned == ""

    def test_with_schema(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                    },
                },
            }
        ]
        tcs, _cleaned = extract_gemma4_tool_calls(
            "call:search{query:restaurants in Paris, France,limit:5}",
            agent_tools=tools,
        )
        assert len(tcs) == 1
        args = json.loads(tcs[0].arguments)
        assert args["query"] == "restaurants in Paris, France"
        assert args["limit"] == "5"

    def test_multiple_calls(self):
        tcs, cleaned = extract_gemma4_tool_calls(
            "call:read_file{path:/tmp/test.txt}\ncall:search{query:hello}"
        )
        assert len(tcs) == 2
        assert tcs[0].name == "read_file"
        assert tcs[1].name == "search"
        assert cleaned == ""

    def test_no_calls(self):
        tcs, cleaned = extract_gemma4_tool_calls("Hello, how can I help?")
        assert len(tcs) == 0
        assert cleaned == "Hello, how can I help?"

    def test_empty(self):
        tcs, cleaned = extract_gemma4_tool_calls("")
        assert len(tcs) == 0
        assert cleaned == ""

    def test_none(self):
        tcs, cleaned = extract_gemma4_tool_calls(None)
        assert len(tcs) == 0
        assert cleaned == ""

    def test_content_before_calls_preserved(self):
        tcs, cleaned = extract_gemma4_tool_calls(
            "Let me check.\ncall:get_weather{city:London}"
        )
        assert len(tcs) == 1
        assert "Let me check." in cleaned

    def test_prose_guard_skips_minority_calls(self):
        long_prose = "Here is a long explanation of how the weather API works. " * 5
        text = f"{long_prose}call:fn{{x:1}}"
        tcs, cleaned = extract_gemma4_tool_calls(text)
        assert len(tcs) == 0
        assert cleaned == text

    def test_all_values_are_strings(self):
        tcs, _ = extract_gemma4_tool_calls("call:fn{count:007,flag:true}")
        args = json.loads(tcs[0].arguments)
        assert args["count"] == "007"
        assert args["flag"] == "true"

    def test_tool_call_id_format(self):
        tcs, _ = extract_gemma4_tool_calls("call:fn{x:1}")
        assert tcs[0].id.startswith("gemma4_")
        assert len(tcs[0].id) == len("gemma4_") + 8

    def test_schema_miss_falls_back_to_naive(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        tcs, _ = extract_gemma4_tool_calls(
            "call:get_weather{location:Paris}",
            agent_tools=tools,
        )
        assert len(tcs) == 1
        args = json.loads(tcs[0].arguments)
        assert args["location"] == "Paris"

    def test_thinking_extraction_simple(self):
        thinking, cleaned = extract_gemma4_thinking(
            "<|channel>thought\nLet me think about this.\n<channel|>The answer is 42."
        )
        assert thinking == "Let me think about this."
        assert cleaned == "The answer is 42."

    def test_thinking_extraction_empty_block(self):
        thinking, cleaned = extract_gemma4_thinking(
            "<|channel>thought\n<channel|>Direct answer."
        )
        assert thinking == ""
        assert cleaned == "Direct answer."

    def test_thinking_extraction_no_markers(self):
        thinking, cleaned = extract_gemma4_thinking("Just a normal response.")
        assert thinking == ""
        assert cleaned == "Just a normal response."

    def test_thinking_extraction_with_tool_calls(self):
        content = (
            "<|channel>thought\nI should search for this.\n<channel|>"
            "call:web_search{query:test}"
        )
        thinking, cleaned = extract_gemma4_thinking(content)
        assert thinking == "I should search for this."
        assert "call:web_search" in cleaned

    def test_thinking_stripped_from_none(self):
        thinking, cleaned = extract_gemma4_thinking(None)
        assert thinking == ""
        assert cleaned == ""

    def test_thinking_stripped_tokens_prefix(self):
        thinking, cleaned = extract_gemma4_thinking(
            "thought\nThe user wants to calculate 2+2.\n\n2 + 2 = 4"
        )
        assert thinking == ""
        assert cleaned == "The user wants to calculate 2+2.\n\n2 + 2 = 4"

    def test_thinking_stripped_tokens_no_false_positive(self):
        thinking, cleaned = extract_gemma4_thinking(
            "I thought about this carefully and the answer is 42."
        )
        assert thinking == ""
        assert cleaned == "I thought about this carefully and the answer is 42."

    def test_single_param_code_execution(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "code_execution",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                    },
                },
            }
        ]
        code = 'import math\nresult = {"pi": math.pi, "e": math.e}\nprint(result)'
        tcs, _ = extract_gemma4_tool_calls(
            f"call:code_execution{{code:{code}}}",
            agent_tools=tools,
        )
        assert len(tcs) == 1
        args = json.loads(tcs[0].arguments)
        assert "import math" in args["code"]
        assert "math.pi" in args["code"]

    def test_code_with_quoted_braces(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                    },
                },
            }
        ]
        tcs, _ = extract_gemma4_tool_calls(
            'call:execute_code{code:print("}")}',
            agent_tools=tools,
        )
        assert len(tcs) == 1
        args = json.loads(tcs[0].arguments)
        assert 'print("}")' == args["code"]

    def test_apostrophe_query_with_schema(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                    },
                },
            }
        ]
        tcs, _ = extract_gemma4_tool_calls(
            "call:web_search{query:what's new in Python,limit:5}",
            agent_tools=tools,
        )
        assert len(tcs) == 1
        assert json.loads(tcs[0].arguments) == {"query": "what's new in Python", "limit": "5"}


class TestExampleCallsNotExecuted:
    def test_fenced_example(self):
        text = "Example:\n```\ncall:terminal{command:echo demo}\n```"
        tcs, cleaned = extract_gemma4_tool_calls(text)
        assert tcs == []
        assert cleaned == text

    def test_fenced_example_with_language_tag(self):
        text = "Example:\n```text\ncall:terminal{command:echo demo}\n```"
        tcs, _ = extract_gemma4_tool_calls(text)
        assert tcs == []

    def test_unclosed_fence(self):
        tcs, _ = extract_gemma4_tool_calls("Example:\n```\ncall:terminal{command:echo demo}")
        assert tcs == []

    def test_tilde_fence(self):
        tcs, _ = extract_gemma4_tool_calls("~~~\ncall:terminal{command:echo demo}\n~~~")
        assert tcs == []

    def test_inline_code(self):
        tcs, _ = extract_gemma4_tool_calls("`run call:terminal{command:echo demo}`")
        assert tcs == []

    def test_blockquote(self):
        tcs, _ = extract_gemma4_tool_calls("> call:terminal{command:echo demo}")
        assert tcs == []

    def test_real_call_after_fenced_example(self):
        text = (
            "```\ncall:a{x:1}\n```\n"
            "call:terminal{command:echo hello world from the hermes agent}"
        )
        tcs, cleaned = extract_gemma4_tool_calls(text)
        assert [tc.name for tc in tcs] == ["terminal"]
        assert cleaned == "```\ncall:a{x:1}\n```"

    def test_fence_inside_call_args_still_extracts(self):
        text = "call:write_file{path:a.md,content:# T\n```py\nprint(1)\n```\n}"
        tcs, _ = extract_gemma4_tool_calls(text)
        assert [tc.name for tc in tcs] == ["write_file"]
