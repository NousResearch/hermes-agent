"""Tests for agent.gemma4_text_tool_parser.

The ``TestLiveFeatherlessShapes`` inputs are verbatim ``content`` strings
returned by Featherless for ``coder3101/gemma-4-31B-it-heretic`` with tools
attached (finish_reason ``stop`` in every case).
"""

from __future__ import annotations

import json
import time

from agent.gemma4_text_tool_parser import (
    _naive_split,
    _schema_aware_split,
    extract_gemma4_thinking,
    extract_gemma4_tool_calls,
    is_gemma4_model,
)


def _tool(name: str, *params: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {p: {"type": "string"} for p in params}},
        },
    }


TOOLS = [
    _tool("get_weather", "city"),
    _tool("get_current_weather", "location"),
    _tool("web_search", "query", "limit"),
    _tool("execute_code", "code"),
    _tool("terminal", "command"),
    _tool("write_file", "path", "content"),
]


def _calls(content: str) -> list[tuple[str, dict]]:
    tcs, _ = extract_gemma4_tool_calls(content, agent_tools=TOOLS)
    return [(tc.name, json.loads(tc.arguments)) for tc in tcs]


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


# ── Real provider output ─────────────────────────────────────────


class TestLiveFeatherlessShapes:
    def test_single(self):
        assert _calls("call:get_weather{city:Paris}") == [("get_weather", {"city": "Paris"})]

    def test_parallel_calls_arrive_glued(self):
        assert _calls("call:get_weather{city:Paris}call:get_weather{city:London}") == [
            ("get_weather", {"city": "Paris"}),
            ("get_weather", {"city": "London"}),
        ]

    def test_mixed_tools_glued(self):
        assert _calls("call:get_weather{city:Berlin}call:terminal{command:ls /tmp}") == [
            ("get_weather", {"city": "Berlin"}),
            ("terminal", {"command": "ls /tmp"}),
        ]

    def test_thinking_reasoning_glued_to_call_after_backtick(self):
        content = (
            "thought\nThe user is asking for the current weather in Tokyo.\n\n"
            "The `get_current_weather` tool takes a `location` argument.\n\n"
            '1. **Tool**: `get_current_weather`\n2. **Argument**: `location="Tokyo"`'
            'call:get_current_weather{location: "Tokyo"}'
        )
        _, visible = extract_gemma4_thinking(content)
        tcs, cleaned = extract_gemma4_tool_calls(visible, agent_tools=TOOLS)
        assert [tc.name for tc in tcs] == ["get_current_weather"]
        assert cleaned.endswith('`location="Tokyo"`')

    def test_code_with_quoted_brace_and_contraction(self):
        code = "d = {'a': 1, 'b': '}'}\nprint(d)\nprint(\"don't stop\")"
        assert _calls(f"call:execute_code{{code:{code}}}") == [("execute_code", {"code": code})]

    def test_shell_adjacent_quotes(self):
        command = "echo prefix\"}\" && echo 'it''s done'"
        assert _calls(f"call:terminal{{command:{command}}}") == [("terminal", {"command": command})]

    def test_multi_param_with_apostrophe(self):
        assert _calls("call:web_search{query:what's new in Python 3.14 sorting comma test,limit:5}") == [
            ("web_search", {"query": "what's new in Python 3.14 sorting comma test", "limit": "5"}),
        ]


# ── Braces and quotes inside arguments ───────────────────────────


class TestArgumentBoundaries:
    def test_nested_braces(self):
        assert _calls("call:execute_code{code:x = {'a': {'b': 1}}}") == [
            ("execute_code", {"code": "x = {'a': {'b': 1}}"}),
        ]

    def test_quoted_closing_brace(self):
        assert _calls('call:execute_code{code:print("}")}') == [("execute_code", {"code": 'print("}")'})]

    def test_quoted_opening_brace(self):
        assert _calls("call:execute_code{code:print('{')}") == [("execute_code", {"code": "print('{')"})]

    def test_escaped_quote_in_string(self):
        code = r'print("he said \"}\"")'
        assert _calls(f"call:execute_code{{code:{code}}}") == [("execute_code", {"code": code})]

    def test_string_prefix_quote(self):
        assert _calls('call:execute_code{code:print(f"{x}}")}') == [
            ("execute_code", {"code": 'print(f"{x}}")'}),
        ]

    def test_apostrophe_in_code_comment(self):
        assert _calls("call:execute_code{code:x = {1: 2}  # don't}") == [
            ("execute_code", {"code": "x = {1: 2}  # don't"}),
        ]

    def test_unterminated_quote_is_literal(self):
        assert _calls("call:terminal{command:'tis the season}") == [
            ("terminal", {"command": "'tis the season"}),
        ]

    def test_shell_single_quote_after_word(self):
        assert _calls("call:terminal{command:echo foo'}'}") == [("terminal", {"command": "echo foo'}'"})]

    def test_contractions_do_not_merge_glued_calls(self):
        assert _calls("call:web_search{query:what's up,limit:1}call:web_search{query:it's,limit:2}") == [
            ("web_search", {"query": "what's up", "limit": "1"}),
            ("web_search", {"query": "it's", "limit": "2"}),
        ]

    def test_fence_inside_call_args_still_extracts(self):
        content = "# T\n```py\nprint(1)\n```"
        assert _calls(f"call:write_file{{path:a.md,content:{content}\n}}") == [
            ("write_file", {"path": "a.md", "content": content}),
        ]


# ── Only the trailing run of calls is real ───────────────────────


class TestTrailingRun:
    def test_newline_separated_calls_with_preamble(self):
        tcs, cleaned = extract_gemma4_tool_calls(
            "Let me check.\ncall:get_weather{city:London}\ncall:get_weather{city:Rome}\n",
        )
        assert [json.loads(tc.arguments)["city"] for tc in tcs] == ["London", "Rome"]
        assert cleaned == "Let me check."

    def test_long_reasoning_before_call_is_extracted(self):
        # Leaked reasoning can dwarf the call; the trailing position, not the
        # prose/call ratio, decides whether it is real.
        assert _calls("Reasoning. " * 40 + "call:get_weather{city:Oslo}") == [("get_weather", {"city": "Oslo"})]

    def test_prose_after_call_is_not_a_call(self):
        text = "Use call:terminal{command:ls} to list files."
        assert extract_gemma4_tool_calls(text) == ([], text)

    def test_misread_brace_is_rejected_not_truncated(self):
        assert _calls("Use call:terminal{command:ls} to list files and then foo}") == []

    def test_word_prefixed_call_is_not_a_call(self):
        assert _calls("recall:terminal{command:echo demo}") == []

    def test_no_calls(self):
        assert extract_gemma4_tool_calls("Hello, how can I help?") == ([], "Hello, how can I help?")

    def test_empty_and_none(self):
        assert extract_gemma4_tool_calls("") == ([], "")
        assert extract_gemma4_tool_calls(None) == ([], "")


# ── Truncated calls never dispatch ───────────────────────────────


class TestTruncatedCalls:
    def test_unbalanced(self):
        assert _calls("call:execute_code{code:x = {1: 2}") == []

    def test_quoted_brace_without_final_close(self):
        assert _calls('call:execute_code{code:print("}")') == []

    def test_unclosed_outer_call_hides_inner_call(self):
        assert _calls(
            "call:write_file{path:a.txt,content:\n"
            "call:terminal{command:echo hello world from the hermes agent}"
        ) == []


# ── Markdown examples never dispatch ─────────────────────────────


class TestExampleCallsNotExecuted:
    def test_fenced_example(self):
        text = "Example:\n```\ncall:terminal{command:echo demo}\n```"
        assert extract_gemma4_tool_calls(text) == ([], text)

    def test_fenced_example_with_language_tag(self):
        assert _calls("Example:\n```text\ncall:terminal{command:echo demo}\n```") == []

    def test_unclosed_fence(self):
        assert _calls("Example:\n```\ncall:terminal{command:echo demo}") == []

    def test_tilde_fence(self):
        assert _calls("~~~\ncall:terminal{command:echo demo}\n~~~") == []

    def test_inline_code(self):
        assert _calls("`run call:terminal{command:echo demo}`") == []

    def test_multiline_inline_code(self):
        assert _calls("Example: `run\ncall:terminal{command:echo demo}`") == []

    def test_double_backtick_inline_code(self):
        assert _calls("Example: `` call:terminal{command:echo demo} ``") == []

    def test_unpaired_backtick_before_call(self):
        assert _calls("Example: `call:terminal{command:echo demo}") == []

    def test_blockquote(self):
        assert _calls("> call:terminal{command:echo demo}") == []

    def test_real_call_after_fenced_example(self):
        text = "```\ncall:a{x:1}\n```\ncall:terminal{command:echo hello}"
        tcs, cleaned = extract_gemma4_tool_calls(text, agent_tools=TOOLS)
        assert [tc.name for tc in tcs] == ["terminal"]
        assert cleaned == "```\ncall:a{x:1}\n```"

    def test_longer_closing_fence_closes_the_block(self):
        assert _calls("```\ncall:a{x:1}\n````\ncall:terminal{command:echo hi}") == [
            ("terminal", {"command": "echo hi"}),
        ]


class TestLinearTime:
    def test_many_glued_calls_followed_by_junk(self):
        text = "call:get_weather{city:x}" * 4000 + " junk}"
        start = time.perf_counter()
        assert _calls(text) == []
        assert time.perf_counter() - start < 2.0

    def test_many_unterminated_quotes(self):
        text = ('call:terminal{command:"' + "a" * 20 + "} ") * 4000 + "}"
        start = time.perf_counter()
        assert _calls(text) == []
        assert time.perf_counter() - start < 2.0


# ── Argument splitting ───────────────────────────────────────────


class TestSchemaAwareSplit:
    def test_single_param(self):
        assert _schema_aware_split("code:print('hello')", ["code"]) == {"code": "print('hello')"}

    def test_single_param_with_commas(self):
        assert _schema_aware_split("query:restaurants in Paris, France", ["query"]) == {
            "query": "restaurants in Paris, France",
        }

    def test_multi_param(self):
        assert _schema_aware_split("query:test,limit:10", ["query", "limit"]) == {"query": "test", "limit": "10"}

    def test_multi_param_comma_in_value(self):
        assert _schema_aware_split("query:restaurants in Paris, France,limit:5", ["query", "limit"]) == {
            "query": "restaurants in Paris, France",
            "limit": "5",
        }

    def test_empty(self):
        assert _schema_aware_split("", ["a"]) == {}

    def test_code_with_braces(self):
        code = 'import json\nresult = {"pi": 3.14}\nprint(result)'
        assert _schema_aware_split(f"code:{code}", ["code"])["code"] == code


class TestNaiveSplit:
    def test_simple(self):
        assert _naive_split("city:Paris") == {"city": "Paris"}

    def test_multi(self):
        assert _naive_split("a:1,b:2") == {"a": "1", "b": "2"}

    def test_nested_braces_preserved(self):
        result = _naive_split("q:test,f:{lang:en}")
        assert result["q"] == "test"
        assert "lang:en" in result["f"]


class TestToolCallShape:
    def test_all_values_are_strings(self):
        tcs, _ = extract_gemma4_tool_calls("call:fn{count:007,flag:true}")
        assert json.loads(tcs[0].arguments) == {"count": "007", "flag": "true"}

    def test_tool_call_id_format(self):
        tcs, _ = extract_gemma4_tool_calls("call:fn{x:1}")
        assert tcs[0].id.startswith("gemma4_")
        assert len(tcs[0].id) == len("gemma4_") + 8

    def test_schema_miss_falls_back_to_naive(self):
        assert _calls("call:get_weather{location:Paris}") == [("get_weather", {"location": "Paris"})]

    def test_unknown_tool_uses_naive_split(self):
        assert _calls("call:search{query:hello,limit:3}") == [("search", {"query": "hello", "limit": "3"})]


# ── Thinking extraction ──────────────────────────────────────────


class TestExtractGemma4Thinking:
    def test_simple(self):
        assert extract_gemma4_thinking(
            "<|channel>thought\nLet me think about this.\n<channel|>The answer is 42."
        ) == ("Let me think about this.", "The answer is 42.")

    def test_empty_block(self):
        assert extract_gemma4_thinking("<|channel>thought\n<channel|>Direct answer.") == ("", "Direct answer.")

    def test_no_markers(self):
        assert extract_gemma4_thinking("Just a normal response.") == ("", "Just a normal response.")

    def test_with_tool_calls(self):
        thinking, cleaned = extract_gemma4_thinking(
            "<|channel>thought\nI should search for this.\n<channel|>call:web_search{query:test}"
        )
        assert thinking == "I should search for this."
        assert cleaned == "call:web_search{query:test}"

    def test_none(self):
        assert extract_gemma4_thinking(None) == ("", "")

    def test_stripped_tokens_prefix(self):
        assert extract_gemma4_thinking("thought\nThe user wants 2+2.\n\n2 + 2 = 4") == (
            "",
            "The user wants 2+2.\n\n2 + 2 = 4",
        )

    def test_stripped_tokens_no_false_positive(self):
        text = "I thought about this carefully and the answer is 42."
        assert extract_gemma4_thinking(text) == ("", text)
