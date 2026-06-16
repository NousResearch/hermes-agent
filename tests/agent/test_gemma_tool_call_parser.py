"""Tests for Gemma/oMLX text tool-call recovery."""

from types import SimpleNamespace

from agent.gemma_tool_call_parser import (
    extract_gemma_tool_calls_from_text,
    gemma_args_blob_to_json,
    recover_gemma_text_tool_calls,
    text_contains_gemma_tool_call,
)
from agent.transports.types import NormalizedResponse, ToolCall


SAMPLE = (
    '<|tool_call>call:web_search{query: "future entrepreneurship trends '
    'academic research 2020-2025 Google Scholar"}<tool_call|>'
)


class TestGemmaArgsBlobToJson:
    def test_quotes_bare_keys(self):
        out = gemma_args_blob_to_json('query: "hello"')
        assert json_loads(out) == {"query": "hello"}


def json_loads(text):
    import json

    return json.loads(text)


class TestExtractGemmaToolCalls:
    def test_extracts_web_search(self):
        pairs, cleaned = extract_gemma_tool_calls_from_text(SAMPLE)
        assert len(pairs) == 1
        assert pairs[0][0] == "web_search"
        assert json_loads(pairs[0][1])["query"].startswith("future entrepreneurship")
        assert cleaned == ""

    def test_respects_valid_tool_names(self):
        pairs, _ = extract_gemma_tool_calls_from_text(
            SAMPLE, valid_tool_names={"terminal"}
        )
        assert pairs == []


class TestRecoverGemmaTextToolCalls:
    def test_populates_normalized_response(self):
        msg = NormalizedResponse(content=SAMPLE, tool_calls=None, finish_reason="stop")
        assert recover_gemma_text_tool_calls(
            msg, valid_tool_names={"web_search"}
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "web_search"
        assert msg.content in (None, "")

    def test_noop_when_structured_calls_present(self):
        msg = NormalizedResponse(
            content=SAMPLE,
            tool_calls=[ToolCall(id="x", name="web_search", arguments="{}")],
            finish_reason="tool_calls",
        )
        assert not recover_gemma_text_tool_calls(msg)


class TestTextContainsGemmaToolCall:
    def test_detects_marker(self):
        assert text_contains_gemma_tool_call(SAMPLE)

    def test_negative_on_plain_text(self):
        assert not text_contains_gemma_tool_call("plain answer")
