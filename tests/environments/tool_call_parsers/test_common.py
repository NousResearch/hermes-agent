"""Tests for the shared helpers in common.py."""
from __future__ import annotations

import json

from environments.tool_call_parsers.common import (
    extract_tagged_payloads,
    make_mistral_id,
    make_tool_call,
    make_uuid_id,
    normalize_arguments,
    split_content_from_first_marker,
)


class TestNormalizeArguments:
    def test_dict_becomes_json_string(self):
        assert json.loads(normalize_arguments({"a": 1, "b": "c"})) == {"a": 1, "b": "c"}

    def test_str_passes_through(self):
        assert normalize_arguments('{"a":1}') == '{"a":1}'

    def test_list_becomes_json(self):
        assert json.loads(normalize_arguments([1, 2])) == [1, 2]

    def test_none_becomes_json_null(self):
        assert normalize_arguments(None) == "null"

    def test_unicode_not_escaped(self):
        out = normalize_arguments({"msg": "héllo"})
        assert "héllo" in out


class TestMakeUuidId:
    def test_default_prefix(self):
        assert make_uuid_id().startswith("call_")
        assert len(make_uuid_id()) == len("call_") + 8

    def test_custom_prefix(self):
        assert make_uuid_id("foo_").startswith("foo_")

    def test_ids_are_unique(self):
        assert make_uuid_id() != make_uuid_id()


class TestMakeMistralId:
    def test_shape(self):
        id_ = make_mistral_id()
        assert len(id_) == 9
        assert id_.isalnum()


class TestMakeToolCall:
    def test_builds_function_with_normalized_args(self):
        tc = make_tool_call(name="get_weather", arguments={"city": "NYC"})
        assert tc.type == "function"
        assert tc.function.name == "get_weather"
        assert json.loads(tc.function.arguments) == {"city": "NYC"}

    def test_accepts_custom_id(self):
        tc = make_tool_call(name="x", arguments={}, call_id="my_id")
        assert tc.id == "my_id"

    def test_default_id_is_uuid(self):
        tc = make_tool_call(name="x", arguments={})
        assert tc.id.startswith("call_")


class TestExtractTaggedPayloads:
    OPEN = "<tool_call>"
    CLOSE = "</tool_call>"

    def test_no_tag_returns_empty(self):
        assert extract_tagged_payloads("just text", self.OPEN, self.CLOSE) == []

    def test_single_closed_tag(self):
        out = extract_tagged_payloads(
            'pre <tool_call>{"a":1}</tool_call> post', self.OPEN, self.CLOSE
        )
        assert out == ['{"a":1}']

    def test_multiple_closed_tags(self):
        out = extract_tagged_payloads(
            '<tool_call>A</tool_call> <tool_call>B</tool_call>', self.OPEN, self.CLOSE
        )
        assert out == ["A", "B"]

    def test_unclosed_tag_at_end(self):
        out = extract_tagged_payloads(
            'stuff <tool_call>partial', self.OPEN, self.CLOSE
        )
        assert out == ["partial"]

    def test_dotall_spans_newlines(self):
        out = extract_tagged_payloads(
            "<tool_call>\n{\n  \"a\":1\n}\n</tool_call>", self.OPEN, self.CLOSE
        )
        assert out == ['{\n  "a":1\n}']

    def test_empty_payload_skipped(self):
        out = extract_tagged_payloads(
            "<tool_call>   </tool_call>", self.OPEN, self.CLOSE
        )
        assert out == []

    def test_custom_tags_with_regex_special_chars(self):
        out = extract_tagged_payloads(
            "<longcat_tool_call>X</longcat_tool_call>",
            "<longcat_tool_call>",
            "</longcat_tool_call>",
        )
        assert out == ["X"]


class TestSplitContentFromFirstMarker:
    def test_content_before_marker(self):
        out, found = split_content_from_first_marker("hello <tag>", "<tag>")
        assert out == "hello"
        assert found is True

    def test_no_marker(self):
        out, found = split_content_from_first_marker("just text", "<tag>")
        assert out == "just text"
        assert found is False

    def test_marker_at_start_empty_prefix(self):
        out, found = split_content_from_first_marker("<tag>body", "<tag>")
        assert out is None
        assert found is True

    def test_empty_input(self):
        out, found = split_content_from_first_marker("", "<tag>")
        assert out is None
        assert found is False
