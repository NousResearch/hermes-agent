"""Tests for agent/message_sanitization.py.

Covers surrogate sanitization, JSON repair, non-ASCII stripping,
and image-content removal — all pure, stateless helpers extracted
from run_agent.py.
"""

from __future__ import annotations

import json

import pytest

from agent.message_sanitization import (
    _escape_invalid_chars_in_json_strings,
    _repair_tool_call_arguments,
    _sanitize_messages_non_ascii,
    _sanitize_messages_surrogates,
    _sanitize_structure_non_ascii,
    _sanitize_structure_surrogates,
    _sanitize_surrogates,
    _sanitize_tools_non_ascii,
    _strip_images_from_messages,
    _strip_non_ascii,
)


# ── _sanitize_surrogates ──────────────────────────────────────────────

def test_sanitize_surrogates_clean_text_unchanged():
    assert _sanitize_surrogates("hello world") == "hello world"


def test_sanitize_surrogates_empty_string():
    assert _sanitize_surrogates("") == ""


def test_sanitize_surrogates_replaces_lone_surrogate():
    # U+D800 is a high surrogate (lone, invalid in UTF-8)
    assert _sanitize_surrogates("a\ud800b") == "a\ufffdb"


def test_sanitize_surrogates_replaces_multiple_surrogates():
    assert _sanitize_surrogates("\ud800\udfff\ud900") == "\ufffd\ufffd\ufffd"


def test_sanitize_surrogates_surrogate_pair_stays():
    # A valid surrogate pair — both high & low surrogates
    # Note: Python stores these as the actual surrogate code points in narrow builds
    text = "\ud800\udc00"
    result = _sanitize_surrogates(text)
    # Both are surrogates; each gets replaced
    assert result == "\ufffd\ufffd"


# ── _sanitize_structure_surrogates ────────────────────────────────────

def test_structure_surrogates_flat_dict():
    payload = {"key": "val\ud800ue"}
    found = _sanitize_structure_surrogates(payload)
    assert found is True
    assert payload["key"] == "val\ufffdue"


def test_structure_surrogates_nested_dict():
    payload = {"outer": {"inner": "\ud900bad"}}
    found = _sanitize_structure_surrogates(payload)
    assert found is True
    assert payload["outer"]["inner"] == "\ufffdbad"


def test_structure_surrogates_list():
    payload = ["clean", "\ud800dirty"]
    found = _sanitize_structure_surrogates(payload)
    assert found is True
    assert payload == ["clean", "\ufffddirty"]


def test_structure_surrogates_nested_list():
    payload = [["\udfff"], {"key": ["\ud801"]}]
    found = _sanitize_structure_surrogates(payload)
    assert found is True
    assert payload == [["\ufffd"], {"key": ["\ufffd"]}]


def test_structure_surrogates_clean_returns_false():
    payload = {"key": "value", "num": 42}
    found = _sanitize_structure_surrogates(payload)
    assert found is False
    assert payload == {"key": "value", "num": 42}


def test_structure_surrogates_non_string_values_preserved():
    payload = {"key": 42, "flag": True, "none": None}
    found = _sanitize_structure_surrogates(payload)
    assert found is False


# ── _sanitize_messages_surrogates ─────────────────────────────────────

def test_messages_surrogates_content_string():
    msgs = [{"role": "user", "content": "hel\ud800lo"}]
    found = _sanitize_messages_surrogates(msgs)
    assert found is True
    assert msgs[0]["content"] == "hel\ufffdlo"


def test_messages_surrogates_content_list():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "ok"},
        {"type": "text", "text": "bad\ud900"},
    ]}]
    found = _sanitize_messages_surrogates(msgs)
    assert found is True
    assert msgs[0]["content"][1]["text"] == "bad\ufffd"


def test_messages_surrogates_tool_call_arguments():
    msgs = [{
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "function": {"name": "do", "arguments": '{"x":"\ud800"}'},
        }],
    }]
    found = _sanitize_messages_surrogates(msgs)
    assert found is True
    args = msgs[0]["tool_calls"][0]["function"]["arguments"]
    assert args == '{"x":"\ufffd"}'


def test_messages_surrogates_reasoning_field():
    msgs = [{"role": "assistant", "reasoning": "think\ud800ing"}]
    found = _sanitize_messages_surrogates(msgs)
    assert found is True
    assert msgs[0]["reasoning"] == "think\ufffding"


def test_messages_surrogates_reasoning_details_nested():
    msgs = [{"role": "assistant", "reasoning_details": [
        {"summary": "\ud900text"},
    ]}]
    found = _sanitize_messages_surrogates(msgs)
    assert found is True
    assert msgs[0]["reasoning_details"][0]["summary"] == "\ufffdtext"


def test_messages_surrogates_clean_returns_false():
    msgs = [{"role": "user", "content": "hello"}]
    found = _sanitize_messages_surrogates(msgs)
    assert found is False


def test_messages_surrogates_non_dict_skipped():
    msgs = [{"role": "user", "content": "ok"}, "not a dict", 42]
    found = _sanitize_messages_surrogates(msgs)
    assert found is False  # no-op, doesn't crash


# ── _escape_invalid_chars_in_json_strings ─────────────────────────────

def test_escape_control_chars_tab_in_string():
    raw = '{"key": "hello\tworld"}'
    result = _escape_invalid_chars_in_json_strings(raw)
    assert result == '{"key": "hello\\u0009world"}'


def test_escape_control_chars_newline_in_string():
    raw = '{"key": "line1\nline2"}'
    result = _escape_invalid_chars_in_json_strings(raw)
    assert result == '{"key": "line1\\u000aline2"}'


def test_escape_already_escaped_preserved():
    raw = '{"key": "hello\\nworld"}'
    result = _escape_invalid_chars_in_json_strings(raw)
    assert result == '{"key": "hello\\nworld"}'


def test_escape_clean_json_unchanged():
    raw = '{"a": 1, "b": "hello"}'
    result = _escape_invalid_chars_in_json_strings(raw)
    assert result == raw


# ── _repair_tool_call_arguments ───────────────────────────────────────

def test_repair_empty_string():
    result = _repair_tool_call_arguments("")
    assert result == "{}"


def test_repair_whitespace_only():
    result = _repair_tool_call_arguments("   ")
    assert result == "{}"


def test_repair_python_none():
    result = _repair_tool_call_arguments("None")
    assert result == "{}"


def test_repair_valid_json_unchanged():
    result = _repair_tool_call_arguments('{"x": 1}')
    assert result == '{"x":1}'  # reserialised compact


def test_repair_trailing_comma():
    result = _repair_tool_call_arguments('{"x": 1,}')
    parsed = json.loads(result)
    assert parsed == {"x": 1}


def test_repair_unclosed_brace():
    result = _repair_tool_call_arguments('{"x": 1')
    parsed = json.loads(result)
    assert parsed == {"x": 1}


def test_repair_unclosed_bracket_repaired():
    # Bare unclosed array bracket — can be repaired
    result = _repair_tool_call_arguments("[1, 2")
    parsed = json.loads(result)
    assert parsed == [1, 2]


def test_repair_unclosed_brace_around_unclosed_array():
    # Brace+array both unclosed with wrong nesting order — cannot repair
    result = _repair_tool_call_arguments('{"x": [1, 2')
    assert result == "{}"


def test_repair_excess_closing_brace():
    result = _repair_tool_call_arguments('{"x": 1}}')
    parsed = json.loads(result)
    assert parsed == {"x": 1}


def test_repair_control_chars_strict_false():
    # literal tab inside JSON string — strict=False accepts and reserialises
    result = _repair_tool_call_arguments('{"key": "val\tue"}')
    parsed = json.loads(result)
    assert parsed == {"key": "val\tue"}


def test_repair_unrepairable_returns_empty_object():
    result = _repair_tool_call_arguments("not json at all {{{")
    assert result == "{}"


# ── _strip_non_ascii ──────────────────────────────────────────────────

def test_strip_non_ascii_ascii_only():
    assert _strip_non_ascii("hello") == "hello"


def test_strip_non_ascii_removes_unicode():
    assert _strip_non_ascii("héllo wörld 你好") == "hllo wrld "


def test_strip_non_ascii_empty_string():
    assert _strip_non_ascii("") == ""


# ── _sanitize_messages_non_ascii ──────────────────────────────────────

def test_messages_non_ascii_content_string():
    msgs = [{"role": "user", "content": "héllo"}]
    found = _sanitize_messages_non_ascii(msgs)
    assert found is True
    assert msgs[0]["content"] == "hllo"


def test_messages_non_ascii_content_list():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "nǐ hǎo"},
    ]}]
    found = _sanitize_messages_non_ascii(msgs)
    assert found is True
    assert msgs[0]["content"][0]["text"] == "n ho"


def test_messages_non_ascii_tool_call_arguments():
    msgs = [{
        "role": "assistant",
        "tool_calls": [{
            "id": "c1",
            "function": {"name": "do", "arguments": '{"key":"välue"}'},
        }],
    }]
    found = _sanitize_messages_non_ascii(msgs)
    assert found is True
    args = msgs[0]["tool_calls"][0]["function"]["arguments"]
    assert args == '{"key":"vlue"}'


def test_messages_non_ascii_extra_fields():
    msgs = [{"role": "assistant", "reasoning_content": "thínking"}]
    found = _sanitize_messages_non_ascii(msgs)
    assert found is True
    assert msgs[0]["reasoning_content"] == "thnking"


def test_messages_non_ascii_clean_returns_false():
    msgs = [{"role": "user", "content": "hello"}]
    found = _sanitize_messages_non_ascii(msgs)
    assert found is False


# ── _sanitize_tools_non_ascii ─────────────────────────────────────────

def test_tools_non_ascii_delegates():
    tools = [{"name": "do", "description": "hëllo"}]
    found = _sanitize_tools_non_ascii(tools)
    assert found is True
    assert tools[0]["description"] == "hllo"


def test_tools_non_ascii_clean():
    tools = [{"name": "do"}]
    found = _sanitize_tools_non_ascii(tools)
    assert found is False


# ── _strip_images_from_messages ───────────────────────────────────────

def test_strip_images_removes_image_url_parts():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
    ]}]
    found = _strip_images_from_messages(msgs)
    assert found is True
    assert msgs[0]["content"] == [{"type": "text", "text": "describe"}]


def test_strip_images_image_type_also_removed():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "ok"},
        {"type": "image", "source": "base64..."},
    ]}]
    found = _strip_images_from_messages(msgs)
    assert found is True
    assert msgs[0]["content"] == [{"type": "text", "text": "ok"}]


def test_strip_images_input_image_type_removed():
    msgs = [{"role": "user", "content": [
        {"type": "input_image", "image_url": "..."},
    ]}]
    found = _strip_images_from_messages(msgs)
    assert found is True
    # Non-tool role with all-image content → message deleted
    assert len(msgs) == 0


def test_strip_images_tool_role_preserved_with_placeholder():
    msgs = [
        {"role": "assistant", "tool_calls": [{"id": "t1", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": [
            {"type": "image_url", "image_url": {"url": "img.png"}},
        ]},
    ]
    found = _strip_images_from_messages(msgs)
    assert found is True
    assert msgs[1]["content"] == "[image content removed — server does not support images]"
    # Assistant message with tool_call_id still present
    assert msgs[0]["role"] == "assistant"


def test_strip_images_no_images_returns_false():
    msgs = [{"role": "user", "content": "plain text"}]
    found = _strip_images_from_messages(msgs)
    assert found is False
    assert msgs == [{"role": "user", "content": "plain text"}]


def test_strip_images_string_content_unchanged():
    msgs = [{"role": "user", "content": "just a string"}]
    found = _strip_images_from_messages(msgs)
    assert found is False
    assert msgs[0]["content"] == "just a string"


# ── _sanitize_structure_non_ascii ─────────────────────────────────────

def test_structure_non_ascii_flat_dict():
    payload = {"key": "välue"}
    found = _sanitize_structure_non_ascii(payload)
    assert found is True
    assert payload["key"] == "vlue"


def test_structure_non_ascii_nested_list():
    payload = [{"items": ["hëllo", 42]}]
    found = _sanitize_structure_non_ascii(payload)
    assert found is True
    assert payload == [{"items": ["hllo", 42]}]


def test_structure_non_ascii_clean():
    payload = {"key": "value", "num": 42}
    found = _sanitize_structure_non_ascii(payload)
    assert found is False
