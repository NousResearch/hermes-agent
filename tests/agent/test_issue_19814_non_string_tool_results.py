"""Regression tests for #19814 — non-string tool results must not break the wire.

Plugin tools (e.g. the ``custom-tools`` plugin) may return a ``dict`` rather than
a string.  Chat Completions requires ``content`` to be a string on ``role: "tool"``
messages, so an unconverted dict makes strict OpenAI-compatible providers
(OpenRouter, DeepSeek, Ollama) reject the *entire* request with
``messages.N.content: Invalid input`` — every turn, until the session is reset.

The same root cause reaches ``tool_guardrails._result_hash()``, which assumes a
string and raises ``AttributeError`` inside ``_sha256()`` when handed a dict.

Recognized multimodal envelopes (``{"_multimodal": True, "content": [...]}``) must
survive untouched — flattening those would break the multimodal path.
"""

from __future__ import annotations

import json

import pytest

from agent.transports.chat_completions import ChatCompletionsTransport
from agent.tool_guardrails import _result_hash


@pytest.fixture
def transport():
    return ChatCompletionsTransport()


def _tool_msg(content):
    return {"role": "tool", "tool_call_id": "call_1", "content": content}


def test_dict_tool_content_is_stringified(transport):
    """A dict tool result is serialized to a JSON string."""
    messages = [_tool_msg({"unread": 3, "subjects": ["a", "b"]})]

    out = transport.convert_messages(messages, model="deepseek-chat")

    assert isinstance(out[0]["content"], str), out[0]
    assert json.loads(out[0]["content"]) == {"unread": 3, "subjects": ["a", "b"]}


def test_list_tool_content_is_stringified(transport):
    """A bare list is also not a valid Chat Completions tool content."""
    messages = [_tool_msg([{"id": 1}, {"id": 2}])]

    out = transport.convert_messages(messages, model="deepseek-chat")

    assert isinstance(out[0]["content"], str), out[0]
    assert json.loads(out[0]["content"]) == [{"id": 1}, {"id": 2}]


def test_stringify_happens_without_any_other_sanitize_trigger(transport):
    """The early ``needs_sanitize`` return must not skip the conversion.

    ``convert_messages`` returns ``messages`` unchanged when nothing needs
    sanitizing.  A message carrying *only* a non-string content has none of the
    other markers, so the detection pass has to recognize it too — otherwise the
    dict flows straight to the provider.
    """
    messages = [_tool_msg({"only": "trigger"})]

    out = transport.convert_messages(messages, model="deepseek-chat")

    assert isinstance(out[0]["content"], str), out[0]


def test_multimodal_envelope_is_preserved(transport):
    """Recognized ``_multimodal`` envelopes must not be flattened."""
    envelope = {
        "_multimodal": True,
        "content": [{"type": "text", "text": "hi"}],
        "text_summary": "hi",
    }
    messages = [_tool_msg(envelope)]

    out = transport.convert_messages(messages, model="deepseek-chat")

    assert out[0]["content"] == envelope


def test_string_and_none_content_are_untouched(transport):
    """Valid content is passed through byte-for-byte."""
    messages = [_tool_msg("already a string"), _tool_msg(None)]

    out = transport.convert_messages(messages, model="deepseek-chat")

    assert out[0]["content"] == "already a string"
    assert out[1]["content"] is None


def test_non_tool_roles_are_untouched(transport):
    """Only ``role: "tool"`` messages are normalized here.

    Assistant/user content parts are lists by design in the OpenAI schema.
    """
    parts = [{"type": "text", "text": "hello"}]
    messages = [{"role": "user", "content": parts}]

    out = transport.convert_messages(messages, model="deepseek-chat")

    assert out[0]["content"] == parts


def test_original_messages_are_not_mutated(transport):
    """Sanitizing copies; the caller's history must stay intact."""
    original = {"unread": 3}
    messages = [_tool_msg(original)]

    transport.convert_messages(messages, model="deepseek-chat")

    assert messages[0]["content"] is original


def test_result_hash_accepts_dict():
    """``_result_hash`` no longer raises AttributeError on a dict result."""
    assert _result_hash({"a": 1}) == _result_hash('{"a":1}')


def test_result_hash_accepts_list():
    assert isinstance(_result_hash([1, 2, 3]), str)


def test_result_hash_still_accepts_str_and_none():
    assert isinstance(_result_hash('{"a": 1}'), str)
    assert isinstance(_result_hash(None), str)


def test_result_hash_survives_unserializable_object():
    """A non-JSON-serializable result degrades to ``str()`` instead of crashing."""

    class Opaque:
        pass

    assert isinstance(_result_hash({"obj": Opaque()}), str)
