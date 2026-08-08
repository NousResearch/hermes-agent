"""Tests for stripping provider-unsupported content blocks before an API call.

GitHub Copilot's Gemini models reject any content block whose ``type`` is not
``text`` or ``image_url``::

    HTTP 400 {"error": {"message": "type has to be either 'image_url' or
              'text'", "code": "invalid_request_body"}}

A session that ran on a Claude model persists extended-thinking blocks as
``{"type": "thinking", ...}`` / ``{"type": "redacted_thinking", ...}``.
Switching that session mid-conversation to a Copilot Gemini model replays those
blocks onto the wire and every subsequent turn 400s.

These tests cover the gate (which models need the strip) and the strip itself
(what it removes, what it preserves, and that stored history is never mutated).
"""

from __future__ import annotations

import copy

from agent.agent_runtime_helpers import (
    _model_requires_text_image_blocks_only,
    strip_unsupported_content_blocks,
)

COPILOT = "https://api.githubcopilot.com"


# --------------------------------------------------------------------------
# Gate: only Copilot-served Gemini needs the strip.
# --------------------------------------------------------------------------


def test_gate_matches_copilot_gemini():
    assert _model_requires_text_image_blocks_only("gemini-3.6-flash", COPILOT) is True


def test_gate_matches_litellm_namespaced_copilot_gemini():
    # litellm routes the same models as ``github_copilot/gemini-*``.
    assert _model_requires_text_image_blocks_only(
        "github_copilot/gemini-3.5-flash", "http://127.0.0.1:4000"
    ) is True


def test_gate_ignores_non_gemini_copilot_models():
    # Claude on Copilot is exactly the model whose thinking blocks we preserve.
    assert _model_requires_text_image_blocks_only("claude-opus-4.8", COPILOT) is False
    assert _model_requires_text_image_blocks_only("gpt-5.5", COPILOT) is False


def test_gate_ignores_gemini_on_native_google_endpoint():
    # Google's native endpoint is a different wire format and is unaffected.
    assert _model_requires_text_image_blocks_only(
        "gemini-2.5-pro", "https://generativelanguage.googleapis.com"
    ) is False


def test_gate_ignores_non_gemini_namespaced_model():
    assert _model_requires_text_image_blocks_only(
        "openai/gpt-5.5", "http://127.0.0.1:4000"
    ) is False


# --------------------------------------------------------------------------
# Strip behaviour.
# --------------------------------------------------------------------------


def test_thinking_block_is_stripped_and_text_survives():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "deliberating", "signature": "sig"},
                {"type": "text", "text": "visible answer"},
            ],
        },
    ]
    out = strip_unsupported_content_blocks(messages)
    assert out[1]["content"] == [{"type": "text", "text": "visible answer"}]


def test_redacted_thinking_block_is_stripped():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "answer"},
            ],
        },
    ]
    out = strip_unsupported_content_blocks(messages)
    assert out[0]["content"] == [{"type": "text", "text": "answer"}]


def test_thinking_only_message_becomes_empty_string_not_empty_list():
    # ``content: []`` is itself rejected by some providers; normalize to "".
    messages = [{"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}]}]
    out = strip_unsupported_content_blocks(messages)
    assert out[0]["content"] == ""


def test_text_and_image_blocks_are_preserved():
    blocks = [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    messages = [{"role": "user", "content": list(blocks)}]
    out = strip_unsupported_content_blocks(messages)
    assert out[0]["content"] == blocks


def test_string_content_is_untouched():
    messages = [{"role": "user", "content": "plain string"}]
    out = strip_unsupported_content_blocks(messages)
    assert out[0]["content"] == "plain string"


def test_stored_history_is_never_mutated():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "x", "signature": "s"},
                {"type": "text", "text": "ok"},
            ],
        },
    ]
    original = copy.deepcopy(messages)
    strip_unsupported_content_blocks(messages)
    # The UI transcript and session persistence must keep the reasoning trace.
    assert messages == original


def test_nothing_to_strip_returns_input_unchanged():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    out = strip_unsupported_content_blocks(messages)
    assert out == messages


def test_empty_messages_is_safe():
    assert strip_unsupported_content_blocks([]) == []


def test_custom_allowed_set_is_honoured():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "u"}},
            ],
        }
    ]
    out = strip_unsupported_content_blocks(messages, allowed={"text"})
    assert out[0]["content"] == [{"type": "text", "text": "hi"}]


def test_tool_calls_and_other_keys_survive_the_copy():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "x"},
                {"type": "text", "text": "calling"},
            ],
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}
            ],
        }
    ]
    out = strip_unsupported_content_blocks(messages)
    assert out[0]["tool_calls"] == messages[0]["tool_calls"]
    assert out[0]["role"] == "assistant"
