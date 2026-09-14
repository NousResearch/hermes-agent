"""llama.cpp media-marker regression tests.

llama-server randomizes its multimodal placeholder per process start
(``media_marker`` in ``/props``, e.g. ``<__media_CROZLhoWqzum1cQTzZQKMYA9gqWTHJ3E__>``)
so a prompt cannot forge one. Before any prompt is tokenized the server splits
it on that exact string and expects one attached bitmap per piece; a marker in
plain text with no image behind it makes ``mtmd`` throw and the request comes
back ``HTTP 400 {"message": "Failed to tokenize prompt"}`` in ~0.2s, before the
model runs. Retrying and failing over cannot help — every endpoint of the same
server shares the marker — so the whole session wedges.

A session poisons itself simply by reading the server's own ``/props`` into a
tool result, which is exactly how this was hit: one ``curl .../props`` dump
landed in the transcript and every subsequent turn 400'd.

The marker is neutralized on the API copy only, at the same chokepoint that
strips lone surrogates, so the stored transcript keeps what the tool actually
returned.
"""

import pytest

from agent.message_sanitization import (
    _neutralize_media_markers,
    _sanitize_messages_media_markers,
)

# The live marker from the 2026-09-12 incident, and the fixed fallback spelling
# older llama.cpp builds use when randomization is off.
RANDOM_MARKER = "<__media_CROZLhoWqzum1cQTzZQKMYA9gqWTHJ3E__>"
FIXED_MARKER = "<__media__>"
LEGACY_MARKER = "<__image__>"


@pytest.mark.parametrize("marker", [RANDOM_MARKER, FIXED_MARKER, LEGACY_MARKER])
def test_marker_is_removed_from_text(marker):
    out = _neutralize_media_markers(f"  media_marker: {marker}\n  model_alias: qwen-27b")
    assert marker not in out
    assert "model_alias: qwen-27b" in out


def test_non_marker_text_is_untouched():
    """Only the placeholder shape is rewritten; neighbouring angle-bracket
    tokens (chat-template specials quoted in a props dump) must survive."""
    text = "bos_token: <|endoftext|> eos_token: <|im_end|> <__mediaish__x>"
    assert _neutralize_media_markers(text) == text


def test_marker_neutralized_in_tool_result_content():
    messages = [
        {"role": "user", "content": "dump props"},
        {"role": "tool", "tool_call_id": "c1", "content": f"media_marker: {RANDOM_MARKER}"},
    ]
    assert _sanitize_messages_media_markers(messages) is True
    assert RANDOM_MARKER not in messages[1]["content"]
    assert messages[0]["content"] == "dump props"


def test_returns_false_when_no_marker_present():
    messages = [{"role": "user", "content": "no markers here"}]
    assert _sanitize_messages_media_markers(messages) is False
