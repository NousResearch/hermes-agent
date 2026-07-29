"""Regression: the final Anthropic request must carry no blank text block.

`convert_messages_to_anthropic` runs per-message converters that coerce blanks
they produce, but a blank/whitespace-only text block can be synthesized *after*
those run — a compression summary message, a role merge, or an upstream message
whose content arrives pre-shaped as content blocks. Any single blank text block
makes Anthropic reject the whole request with HTTP 400 "text content blocks must
contain non-whitespace text", which then replays on every turn and wedges the
session.

`_coerce_blank_text_blocks` is the final backstop on the fully-assembled request.
Ref #69512 (follow-up: request-level guard, not just per-message).
"""
from agent.anthropic_adapter import (
    _EMPTY_TEXT_PLACEHOLDER,
    convert_messages_to_anthropic,
)


def _all_text_blocks(messages):
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    yield b


def _assert_no_blank(system, messages):
    if isinstance(system, list):
        for b in system:
            if isinstance(b, dict) and b.get("type") == "text":
                assert b["text"].strip(), f"blank text block in system: {b!r}"
    for b in _all_text_blocks(messages):
        assert b["text"].strip(), f"blank text block survived: {b!r}"


def test_pre_shaped_blank_block_in_user_content_is_coerced():
    # Content arrives already as blocks with a blank text part — the per-message
    # user converter does not walk-and-coerce these, so the final guard must.
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "   "},
            {"type": "text", "text": "real question"},
        ]},
    ]
    system, result = convert_messages_to_anthropic(messages)
    _assert_no_blank(system, result)


def test_blank_summary_style_user_message_is_coerced():
    # A compression summary that came back empty becomes a whitespace user
    # message; it must not reach the wire as a blank block.
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "\n\n"},  # empty "summary"-style turn
    ]
    system, result = convert_messages_to_anthropic(messages)
    _assert_no_blank(system, result)


def test_blank_system_block_is_coerced():
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": "  ", "cache_control": {"type": "ephemeral"}},
        ]},
        {"role": "user", "content": "hello"},
    ]
    system, result = convert_messages_to_anthropic(messages)
    _assert_no_blank(system, result)


def test_prepended_leading_user_turn_is_not_blank():
    """The root cause: _ensure_leading_user_turn prepends a placeholder turn when
    messages[0] is not a user turn (post-compaction histories start with an
    assistant summary). That placeholder must be NON-whitespace — a bare " "
    is itself a blank text block and 400s the whole request, wedging every turn.
    Bedrock's equivalent already uses the shared placeholder.
    """
    messages = [
        {"role": "assistant", "content": "summary of earlier turns"},
        {"role": "user", "content": "continue"},
    ]
    system, result = convert_messages_to_anthropic(messages)
    assert result[0]["role"] == "user", "a leading user turn must be prepended"
    _assert_no_blank(system, result)


def test_real_text_is_left_untouched():
    # A real, non-blank turn keeps its text verbatim and is never replaced by
    # the placeholder (the guard only touches blank blocks).
    messages = [
        {"role": "user", "content": "what is 2+2?"},
    ]
    system, result = convert_messages_to_anthropic(messages)
    # Content may be a plain string or a list of blocks; collect text either way.
    texts = []
    for m in result:
        c = m.get("content")
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            texts.extend(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    assert "what is 2+2?" in texts
    assert _EMPTY_TEXT_PLACEHOLDER not in texts
