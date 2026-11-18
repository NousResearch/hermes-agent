from __future__ import annotations

from types import SimpleNamespace

from session_rules.turns import flatten_message_text, split_thinking_content


def test_flatten_message_text_accepts_chat_and_responses_text_parts():
    content = [
        {"type": "text", "text": "chat text"},
        {"type": "input_text", "text": "user text"},
        {"type": "output_text", "text": "assistant text"},
        {"type": "summary_text", "text": "summary text"},
    ]

    assert flatten_message_text(content) == "chat text\nuser text\nassistant text\nsummary text"


def test_flatten_message_text_accepts_object_parts():
    content = [
        SimpleNamespace(type="output_text", text="object text"),
        {"content": "legacy content"},
    ]

    assert flatten_message_text(content) == "object text\nlegacy content"


def test_split_thinking_content_routes_typed_thinking_blocks_out_of_visible_text():
    """Mistral streams reasoning as {"type": "thinking", "thinking": [{"type": "text", …}]}.

    Flattening alone treats that block as non-text, so the reasoning is discarded and the
    reasoning display stays empty even though the provider billed reasoning tokens.
    """
    content = [
        {"type": "thinking", "thinking": [{"type": "text", "text": "let me think"}], "closed": True},
        {"type": "text", "text": "The answer is 5."},
    ]

    visible, thinking = split_thinking_content(content)

    assert visible == "The answer is 5."
    assert thinking == "let me think"


def test_split_thinking_content_keeps_plain_text_when_no_thinking_block():
    assert split_thinking_content([{"type": "text", "text": "hello"}]) == ("hello", "")
    assert split_thinking_content("just text") == ("just text", "")


def test_split_thinking_content_accepts_a_string_thinking_value():
    assert split_thinking_content({"type": "thinking", "thinking": "raw thought"}) == ("", "raw thought")
