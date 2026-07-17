from __future__ import annotations

from types import SimpleNamespace

from agent.message_content import flatten_message_text


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


def test_strip_think_blocks_accepts_content_part_lists():
    from agent.agent_runtime_helpers import strip_think_blocks

    content = [
        {"type": "thinking", "thinking": "internal scratchpad"},
        {"type": "output_text", "text": "<think>hidden</think>Visible answer"},
    ]

    assert strip_think_blocks(None, content) == "Visible answer"
