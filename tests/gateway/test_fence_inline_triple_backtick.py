"""A ```code``` span alone on a line is inline code, never a fence opener (CommonMark
forbids a backtick in a backtick fence's info string). Port of openclaw/openclaw#180588."""

from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.helpers import split_text_fence_aware

TEXT = "Install it with:\n\n```pip install foo```\n\n" + (
    "Then configure the thing and run it. " * 4 + "\n\n") * 8


def test_truncate_message_does_not_wrap_prose_after_inline_triple_backticks():
    chunks = BasePlatformAdapter.truncate_message(TEXT, 300)
    assert len(chunks) > 1
    for chunk in chunks[1:]:
        assert not chunk.startswith("```"), chunk[:40]
    assert all("```pip\n" not in c for c in chunks)


def test_paragraph_chunker_splits_prose_after_inline_triple_backticks():
    text = "```literal```\nParagraph outside code with ordinary words."
    chunks = split_text_fence_aware(text, 32)
    assert all(len(c) <= 32 for c in chunks), chunks
    # A real fence still stays indivisible.
    fenced = "```python\n" + "x = 1\n" * 20 + "```"
    assert split_text_fence_aware(fenced, 32) == [fenced]
