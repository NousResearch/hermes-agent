"""Tests for reply-to pointer injection in _prepare_inbound_message_text.

The `[Replying to: "..."]` prefix is a *disambiguation pointer*, not
deduplication. It must always be injected when the user explicitly replies
to a prior message — even when the quoted text already exists somewhere
in the conversation history. History can contain the same or similar text
multiple times, and without an explicit pointer the agent has to guess
which prior message the user is referencing.
"""
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_name="DM",
        chat_type="private",
        user_name="Alice",
    )


@pytest.mark.asyncio
async def test_reply_prefix_injected_when_text_absent_from_history():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="What's the best time to go?",
        source=source,
        reply_to_message_id="42",
        reply_to_text="Japan is great for culture, food, and efficiency.",
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[{"role": "user", "content": "unrelated"}],
    )

    assert result is not None
    assert result.startswith(
        '[Replying to: "Japan is great for culture, food, and efficiency."]'
    )
    assert result.endswith("What's the best time to go?")


@pytest.mark.asyncio
async def test_reply_prefix_still_injected_when_text_in_history():
    """Regression test: the pointer must survive even when the quoted text
    already appears in history. Previously a `found_in_history` guard
    silently dropped the prefix, leaving the agent to guess which prior
    message the user was referencing."""
    runner = _make_runner()
    source = _source()
    quoted = "Japan is great for culture, food, and efficiency."
    event = MessageEvent(
        text="What's the best time to go?",
        source=source,
        reply_to_message_id="42",
        reply_to_text=quoted,
    )

    history = [
        {"role": "user", "content": "I'm thinking of going to Japan or Italy."},
        {
            "role": "assistant",
            "content": (
                f"{quoted} Italy is better if you prefer a relaxed pace."
            ),
        },
        {"role": "user", "content": "How long should I stay?"},
        {"role": "assistant", "content": "For Japan, 10-14 days is ideal."},
    ]

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=history,
    )

    assert result is not None
    assert result.startswith(f'[Replying to: "{quoted}"]')
    assert result.endswith("What's the best time to go?")


@pytest.mark.asyncio
async def test_own_message_reply_prefix_marks_assistant_message():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="this one",
        source=source,
        reply_to_message_id="42",
        reply_to_text="Use the direct train.",
        reply_to_is_own_message=True,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert result.startswith('[Replying to your previous message: "Use the direct train."]')
    assert result.endswith("this one")


@pytest.mark.asyncio
async def test_no_prefix_without_reply_context():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "hello"


@pytest.mark.asyncio
async def test_no_prefix_when_reply_to_text_is_empty():
    """reply_to_message_id alone without text (e.g. a reply to a media-only
    message) should not produce an empty `[Replying to: ""]` prefix."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="hi",
        source=source,
        reply_to_message_id="42",
        reply_to_text=None,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "hi"


@pytest.mark.asyncio
async def test_reply_snippet_truncated_to_500_chars():
    runner = _make_runner()
    source = _source()
    long_text = "x" * 800
    event = MessageEvent(
        text="follow-up",
        source=source,
        reply_to_message_id="42",
        reply_to_text=long_text,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    # After sanitization, the snippet is truncated at word boundary with ellipsis.
    # For a single long word "x"*800, the truncation falls back to max_len chars.
    assert "x" * 501 not in result
    assert "…" in result


@pytest.mark.asyncio
async def test_reply_snippet_strips_markdown_formatting():
    """Markdown tables, emphasis, code blocks, and links should be stripped
    to plaintext before injection (issue #69060)."""
    runner = _make_runner()
    source = _source()
    md_text = (
        "**Bold** and *italic* text.\n\n"
        "| Col A | Col B |\n|---|---|\n| val 1 | val 2 |\n\n"
        "Read [this link](https://example.com) and `code` here.\n\n"
        "```python\nprint('hello')\n```"
    )
    event = MessageEvent(
        text="my reply",
        source=source,
        reply_to_message_id="99",
        reply_to_text=md_text,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    # No raw markdown delimiters should survive
    assert "|---|" not in result
    assert "**" not in result
    assert "```" not in result
    # Content words should still be present
    assert "Bold" in result
    assert "italic" in result
    assert "val 1" in result or "val 2" in result
    assert "this link" in result
    assert "code" in result


@pytest.mark.asyncio
async def test_reply_snippet_truncates_at_word_boundary():
    """Long plaintext snippets should truncate at a word boundary and append
    an ellipsis rather than cutting mid-word."""
    runner = _make_runner()
    source = _source()
    # Build a long message with real words
    long_text = " ".join(["word"] * 200)  # 200 words, ~800 chars
    event = MessageEvent(
        text="follow-up",
        source=source,
        reply_to_message_id="42",
        reply_to_text=long_text,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    # The snippet is truncated with an ellipsis inside the quotes
    assert "…" in result
    # The ellipsis appears before the closing quote
    assert '…"]' in result
    # Total reply snippet should be <= 500 chars + ellipsis overhead
    import re
    snippet_match = re.search(r'Replying to.*?"(.+?)"', result)
    assert snippet_match is not None
    snippet = snippet_match.group(1)
    assert snippet.endswith("…")
    # Should not end mid-word
    assert not snippet.removesuffix("…").endswith("wor")
