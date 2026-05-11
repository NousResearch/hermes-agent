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
    assert result.startswith('[Replying to: "' + "x" * 500 + '"]')
    assert "x" * 501 not in result


# ── Reply-with-media path injection ───────────────────────────────────────
#
# When a user replies to a media message (PDF / text doc / etc.) with a
# text message ("send this to alice@example.com"), the Telegram adapter's
# reply-to hydrator re-fetches the bytes and appends the cached path to
# ``event.media_urls``. _prepare_inbound_message_text must inject a
# path-level context note so the agent knows the quoted file is at that
# path — otherwise the agent only sees ``[Replying to: [file: ...]]``
# (the filename), which is not enough to operate on.


@pytest.mark.asyncio
async def test_reply_to_pdf_with_text_injects_path():
    """The agent must learn the cached path of the replied-to file."""
    from gateway.platforms.base import MessageType

    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="send this to alice@example.com",
        message_type=MessageType.TEXT,  # critical: TEXT, not DOCUMENT
        source=source,
        reply_to_message_id="42",
        reply_to_text="[file: quarterly_report.pdf (application/pdf)]",
        media_urls=["/cache/doc_abc_quarterly_report.pdf"],
        media_types=["application/pdf"],
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    # The path-injection block must fire even though message_type is TEXT
    assert "The user is replying to a document" in result
    assert "quarterly_report.pdf" in result
    assert "/cache/doc_abc_quarterly_report.pdf" in result
    # And the existing reply-to pointer still appears
    assert "[Replying to:" in result
    # And the user's original text survives
    assert "send this to alice@example.com" in result


@pytest.mark.asyncio
async def test_reply_to_text_doc_includes_content_and_path():
    """Reply-to .txt/.md files should still get the text-document phrasing
    so the agent knows the content is inlined (vs needing to read the file)."""
    from gateway.platforms.base import MessageType

    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="what's in this?",
        message_type=MessageType.TEXT,
        source=source,
        reply_to_message_id="42",
        reply_to_text="[file: notes.md]",
        media_urls=["/cache/doc_xyz_notes.md"],
        media_types=["text/markdown"],
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "The user is replying to a text document" in result
    assert "notes.md" in result
    assert "/cache/doc_xyz_notes.md" in result


@pytest.mark.asyncio
async def test_inbound_document_phrasing_unchanged():
    """Regression: when the user UPLOADS a doc (not reply), the existing
    'The user sent a document' phrasing must still fire — not the
    reply-to phrasing."""
    from gateway.platforms.base import MessageType

    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="here's the file",
        message_type=MessageType.DOCUMENT,
        source=source,
        media_urls=["/cache/doc_def_report.pdf"],
        media_types=["application/pdf"],
        # no reply_to fields
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "The user sent a document" in result
    assert "The user is replying to" not in result


@pytest.mark.asyncio
async def test_reply_to_image_does_not_trigger_path_injection():
    """Images go through the vision pipeline, not file-path injection.
    A reply to a photo with a text caption must NOT produce a
    'The user is replying to a document' note for the image entry."""
    from gateway.platforms.base import MessageType

    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="translate the text in this",
        message_type=MessageType.TEXT,
        source=source,
        reply_to_message_id="42",
        reply_to_text="[photo]",
        media_urls=["/cache/img_abc.jpg"],
        media_types=["image/jpeg"],
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    # Image MIMEs are filtered out by the inner mtype check
    assert "The user is replying to a document" not in result
    assert "The user is replying to a text document" not in result
    # But the reply-to pointer still appears
    assert "[Replying to:" in result
