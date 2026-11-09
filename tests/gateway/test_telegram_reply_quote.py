"""Tests for Telegram native partial-quote handling in _build_message_event.

When a Telegram user replies using Telegram's native quote feature to
select only part of a prior message, the adapter must use ``message.quote.text``
(the user-selected substring) rather than ``message.reply_to_message.text``
(the entire replied-to message). Otherwise the agent receives the full prior
message as ``reply_to_text``, which can cause it to act on unrelated
actionable-looking text the user did not quote (#22619).
"""

from types import SimpleNamespace

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    return TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))


def _make_message(
    text="follow-up",
    reply_to_text=None,
    reply_to_caption=None,
    reply_to_id=42,
    quote_text=None,
    reply_to_entities=None,
    reply_to_caption_entities=None,
    quote_position=None,
):
    chat = SimpleNamespace(id=111, type="private", title=None, full_name="Alice")
    user = SimpleNamespace(id=42, full_name="Alice")

    reply_to_message = None
    if reply_to_text is not None or reply_to_caption is not None:
        reply_to_message = SimpleNamespace(
            message_id=reply_to_id,
            text=reply_to_text,
            caption=reply_to_caption,
            entities=reply_to_entities,
            caption_entities=reply_to_caption_entities,
        )

    quote = None
    if quote_text is not None:
        quote = SimpleNamespace(text=quote_text, position=quote_position)

    return SimpleNamespace(
        chat=chat,
        from_user=user,
        text=text,
        message_thread_id=None,
        message_id=1001,
        reply_to_message=reply_to_message,
        quote=quote,
        date=None,
        forum_topic_created=None,
    )


def test_native_partial_quote_used_as_reply_to_text():
    """When ``message.quote`` is present, prefer the selected substring."""
    from gateway.platforms.event import MessageType

    adapter = _make_adapter()
    msg = _make_message(
        text="mark this one as done",
        reply_to_text=(
            "Briefing:\n- Item A: deploy fix\n- Item B: rotate keys\n- Item C: update docs"
        ),
        quote_text="Item B: rotate keys",
    )

    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "Item B: rotate keys"
    assert event.reply_to_message_id == "42"


def _text_link(offset, length, url):
    return SimpleNamespace(type="text_link", offset=offset, length=length, url=url)


def test_hidden_link_in_replied_to_text_is_inlined():
    """A reply to a message whose link sits behind its anchor text still carries the URL."""
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="summarize the linked article",
        reply_to_text="New post: Read the full report here",
        reply_to_entities=[_text_link(19, 11, "https://example.com/report-2026")],
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == (
        "New post: Read the full report (https://example.com/report-2026) here"
    )


def test_hidden_link_in_replied_to_caption_is_inlined():
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="what is this?",
        reply_to_caption="Chart from the dashboard",
        reply_to_caption_entities=[_text_link(15, 9, "https://example.com/dash")],
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "Chart from the dashboard (https://example.com/dash)"


def test_hidden_link_in_native_quote_is_inlined():
    """Quotes keep no ``text_link`` entities, so the link comes from the replied-to message."""
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="open this one",
        reply_to_text="Links: the docs and the changelog",
        reply_to_entities=[
            _text_link(11, 4, "https://example.com/docs"),
            _text_link(24, 9, "https://example.com/changes"),
        ],
        quote_text="the changelog",
        quote_position=20,
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "the changelog (https://example.com/changes)"


def test_native_quote_at_a_mismatched_position_is_left_as_quoted():
    """``position`` is approximate; links are only mapped where the quote matches exactly."""
    from gateway.platforms.event import MessageType

    msg = _make_message(
        text="open this one",
        reply_to_text="Links: the docs and the changelog",
        reply_to_entities=[_text_link(24, 9, "https://example.com/changes")],
        quote_text="the changelog",
        quote_position=18,
    )

    event = _make_adapter()._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "the changelog"
