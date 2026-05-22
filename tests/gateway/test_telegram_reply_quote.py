"""Tests for Telegram native partial-quote handling in _build_message_event.

When a Telegram user replies using Telegram's native quote feature to
select only part of a prior message, the adapter must use ``message.quote.text``
(the user-selected substring) rather than ``message.reply_to_message.text``
(the entire replied-to message). Otherwise the agent receives the full prior
message as ``reply_to_text``, which can cause it to act on unrelated
actionable-looking text the user did not quote (#22619).
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


def _make_adapter():
    return TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))


def _make_message(
    text="follow-up",
    reply_to_text=None,
    reply_to_caption=None,
    reply_to_id=42,
    quote_text=None,
):
    chat = SimpleNamespace(id=111, type="private", title=None, full_name="Alice")
    user = SimpleNamespace(id=42, full_name="Alice")

    reply_to_message = None
    if reply_to_text is not None or reply_to_caption is not None:
        reply_to_message = SimpleNamespace(
            message_id=reply_to_id,
            text=reply_to_text,
            caption=reply_to_caption,
        )

    quote = None
    if quote_text is not None:
        quote = SimpleNamespace(text=quote_text)

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
    from gateway.platforms.base import MessageType

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


def test_full_reply_text_used_when_no_native_quote():
    """No ``message.quote`` → fall back to the whole replied-to message text."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    msg = _make_message(
        text="thanks",
        reply_to_text="Whole prior message body",
        quote_text=None,
    )

    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "Whole prior message body"
    assert event.reply_to_message_id == "42"


def test_caption_fallback_when_no_quote_and_no_text():
    """Replied-to media message: caption is used when text is absent."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    msg = _make_message(
        text="see this",
        reply_to_text=None,
        reply_to_caption="Photo caption from earlier",
        quote_text=None,
    )

    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "Photo caption from earlier"


def test_empty_quote_text_falls_back_to_full_reply():
    """Defensive: a present-but-empty quote.text shouldn't blank the prefix."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    msg = _make_message(
        text="follow-up",
        reply_to_text="Prior message body",
        quote_text="",
    )

    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "Prior message body"
