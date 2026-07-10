"""Tests for internal <TOOLCALL> markup redaction in reply_to_text (#61217).

When a user replies to a bot message whose content carries raw internal
tool-call markup (``<TOOLCALL>{...}</TOOLCALL>`` from the agent's output
stream), the Telegram adapter must redact the markup before exposing it as
``reply_to_text`` — otherwise the internal protocol detail leaks verbatim
into the user-facing reply preview and the agent's injected reply context.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.config import PlatformConfig

from agent.message_sanitization import strip_internal_markup


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

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


# ── strip_internal_markup unit tests ────────────────────────────────────


def test_strips_closed_toolcall_block():
    text = '<TOOLCALL>{"name": "terminal", "arguments": {}}</TOOLCALL>'
    result = strip_internal_markup(text)
    assert "<TOOLCALL>" not in result
    assert "</TOOLCALL>" not in result
    assert "[processing...]" in result


def test_strips_unclosed_toolcall_block():
    # Stored reply previews are truncated, so a block can be cut mid-JSON
    # with no closing tag — exactly the shape seen in the issue logs.
    text = '<TOOLCALL>{"name": "terminal", "arguments": {"command": "ls'
    result = strip_internal_markup(text)
    assert "<TOOLCALL>" not in result
    assert result == "[processing...]"


def test_strips_block_embedded_in_surrounding_text():
    text = 'Done!\n<TOOLCALL>{"name": "skill_view"}</TOOLCALL>\nHere is the result.'
    result = strip_internal_markup(text)
    assert "<TOOLCALL>" not in result
    assert "Done!" in result
    assert "Here is the result." in result


def test_strips_multiple_blocks():
    text = (
        '<TOOLCALL>{"name": "a"}</TOOLCALL> middle '
        '<TOOLCALL>{"name": "b"}</TOOLCALL>'
    )
    result = strip_internal_markup(text)
    assert "<TOOLCALL>" not in result
    assert "middle" in result


def test_strips_multiline_block():
    text = '<TOOLCALL>{\n  "name": "browser_press",\n  "arguments": {}\n}</TOOLCALL>'
    result = strip_internal_markup(text)
    assert "<TOOLCALL>" not in result
    assert result == "[processing...]"


def test_plain_text_returned_unchanged():
    text = "Just a normal reply with no markup.\n"
    assert strip_internal_markup(text) is text


def test_none_and_empty_pass_through():
    assert strip_internal_markup(None) is None
    assert strip_internal_markup("") == ""


# ── Telegram adapter integration ────────────────────────────────────────


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


def test_reply_to_text_with_toolcall_markup_is_redacted():
    """Raw <TOOLCALL> markup in the replied-to message must not survive
    into MessageEvent.reply_to_text."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    msg = _make_message(
        text="什么情况",
        reply_to_text='<TOOLCALL>{"name": "terminal", "arguments": {"command": "ls',
    )

    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert "<TOOLCALL>" not in (event.reply_to_text or "")
    assert event.reply_to_text == "[processing...]"
    assert event.reply_to_message_id == "42"


def test_reply_to_text_without_markup_is_untouched():
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    msg = _make_message(text="thanks", reply_to_text="Whole prior message body")

    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "Whole prior message body"


def test_native_quote_with_markup_is_redacted():
    """The markup can also arrive via a native partial quote of a bot
    message; the redaction covers every reply_to_text source."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    msg = _make_message(
        text="what is this?",
        reply_to_text="full body",
        quote_text='<TOOLCALL>{"name": "skill_view"}</TOOLCALL>',
    )

    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert "<TOOLCALL>" not in (event.reply_to_text or "")


def test_no_reply_context_stays_none():
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    msg = _make_message(text="hello", reply_to_text=None)

    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text is None
    assert event.reply_to_message_id is None
