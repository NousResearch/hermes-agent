"""Tests for Telegram forwarded-message context extraction in _build_message_event.

Telegram PTB 22.6 exposes ``Message.forward_origin`` (one of MessageOriginUser,
MessageOriginHiddenUser, MessageOriginChat, MessageOriginChannel) and
``Message.is_automatic_forward``. The adapter must normalize these into a
platform-neutral ``MessageContextRef`` on the MessageEvent so the agent sees a
forward's provenance rather than treating it as plain pasted text (#43397).

Parsing is intentionally attribute-based (defensive ``getattr``) so these tests
can use lightweight ``SimpleNamespace`` stand-ins for PTB origin objects.
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

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _make_adapter():
    return TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))


def _make_message(
    text="forwarded body",
    forward_origin=None,
    is_automatic_forward=False,
):
    chat = SimpleNamespace(id=111, type="private", title=None, full_name="Alice")
    user = SimpleNamespace(id=42, full_name="Alice")
    return SimpleNamespace(
        chat=chat,
        from_user=user,
        text=text,
        message_thread_id=None,
        message_id=1001,
        reply_to_message=None,
        quote=None,
        date=None,
        forum_topic_created=None,
        forward_origin=forward_origin,
        is_automatic_forward=is_automatic_forward,
    )


def test_forward_from_user():
    """MessageOriginUser → origin name / user id / username, kind 'forward'."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    origin = SimpleNamespace(
        type="user",
        date=None,
        sender_user=SimpleNamespace(id=99, full_name="Bob Jones", username="bobj"),
    )
    msg = _make_message(text="check this out", forward_origin=origin)

    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert len(event.context_refs) == 1
    ref = event.context_refs[0]
    assert ref.kind == "forward"
    assert ref.platform == "telegram"
    assert ref.origin_type == "user"
    assert ref.origin_name == "Bob Jones"
    assert ref.origin_id == "99"
    assert ref.origin_username == "bobj"
    assert ref.is_confidence_limited is False
    # The forwarded text remains the message body.
    assert event.text == "check this out"


def test_forward_from_hidden_user():
    """MessageOriginHiddenUser → display name only, confidence-limited, no id."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    origin = SimpleNamespace(
        type="hidden_user",
        date=None,
        sender_user_name="Anonymous Coward",
    )
    msg = _make_message(forward_origin=origin)

    event = adapter._build_message_event(msg, MessageType.TEXT)

    ref = event.context_refs[0]
    assert ref.kind == "forward"
    assert ref.origin_type == "hidden_user"
    assert ref.origin_name == "Anonymous Coward"
    assert ref.origin_id is None
    assert ref.origin_username is None
    assert ref.is_confidence_limited is True


def test_forward_from_chat():
    """MessageOriginChat → chat title/id + author_signature."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    origin = SimpleNamespace(
        type="chat",
        date=None,
        sender_chat=SimpleNamespace(id=-100, title="Project Group", full_name=None),
        author_signature="Moderator",
    )
    msg = _make_message(forward_origin=origin)

    event = adapter._build_message_event(msg, MessageType.TEXT)

    ref = event.context_refs[0]
    assert ref.origin_type == "chat"
    assert ref.origin_chat == "Project Group"
    assert ref.origin_id == "-100"
    assert ref.origin_name == "Moderator"


def test_forward_from_channel():
    """MessageOriginChannel → channel title/id + original message_id + signature."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    origin = SimpleNamespace(
        type="channel",
        date=None,
        chat=SimpleNamespace(id=-1009, title="News Channel", full_name=None),
        message_id=555,
        author_signature="Editor",
    )
    msg = _make_message(forward_origin=origin)

    event = adapter._build_message_event(msg, MessageType.TEXT)

    ref = event.context_refs[0]
    assert ref.origin_type == "channel"
    assert ref.origin_chat == "News Channel"
    assert ref.origin_id == "-1009"
    assert ref.origin_message_id == "555"
    assert ref.origin_name == "Editor"


def test_automatic_forward_without_origin():
    """is_automatic_forward with no forward_origin → kind 'automatic_forward'."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    msg = _make_message(forward_origin=None, is_automatic_forward=True)

    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert len(event.context_refs) == 1
    ref = event.context_refs[0]
    assert ref.kind == "automatic_forward"
    assert ref.platform == "telegram"


def test_ordinary_message_has_no_context_refs():
    """A plain message (no forward, no auto-forward) gets no context refs."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    msg = _make_message(forward_origin=None, is_automatic_forward=False)

    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.context_refs == []
