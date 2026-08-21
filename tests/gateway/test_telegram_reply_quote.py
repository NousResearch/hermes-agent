"""Tests for Telegram native partial-quote handling in _build_message_event.

When a Telegram user replies using Telegram's native quote feature to
select only part of a prior message, the adapter must use ``message.quote.text``
(the user-selected substring) rather than ``message.reply_to_message.text``
(the entire replied-to message). Otherwise the agent receives the full prior
message as ``reply_to_text``, which can cause it to act on unrelated
actionable-looking text the user did not quote (#22619).
"""

from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


BOT_ID = 777
USER_ID = 42


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))
    adapter._bot = SimpleNamespace(id=BOT_ID)
    return adapter


def _make_message(
    text="follow-up",
    reply_to_text=None,
    reply_to_caption=None,
    reply_to_id=42,
    quote_text=None,
    chat_type="private",
    reply_author_id=BOT_ID,
    reply_author_name="Hermes",
    reply_author_is_bot=True,
):
    is_group = chat_type in {"group", "supergroup"}
    chat = SimpleNamespace(
        id=-100111 if is_group else 111,
        type=chat_type,
        title="Test group" if is_group else None,
        full_name=None if is_group else "Alice",
    )
    user = SimpleNamespace(id=USER_ID, full_name="Alice", is_bot=False)

    reply_to_message = None
    if reply_to_text is not None or reply_to_caption is not None:
        reply_to_message = SimpleNamespace(
            message_id=reply_to_id,
            text=reply_to_text,
            caption=reply_to_caption,
            from_user=SimpleNamespace(
                id=reply_author_id,
                full_name=reply_author_name,
                is_bot=reply_author_is_bot,
            ),
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


@pytest.mark.parametrize("chat_type", ["private", "supergroup"])
@pytest.mark.parametrize(
    ("reply_author_id", "reply_author_name", "reply_author_is_bot", "expected_own"),
    [
        (BOT_ID, "Hermes", True, True),
        (USER_ID, "Alice", False, False),
    ],
)
def test_partial_quote_preserves_selection_and_author_across_chat_types(
    chat_type,
    reply_author_id,
    reply_author_name,
    reply_author_is_bot,
    expected_own,
):
    """Telegram quote semantics must not depend on chat type or quoted author."""
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    msg = _make_message(
        text="please address this part",
        reply_to_text="alpha beta gamma",
        quote_text="beta",
        chat_type=chat_type,
        reply_author_id=reply_author_id,
        reply_author_name=reply_author_name,
        reply_author_is_bot=reply_author_is_bot,
    )

    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.reply_to_text == "beta"
    assert event.reply_to_author_id == str(reply_author_id)
    assert event.reply_to_author_name == reply_author_name
    assert event.reply_to_is_own_message is expected_own
    assert event.reply_to_is_partial_quote is True

