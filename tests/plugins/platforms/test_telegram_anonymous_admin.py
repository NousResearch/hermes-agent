"""An anonymous group admin is a human author, not a peer bot.

When a group admin posts "as the group", Telegram's ``from_user`` is ``GroupAnonymousBot``: a
pseudo-user whose ``is_bot`` flag is True even though a person wrote the message. The gateway's
bot-loop guard keys on that flag, so it treated every anonymous-admin message as peer-bot traffic
and dropped it once the bot had replied in that chat — follow-ups from the group's admins went
unanswered.

``_is_peer_bot_user`` keeps the real-bot guard intact and excludes only that one id.
"""

from __future__ import annotations

from types import SimpleNamespace

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter, _is_peer_bot_user

ANONYMOUS_ADMIN_ID = 1087968824


def _user(user_id: int, is_bot: bool) -> SimpleNamespace:
    return SimpleNamespace(id=user_id, is_bot=is_bot)


def test_anonymous_group_admin_is_not_a_peer_bot():
    assert _is_peer_bot_user(_user(ANONYMOUS_ADMIN_ID, True)) is False


def test_real_bot_is_still_a_peer_bot():
    assert _is_peer_bot_user(_user(123456789, True)) is True


def test_human_is_not_a_peer_bot():
    assert _is_peer_bot_user(_user(123456789, False)) is False


def test_missing_author_is_not_a_peer_bot():
    assert _is_peer_bot_user(None) is False


def test_anonymous_admin_survives_bot_mention_gate_in_free_response_group(monkeypatch):
    """#156228: a group admin must bypass the *earlier* bot-to-bot trigger guard too.

    The source/event-level fix alone still drops an unmentioned anonymous admin when the
    supported bots_require_mention option is enabled in an otherwise free-response group.
    """
    # These gates read the env rung before config.extra, so a developer machine with a real
    # Telegram allowlist would otherwise judge this synthetic chat as not allowlisted.
    for var in ("TELEGRAM_ALLOWED_CHATS", "TELEGRAM_GROUP_ALLOWED_CHATS", "TELEGRAM_REQUIRE_MENTION",
                "TELEGRAM_BOTS_REQUIRE_MENTION", "TELEGRAM_FREE_RESPONSE_CHATS",
                "TELEGRAM_EXCLUSIVE_BOT_MENTIONS"):
        monkeypatch.delenv(var, raising=False)
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(
        enabled=True, token="test", extra={
            "require_mention": True,
            "bots_require_mention": True,
            "free_response_chats": ["-100"],
            "allowed_chats": ["-100"],
            "allowed_topics": [],
        },
    )
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")

    chat = SimpleNamespace(id=-100, type="group", title="Test Group", is_forum=False)

    def group_message(user_id: int):
        return SimpleNamespace(
            message_id=42,
            chat=chat,
            sender_chat=chat if user_id == ANONYMOUS_ADMIN_ID else None,
            from_user=_user(user_id, True),
            text="following up",
            caption=None,
            entities=[],
            caption_entities=[],
            message_thread_id=None,
            is_topic_message=False,
            reply_to_message=None,
        )

    assert adapter._should_process_message(group_message(ANONYMOUS_ADMIN_ID)) is True
    # A real peer bot without an explicit @mention must remain suppressed.
    assert adapter._should_process_message(group_message(123456789)) is False
