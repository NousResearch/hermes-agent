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

from plugins.platforms.telegram.adapter import _is_peer_bot_user

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
