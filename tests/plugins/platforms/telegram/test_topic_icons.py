"""Semantic icon selection for auto-renamed Telegram DM topics.

Contracts under test:
- the rename lane attaches a semantic icon (from the title, via Telegram's
  fixed forum-icon set) to the same ``editForumTopic`` call as the name —
  no second API call, no extra model call;
- an icon rejected by the API (sticker set mismatch, transient failure)
  falls back to a name-only rename: the semantic name must never be lost;
- the keyword mapping projects the TITLE's emphasis (first matched title
  word wins — "Home lights" icons home even though a preference order would
  rank other faces first);
- a title with no eligible icon face leaves the icon untouched
  (``icon_custom_emoji_id`` omitted, not an empty string);
- the allowed-set fetch is cached per index (one
  ``getForumTopicIconStickers`` call) and a failed fetch degrades to
  name-only renames under a retry budget.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.telegram.topic_icons import (  # noqa: E402
    ForumTopicIconIndex,
    select_topic_icon,
    suggest_topic_icon_emoji,
)


def _sticker(face: str, custom_emoji_id: str) -> SimpleNamespace:
    return SimpleNamespace(emoji=face, custom_emoji_id=custom_emoji_id)


_ALLOWED_SET = (
    _sticker("🏠", "540210001"),
    _sticker("📷", "540210002"),
    _sticker("💼", "540210003"),
    _sticker("⚡️", "540210004"),
    _sticker("🔧", "540210005"),
    _sticker("✅", "540210006"),
    _sticker("⚠️", "540210007"),
)


def _fake_bot(stickers=_ALLOWED_SET, *, fail_fetch: bool = False) -> MagicMock:
    bot = MagicMock()
    bot.get_forum_topic_icon_stickers = AsyncMock(
        side_effect=RuntimeError("network down") if fail_fetch else None,
        return_value=stickers,
    )
    return bot


# ── pure keyword projection ────────────────────────────────────────────────


@pytest.mark.parametrize("title,expected", [
    ("Home renovation photos", "🏠"),
    ("Home lights automation fix", "🏠"),  # leading title concept wins
    ("Fix flaky auth test", "🔧"),
    ("Fix login button on mobile", "🔧"),
    ("Trip planning", "✈️"),
    ("Greeting", "👋"),
    ("Quarterly review sync", "👀"),
    ("Arbitrage hedge fund memo", None),  # no face mapped: nothing eligible
    ("", None),
])
def test_suggest_topic_icon_emoji_projects_title_semantics(title, expected):
    assert suggest_topic_icon_emoji(title) == expected


def test_first_title_word_wins_over_icon_preference():
    # The title's word order is the user's emphasis: a keyword preference
    # order would let a trailing mapped word steal the icon from the leading
    # concept the conversation is actually about.
    assert suggest_topic_icon_emoji("Work dinner photos") == "💼"


# ── index fetch + resolve ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_index_fetches_allowed_set_once_and_resolves():
    bot = _fake_bot()
    index = ForumTopicIconIndex()
    icon = await select_topic_icon("Home renovation photos", bot, index)
    assert icon == "540210001"
    await select_topic_icon("Fix login bug", bot, index)
    bot.get_forum_topic_icon_stickers.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_failure_returns_none_without_blocking_rename():
    bot = _fake_bot(fail_fetch=True)
    index = ForumTopicIconIndex()
    assert await select_topic_icon("Home renovation photos", bot, index) is None
    # Retry budget: an immediate second call does not hammer the failing API.
    await select_topic_icon("Fix login bug", bot, index)
    bot.get_forum_topic_icon_stickers.assert_awaited_once()


@pytest.mark.asyncio
async def test_face_missing_from_allowed_set_is_skipped():
    # The Bot API only allows icons from the fixed set: a face Telegram does
    # not offer must resolve to None (icon left unchanged), never a guess.
    bot = _fake_bot(stickers=(_sticker("🏠", "540210001"),))
    index = ForumTopicIconIndex()
    assert await select_topic_icon("Trip planning", bot, index) is None


@pytest.mark.asyncio
async def test_no_bot_means_no_icon():
    assert await select_topic_icon("Home photos", None, ForumTopicIconIndex()) is None


# ── adapter rename with icon fallback ─────────────────────────────────────


def _telegram_adapter():
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter
    return TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))


@pytest.mark.asyncio
async def test_rename_dm_topic_passes_icon_in_same_edit_call():
    adapter = _telegram_adapter()
    adapter._bot = MagicMock()
    adapter._bot.edit_forum_topic = AsyncMock(return_value=True)
    ok = await adapter.rename_dm_topic(
        chat_id=123, thread_id=42, name="Home renovation", icon_custom_emoji_id="540210001")
    assert ok is True
    adapter._bot.edit_forum_topic.assert_awaited_once_with(
        chat_id=123, message_thread_id=42, name="Home renovation", icon_custom_emoji_id="540210001")


@pytest.mark.asyncio
async def test_rename_dm_topic_none_icon_keeps_icon_unchanged():
    # None means "icon unchanged" (PTB drops None fields server-side); an
    # empty string would REMOVE the icon, which the lane must never send.
    adapter = _telegram_adapter()
    adapter._bot = MagicMock()
    adapter._bot.edit_forum_topic = AsyncMock(return_value=True)
    await adapter.rename_dm_topic(chat_id=123, thread_id=42, name="Greeting", icon_custom_emoji_id=None)
    adapter._bot.edit_forum_topic.assert_awaited_once_with(
        chat_id=123, message_thread_id=42, name="Greeting", icon_custom_emoji_id=None)


@pytest.mark.asyncio
async def test_rename_dm_topic_icon_failure_falls_back_to_name_only():
    # A rejected custom-emoji id (sticker set drift, transient API failure)
    # must not lose the semantic rename.
    adapter = _telegram_adapter()
    adapter._bot = MagicMock()
    calls = []

    async def _edit(**kwargs):
        calls.append(kwargs)
        if kwargs.get("icon_custom_emoji_id"):
            raise RuntimeError("Bad Request: STICKERSET_INVALID")
        return True

    adapter._bot.edit_forum_topic = AsyncMock(side_effect=_edit)
    ok = await adapter.rename_dm_topic(
        chat_id=123, thread_id=42, name="Home renovation", icon_custom_emoji_id="999999999")
    assert ok is True
    assert len(calls) == 2
    assert calls[0]["icon_custom_emoji_id"] == "999999999"
    assert calls[1].get("icon_custom_emoji_id") is None


@pytest.mark.asyncio
async def test_rename_dm_topic_name_failure_without_icon_raises():
    adapter = _telegram_adapter()
    adapter._bot = MagicMock()
    adapter._bot.edit_forum_topic = AsyncMock(side_effect=RuntimeError("chat not found"))
    with pytest.raises(RuntimeError):
        await adapter.rename_dm_topic(chat_id=123, thread_id=42, name="Whatever", icon_custom_emoji_id=None)


# ── user-icon cache from service messages ─────────────────────────────────


class _ServiceMessage:
    """Minimal message shape for _resolve_topic_binding's service branches."""

    def __init__(self, chat_id, **service):
        self.chat = SimpleNamespace(id=chat_id)
        for key, value in service.items():
            setattr(self, key, value)


def test_user_icon_cached_from_topic_created_service_message():
    adapter = _telegram_adapter()
    message = _ServiceMessage(
        123, forum_topic_created=SimpleNamespace(name="My stuff", icon_custom_emoji_id="5317777777777"))
    adapter._resolve_topic_binding(message, "dm", "42")
    assert adapter.get_dm_topic_user_icon("123", "42") == "5317777777777"


def test_created_service_message_without_icon_records_cleared_icon():
    # "" means the user created the topic with the default (no) icon: the lane
    # must never auto-add one over an explicit no-icon choice it has seen.
    adapter = _telegram_adapter()
    message = _ServiceMessage(123, forum_topic_created=SimpleNamespace(name="My stuff", icon_custom_emoji_id=None))
    adapter._resolve_topic_binding(message, "dm", "42")
    assert adapter.get_dm_topic_user_icon("123", "42") == ""


def test_edited_service_message_with_icon_caches_user_choice():
    adapter = _telegram_adapter()
    message = _ServiceMessage(123, forum_topic_edited=SimpleNamespace(name="Renamed", icon_custom_emoji_id="5318888888888"))
    adapter._resolve_topic_binding(message, "dm", "42")
    assert adapter.get_dm_topic_user_icon("123", "42") == "5318888888888"


def test_edited_service_message_without_icon_does_not_mark_cleared():
    # An icon-less edit means "icon unchanged", not "cleared": it must not
    # turn an unknown icon into a fake "user cleared it" observation.
    adapter = _telegram_adapter()
    message = _ServiceMessage(123, forum_topic_edited=SimpleNamespace(name="Renamed", icon_custom_emoji_id=None))
    adapter._resolve_topic_binding(message, "dm", "42")
    assert adapter.get_dm_topic_user_icon("123", "42") is None


def test_user_icon_unknown_for_unseen_topic():
    adapter = _telegram_adapter()
    assert adapter.get_dm_topic_user_icon("123", "42") is None
    assert adapter.get_dm_topic_user_icon("123", None) is None


@pytest.mark.asyncio
async def test_rename_dm_topic_no_bot_returns_false():
    adapter = _telegram_adapter()
    adapter._bot = None
    assert await adapter.rename_dm_topic(123, 42, "name") is False
