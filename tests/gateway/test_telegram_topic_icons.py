"""Telegram forum-topic icon selection (gateway/topic_icons.py) and its wiring into the auto-title
rename lane: opt-in via ``extra.topic_auto_icon``, icon comes only from Telegram's catalog, and any
miss degrades to a plain rename."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.topic_icons import TopicIconCatalog, choose_topic_icon
from hermes_state import SessionDB

CATALOG = [
    {"emoji": "💻", "custom_emoji_id": "5350554349074391003"},
    {"emoji": "⚡️", "custom_emoji_id": "5312016608254762256"},
    {"emoji": "💬", "custom_emoji_id": "5417915203100613993"},
]


def _llm_reply(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_catalog_lookup_ignores_variation_selectors():
    catalog = TopicIconCatalog()
    catalog.load(CATALOG)
    assert catalog.lookup("⚡") == catalog.lookup("⚡️") == "5312016608254762256"


def test_choose_topic_icon_rejects_emoji_outside_catalog():
    catalog = TopicIconCatalog()
    catalog.load(CATALOG)
    with patch("gateway.topic_icons.call_llm", return_value=_llm_reply('{"emoji": "🐘"}')):
        assert choose_topic_icon("Postgres pool exhaustion", catalog) is None
    with patch("gateway.topic_icons.call_llm", return_value=_llm_reply('```json\n{"emoji": "💻"}\n```')):
        assert choose_topic_icon("Fix driver config", catalog) == "5350554349074391003"


def test_choose_topic_icon_swallows_aux_failure():
    catalog = TopicIconCatalog()
    catalog.load(CATALOG)
    with patch("gateway.topic_icons.call_llm", side_effect=RuntimeError("aux down")):
        assert choose_topic_icon("anything", catalog) is None


def _make_runner(db, extra=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***", extra=extra or {})})
    adapter = MagicMock()
    adapter._bot = None
    adapter.rename_dm_topic = AsyncMock()
    adapter.fetch_forum_topic_icon_stickers = AsyncMock(return_value=CATALOG)
    runner.adapters = {Platform.TELEGRAM: adapter}
    from hermes_state import AsyncSessionDB
    runner._session_db = AsyncSessionDB(db)
    runner._telegram_topic_mode_enabled = lambda source: True
    runner._delivery_adapter_for = lambda source: adapter
    return runner, adapter


def _bound_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_telegram_topic_migration()
    db.create_session("sess-topic", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988", thread_id="42", user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:42", session_id="sess-topic",
    )
    return db


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="208214988", chat_type="dm", user_id="208214988", thread_id="42")


@pytest.mark.asyncio
async def test_rename_lane_sets_catalog_icon_when_opted_in(tmp_path):
    runner, adapter = _make_runner(_bound_db(tmp_path), extra={"topic_auto_icon": True})
    with patch("gateway.topic_icons.call_llm", return_value=_llm_reply('{"emoji": "💻"}')):
        await runner._rename_telegram_topic_for_session_title(_source(), "sess-topic", "Fix driver config")
    adapter.rename_dm_topic.assert_awaited_once_with(
        chat_id="208214988", thread_id="42", name="Fix driver config", icon_custom_emoji_id="5350554349074391003",
    )


@pytest.mark.asyncio
async def test_rename_lane_is_plain_rename_without_opt_in(tmp_path):
    runner, adapter = _make_runner(_bound_db(tmp_path))
    with patch("gateway.topic_icons.call_llm") as llm:
        await runner._rename_telegram_topic_for_session_title(_source(), "sess-topic", "Fix driver config")
    llm.assert_not_called()
    adapter.fetch_forum_topic_icon_stickers.assert_not_awaited()
    adapter.rename_dm_topic.assert_awaited_once_with(chat_id="208214988", thread_id="42", name="Fix driver config")


@pytest.mark.asyncio
async def test_rename_lane_still_renames_when_icon_lookup_fails(tmp_path):
    runner, adapter = _make_runner(_bound_db(tmp_path), extra={"topic_auto_icon": True})
    adapter.fetch_forum_topic_icon_stickers = AsyncMock(side_effect=RuntimeError("telegram down"))
    await runner._rename_telegram_topic_for_session_title(_source(), "sess-topic", "Fix driver config")
    adapter.rename_dm_topic.assert_awaited_once_with(chat_id="208214988", thread_id="42", name="Fix driver config")
