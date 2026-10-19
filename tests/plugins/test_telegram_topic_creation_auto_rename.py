"""User-created Telegram topics stay eligible for session-title renaming (Refs #136567).

Exercise real PTB dispatch and adapter lookup; only Telegram's outbound rename is mocked.
"""

from unittest.mock import AsyncMock

import pytest

pytest.importorskip("telegram")

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import build_session_key
from hermes_state import AsyncSessionDB, SessionDB
from tests.plugins.test_telegram_forum_topic_created_opener import (
    _deliver,
    connected,
    opener,
    topic_created,
)


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_telegram_topic_migration()
    try:
        yield db
    finally:
        db.close()


def _make_runner(adapter, source, db):
    from gateway.run import GatewayRunner

    session_id = "sess-created-topic"
    db.create_session(session_id, source="telegram", user_id=source.user_id)
    db.bind_telegram_topic(
        chat_id=source.chat_id,
        thread_id=source.thread_id,
        user_id=source.user_id,
        session_key=build_session_key(source),
        session_id=session_id,
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._session_db = AsyncSessionDB(db)
    runner._telegram_topic_mode_enabled = lambda source: True
    return runner, session_id


@pytest.mark.asyncio
async def test_user_created_topic_opener_allows_auto_title_rename(monkeypatch, session_db):
    async with connected(monkeypatch) as (adapter, app, delivered):
        await _deliver(
            app, adapter,
            topic_created(app.bot, name="New Chat"),
            opener(app.bot),
        )
        assert len(delivered) == 1
        source = delivered[0].source
        assert source.thread_id == "40798"

        rename = AsyncMock()
        monkeypatch.setattr(adapter, "rename_dm_topic", rename)
        runner, session_id = _make_runner(adapter, source, session_db)
        assert runner._delivery_adapter_for(source) is adapter
        assert runner._is_telegram_topic_lane(source)

        await runner._rename_telegram_topic_for_session_title(source, session_id, "Auto Title")

        rename.assert_awaited_once_with(
            chat_id="42", thread_id="40798", name="Auto Title",
        )


@pytest.mark.asyncio
async def test_operator_fixed_topic_name_is_not_auto_renamed(monkeypatch, session_db):
    async with connected(monkeypatch) as (adapter, app, delivered):
        # Model the state loaded from extra.dm_topics, without creating a remote topic.
        fixed_topic = {"name": "Fixed", "thread_id": 40798}
        adapter._dm_topics_config = [{"chat_id": 42, "topics": [fixed_topic]}]
        adapter._dm_topics["42:Fixed"] = 40798
        await _deliver(
            app, adapter,
            topic_created(app.bot, name="Fixed"),
            opener(app.bot),
        )
        assert len(delivered) == 1
        source = delivered[0].source
        assert source.thread_id == "40798"
        assert adapter._get_dm_topic_info("42", "40798") == fixed_topic

        rename = AsyncMock()
        monkeypatch.setattr(adapter, "rename_dm_topic", rename)
        runner, session_id = _make_runner(adapter, source, session_db)
        assert runner._delivery_adapter_for(source) is adapter
        assert runner._is_telegram_topic_lane(source)

        await runner._rename_telegram_topic_for_session_title(source, session_id, "Auto Title")

        rename.assert_not_called()


@pytest.mark.asyncio
async def test_user_creation_remains_pending_without_operator_name_cache(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, delivered):
        await _deliver(app, adapter, topic_created(app.bot, name="New Chat"))

        assert delivered == []
        assert set(adapter._pending_created_topics) == {("42", "88")}
        assert adapter._pending_created_topics[("42", "88")][:2] == ("40798", 40799)
        assert "42:New Chat" not in adapter._dm_topics
