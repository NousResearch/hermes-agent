"""A Telegram topic is named once, by its first session.

An idle reset or a Kanban wake opens a new session in an existing topic; its auto-title must not rename
the topic (James's topics kept showing whatever the latest reply was about).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from hermes_state import AsyncSessionDB, SessionDB


def _runner(db):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    adapter = MagicMock()
    adapter.rename_dm_topic = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._session_db = AsyncSessionDB(db)
    runner._telegram_topic_mode_enabled = lambda source: True
    runner._delivery_adapter_for = lambda source: adapter
    return runner, adapter


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="700", chat_type="dm", user_id="700", thread_id="42")


def _db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_telegram_topic_migration()
    return db


def _bind(db, session_id):
    db.bind_telegram_topic(chat_id="700", thread_id="42", user_id="700",
                           session_key="agent:main:telegram:dm:700:42", session_id=session_id)


@pytest.mark.asyncio
async def test_first_session_names_a_new_topic(tmp_path):
    db = _db(tmp_path)
    db.create_session("first", source="telegram", chat_id="700", chat_type="dm", thread_id="42")
    _bind(db, "first")
    runner, adapter = _runner(db)

    await runner._rename_telegram_topic_for_session_title(_source(), "first", "OKX integration requirements review")

    adapter.rename_dm_topic.assert_awaited_once_with(chat_id="700", thread_id="42",
                                                     name="OKX integration requirements review")


@pytest.mark.asyncio
async def test_second_session_in_a_named_topic_does_not_rename_it(tmp_path):
    db = _db(tmp_path)
    db.create_session("first", source="telegram", chat_id="700", chat_type="dm", thread_id="42")
    db.set_auto_title("first", "OKX integration requirements review", source="llm")
    db.create_session("later", source="telegram", chat_id="700", chat_type="dm", thread_id="42")
    _bind(db, "later")  # the idle reset rebinds the topic to the new session
    runner, adapter = _runner(db)

    await runner._rename_telegram_topic_for_session_title(
        _source(), "later", "OKX integration requirements review · OKX 82000 delisting rule questions")

    adapter.rename_dm_topic.assert_not_awaited()
