"""Telegram forum-topic title ingestion tests."""

from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


def _source(*, thread_id: str = "42") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_name="Dev",
        chat_type="group",
        user_id="7",
        user_name="tester",
        thread_id=thread_id,
    )


def _update(*, created_name=None, edited_name=None):
    return SimpleNamespace(
        update_id=99,
        effective_message=SimpleNamespace(
            from_user=SimpleNamespace(id=7, is_bot=False),
            forum_topic_created=(
                SimpleNamespace(name=created_name) if created_name is not None else None
            ),
            forum_topic_edited=(
                SimpleNamespace(name=edited_name) if edited_name is not None else None
            ),
        )
    )


def _adapter(event: MessageEvent) -> Any:
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._build_message_event = MagicMock(return_value=event)
    adapter._is_user_authorized_from_message = MagicMock(return_value=True)
    adapter.handle_message = AsyncMock()
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("update", "name"),
    [
        (_update(created_name="Hermes: updates"), "Hermes: updates"),
        (_update(edited_name="Hermes: releases"), "Hermes: releases"),
    ],
)
async def test_forum_topic_name_update_is_forwarded_as_metadata(update, name):
    event = MessageEvent(text="", source=_source(), message_id="1")
    adapter = _adapter(event)

    await adapter._handle_forum_topic_name_update(update, SimpleNamespace())

    adapter.handle_message.assert_awaited_once_with(event)
    assert event.text == ""
    assert event.allow_gateway_control is False
    assert event.source.chat_topic == name
    assert event.metadata == {"telegram_forum_topic_name": name}


@pytest.mark.asyncio
async def test_forum_topic_edit_without_name_is_ignored():
    event = MessageEvent(text="", source=_source(), message_id="1")
    adapter = _adapter(event)
    update = SimpleNamespace(
        effective_message=SimpleNamespace(
            from_user=SimpleNamespace(id=7, is_bot=False),
            forum_topic_created=None,
            forum_topic_edited=SimpleNamespace(name=None),
        )
    )

    await adapter._handle_forum_topic_name_update(update, SimpleNamespace())

    update.effective_message.forum_topic_edited.name = "Automatic title"
    update.effective_message.from_user.is_bot = True
    await adapter._handle_forum_topic_name_update(update, SimpleNamespace())

    adapter._build_message_event.assert_not_called()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_persists_created_and_edited_names_on_the_same_session():
    from gateway.run import GatewayRunner

    source = _source()
    entry = SessionEntry(
        session_key="agent:main:telegram:group:-100123:42",
        session_id="session-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
        origin=source,
    )
    runner: Any = object.__new__(GatewayRunner)
    store = object()
    get_or_create_session = AsyncMock(return_value=entry)
    runner.session_store = store
    runner._async_session_store = SimpleNamespace(
        _store=store,
        get_or_create_session=get_or_create_session,
    )
    set_session_title = AsyncMock(return_value=True)
    runner._session_db = SimpleNamespace(set_session_title=set_session_title)

    for name in ("Hermes: updates", "Hermes: releases"):
        event = MessageEvent(
            text="",
            source=source,
            metadata={"telegram_forum_topic_name": name},
        )
        assert await runner._handle_telegram_forum_topic_name(event) is True

    assert get_or_create_session.await_count == 2
    for call in get_or_create_session.await_args_list:
        assert call.args == (source,)
        assert call.kwargs == {"touch_activity": False}
    assert set_session_title.await_args_list[0].args == (
        "session-1",
        "Hermes: updates",
    )
    assert set_session_title.await_args_list[1].args == (
        "session-1",
        "Hermes: releases",
    )


@pytest.mark.asyncio
async def test_gateway_ignores_non_topic_messages():
    from gateway.run import GatewayRunner

    runner: Any = object.__new__(GatewayRunner)
    store = object()
    get_or_create_session = AsyncMock()
    runner.session_store = store
    runner._async_session_store = SimpleNamespace(
        _store=store,
        get_or_create_session=get_or_create_session,
    )
    set_session_title = AsyncMock()
    runner._session_db = SimpleNamespace(set_session_title=set_session_title)
    event = MessageEvent(text="hello", source=_source(), metadata={})

    assert await runner._handle_telegram_forum_topic_name(event) is False
    get_or_create_session.assert_not_awaited()
    set_session_title.assert_not_awaited()
