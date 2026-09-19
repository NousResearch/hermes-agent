from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_name="Dev",
        chat_type="group",
        user_id="7",
        user_name="tester",
        thread_id="42",
    )


def _update(*, created_name=None, edited_name=None, is_bot=False):
    return SimpleNamespace(
        update_id=99,
        effective_message=SimpleNamespace(
            from_user=SimpleNamespace(id=7, is_bot=is_bot),
            forum_topic_created=(
                SimpleNamespace(name=created_name) if created_name is not None else None
            ),
            forum_topic_edited=(
                SimpleNamespace(name=edited_name) if edited_name is not None else None
            ),
        ),
    )


@pytest.mark.asyncio
async def test_authorized_topic_create_and_edit_update_one_canonical_session_title():
    from gateway.run import GatewayRunner
    from plugins.platforms.telegram.adapter import TelegramAdapter

    source = _source()
    event = MessageEvent(text="", source=source, message_id="1")
    adapter: Any = object.__new__(TelegramAdapter)
    adapter._build_message_event = MagicMock(return_value=event)
    adapter._is_user_authorized_from_message = MagicMock(return_value=True)
    adapter.handle_message = AsyncMock()

    await adapter._handle_forum_topic_name_update(
        _update(created_name="Hermes: updates"), SimpleNamespace()
    )

    adapter.handle_message.assert_awaited_once_with(event)
    assert event.text == ""
    assert event.allow_gateway_control is False
    assert event.source.chat_topic == "Hermes: updates"
    assert event.metadata == {"telegram_forum_topic_name": "Hermes: updates"}

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
    get_or_create = AsyncMock(return_value=entry)
    runner.session_store = store
    runner._async_session_store = SimpleNamespace(
        _store=store,
        get_or_create_session=get_or_create,
    )
    set_title = AsyncMock(return_value=True)
    runner._session_db = SimpleNamespace(set_session_title=set_title)

    assert await runner._handle_telegram_forum_topic_name(event) is True
    event.metadata["telegram_forum_topic_name"] = "Hermes: releases"
    assert await runner._handle_telegram_forum_topic_name(event) is True

    assert [call.args for call in set_title.await_args_list] == [
        ("session-1", "Hermes: updates"),
        ("session-1", "Hermes: releases"),
    ]
    assert all(
        call.kwargs == {"touch_activity": False}
        for call in get_or_create.await_args_list
    )

    adapter.handle_message.reset_mock()
    await adapter._handle_forum_topic_name_update(
        _update(edited_name="Automatic title", is_bot=True), SimpleNamespace()
    )
    await adapter._handle_forum_topic_name_update(
        _update(edited_name=""), SimpleNamespace()
    )
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_topic_service_update_is_consumed_after_authorization_before_turn_start():
    from gateway.run import GatewayRunner

    event = MessageEvent(
        text="",
        source=_source(),
        metadata={"telegram_forum_topic_name": "Hermes: updates"},
    )
    runner: Any = object.__new__(GatewayRunner)
    runner._hm_admit_event = AsyncMock(return_value=(event, event.source, False))
    runner._handle_telegram_forum_topic_name = AsyncMock(return_value=True)
    runner._hm_estop_gate = MagicMock(side_effect=AssertionError("turn path reached"))

    assert await runner._handle_message(event) is None
    runner._hm_admit_event.assert_awaited_once_with(event)
    runner._handle_telegram_forum_topic_name.assert_awaited_once_with(event)
    runner._hm_estop_gate.assert_not_called()
